"""
Edge Node 実装: 階層型連合学習の中核。

Global Server に対しては NumPyClient として振る舞い、
fit() が呼ばれると内部で flwr.server.start_server を起動して
配下の Leaf Client (+ Internal Client) から集約を行う。

■ 動作フロー (1 Global Round あたり)
1. Global Server が Edge の fit() を呼ぶ (パラメータ配布)
2. Edge 内部で sub-server を起動 (ブロッキング)
3. Internal Client を別スレッドで sub-server にループバック接続
4. 外部 Leaf Client も sub-server に接続して学習
5. sub-server が sub_rounds ラウンド分の FedAvg 集約
6. evaluate_fn callback で最終パラメータを capture
7. 集約結果を Global Server への fit() 戻り値として返却

NOTE: flwr.server.start_server は 1.13+ で deprecated だが、
      入れ子構造では新 SuperLink API が未対応のためレガシー API を使用。
"""

from __future__ import annotations

import threading
import time
from typing import Any

import flwr as fl
from flwr.common import NDArrays, Scalar, ndarrays_to_parameters
from flwr.server import ServerConfig

from src.core.client import HFLClient
from src.models.nets import create_model, get_parameters, set_parameters
from src.strategies.aggregation import create_strategy
from src.utils.config import resolve_device
from src.utils.logger import get_logger

logger = get_logger("edge")


class EdgeNode(fl.client.NumPyClient):
    """
    Edge Node: Global Server に対するクライアント兼、Leaf Client に対するサーバー。
    """

    def __init__(
        self,
        edge_config: dict[str, Any],
        global_config: dict[str, Any],
        topology_config: dict[str, Any],
        defaults_config: dict[str, Any],
    ) -> None:
        ec = edge_config["edge"]
        self.edge_id: str = ec["id"]
        self.sub_server_address: str = ec["sub_server_address"]
        self.sub_rounds: int = ec.get("sub_rounds", 3)
        self.sub_round_timeout: float = ec.get("sub_round_timeout", 120.0)
        self.min_fit_clients: int = ec.get("min_fit_clients", 1)
        self.min_available_clients: int = ec.get("min_available_clients", 1)
        self.contribution_factor: float = ec.get("contribution_factor", 1.0)
        self.internal_client_enabled: bool = ec.get("internal_client", {}).get("enabled", True)

        # モード
        mode = global_config.get("mode", "production")
        self.dry_run = mode == "dry_run"

        # デバイス
        device_cfg = edge_config.get("device", defaults_config.get("device", "auto"))
        self.device = resolve_device(device_cfg)

        # モデル設定
        model_cfg = global_config.get("model", {})
        self.model_name = model_cfg.get("name", "simplecnn")
        self.num_classes = model_cfg.get("num_classes", 10)
        ds_cfg = global_config.get("dataset", {})
        self.dataset_name = ds_cfg.get("name", "cifar10")
        in_channels = 1 if self.dataset_name == "mnist" else 3

        self.model = create_model(
            name=self.model_name, num_classes=self.num_classes,
            in_channels=in_channels, dry_run=self.dry_run,
        )

        # 学習設定 (Edge固有 > defaults)
        self.training_config = {
            **defaults_config.get("training", {}),
            **edge_config.get("training", {}),
        }

        # データ分割設定
        dp = topology_config.get("data_partition", {})
        self.partition_method = dp.get("method", "dirichlet")
        self.partition_params = dp.get("params", {})
        self.total_partitions = dp.get("total_partitions", 6)
        assignments = dp.get("assignments", {})

        self.internal_partition_id = assignments.get(f"{self.edge_id}_internal", 0)

        edges_cfg = topology_config.get("edges", {})
        edge_topo = edges_cfg.get(self.edge_id, {})
        self.leaf_ids: list[str] = edge_topo.get("leaf_clients", [])
        self.leaf_partition_ids: dict[str, int] = {
            lid: assignments.get(lid, 0) for lid in self.leaf_ids
        }

        # dry_run 設定
        self.dry_run_config = global_config.get("dry_run", {})
        if self.dry_run:
            self.sub_rounds = self.dry_run_config.get("num_rounds", 1)

        logger.info(
            f"[{self.edge_id}] Initialized "
            f"(sub_addr={self.sub_server_address}, sub_rounds={self.sub_rounds}, "
            f"leaves={self.leaf_ids}, internal={self.internal_client_enabled}, "
            f"dry_run={self.dry_run})"
        )

    # =====================================================================
    # Flower NumPyClient インターフェース
    # =====================================================================

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.model)

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Global Server から呼ばれる。内部 sub-federation を実行して集約結果を返す。"""
        set_parameters(self.model, parameters)
        logger.info(f"[{self.edge_id}] fit() called. Starting sub-federation...")

        result_params, num_examples, metrics = self._run_sub_federation(parameters)
        set_parameters(self.model, result_params)
        return result_params, num_examples, metrics

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Global Server からの評価要求。主要な評価は Global Server 側で実施。"""
        set_parameters(self.model, parameters)
        return 0.0, 0, {"edge_id": self.edge_id}

    # =====================================================================
    # Sub-federation ロジック
    # =====================================================================

    def _run_sub_federation(
        self, initial_parameters: NDArrays
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """サブ連合学習を実行し、集約結果を返す。"""
        initial_params_proto = ndarrays_to_parameters(initial_parameters)

        expected_clients = len(self.leaf_ids) + (1 if self.internal_client_enabled else 0)
        if expected_clients == 0:
            logger.warning(f"[{self.edge_id}] No clients configured.")
            return initial_parameters, 0, {}

        # パラメータ capture 用コンテナ
        captured: dict[str, Any] = {"params": initial_parameters}

        def capture_evaluate_fn(server_round, parameters_ndarrays, config):
            """各ラウンド終了時に集約後パラメータを capture する。"""
            captured["params"] = parameters_ndarrays
            return None

        sub_strategy = create_strategy(
            name="fedavg",
            initial_parameters=initial_params_proto,
            evaluate_fn=capture_evaluate_fn,
            fraction_fit=1.0,
            min_fit_clients=min(expected_clients, self.min_fit_clients),
            min_available_clients=min(expected_clients, self.min_available_clients),
        )

        # Internal Client をバックグラウンドスレッドで起動
        if self.internal_client_enabled:
            threading.Thread(target=self._run_internal_client, daemon=True).start()

        # Sub-server 起動 (ブロッキング: 全ラウンド完了まで)
        logger.info(
            f"[{self.edge_id}] Sub-server starting on {self.sub_server_address} "
            f"(expecting {expected_clients} clients, {self.sub_rounds} rounds)"
        )
        result = fl.server.start_server(
            server_address=self.sub_server_address,
            config=ServerConfig(
                num_rounds=self.sub_rounds,
                round_timeout=self.sub_round_timeout,
            ),
            strategy=sub_strategy,
        )

        elapsed = result[1] if isinstance(result, tuple) else 0.0
        final_params = captured["params"]

        total_examples = self._estimate_total_examples()
        reported_examples = max(int(total_examples * self.contribution_factor), 1)

        logger.info(
            f"[{self.edge_id}] Sub-federation complete "
            f"(elapsed={elapsed:.1f}s, reported_examples={reported_examples})"
        )
        return final_params, reported_examples, {"edge_id": self.edge_id}

    def _run_internal_client(self) -> None:
        """Internal Client を起動して sub-server にループバック接続する。"""
        logger.info(f"[{self.edge_id}] Starting internal client...")
        time.sleep(2.0)  # sub-server の起動を待機

        connect_address = self.sub_server_address.replace("0.0.0.0", "127.0.0.1")

        client = HFLClient(
            client_id=f"{self.edge_id}_internal",
            model_name=self.model_name,
            num_classes=self.num_classes,
            dataset_name=self.dataset_name,
            partition_id=self.internal_partition_id,
            total_partitions=self.total_partitions,
            partition_method=self.partition_method,
            partition_params=self.partition_params,
            training_config=self.training_config,
            device=self.device,
            dry_run=self.dry_run,
            dry_run_config=self.dry_run_config,
            contribution_factor=1.0,
        )

        try:
            fl.client.start_client(
                server_address=connect_address,
                client=client.to_client(),
                insecure=True,
            )
            logger.info(f"[{self.edge_id}] Internal client finished.")
        except Exception as e:
            logger.error(f"[{self.edge_id}] Internal client error: {e}")

    def _estimate_total_examples(self) -> int:
        """Edge配下の推定総データ件数。"""
        num_clients = len(self.leaf_ids) + (1 if self.internal_client_enabled else 0)
        return max(num_clients * 1000, 1)
