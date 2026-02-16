"""
Flower Client 実装。

Leaf Client と Edge Internal Client の両方がこのクラスを使用する。
差異は接続先アドレスとデータのみ。
"""

from __future__ import annotations

from typing import Any

import flwr as fl
from flwr.common import NDArrays, Scalar

from src.core.training import evaluate_model, train_one_epoch
from src.data.loader import create_dummy_dataloader, load_partition
from src.models.nets import create_model, get_parameters, set_parameters
from src.utils.config import resolve_device
from src.utils.logger import get_logger

logger = get_logger("client")


class HFLClient(fl.client.NumPyClient):
    """階層型連合学習のクライアント (Leaf / Edge Internal 共用)。"""

    def __init__(
        self,
        client_id: str,
        model_name: str = "simplecnn",
        num_classes: int = 10,
        dataset_name: str = "cifar10",
        partition_id: int = 0,
        total_partitions: int = 6,
        partition_method: str = "dirichlet",
        partition_params: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
        device: str = "auto",
        dry_run: bool = False,
        dry_run_config: dict[str, Any] | None = None,
        contribution_factor: float = 1.0,
    ) -> None:
        self.client_id = client_id
        self.device = resolve_device(device)
        self.dry_run = dry_run
        self.contribution_factor = contribution_factor

        # 学習ハイパーパラメータ
        tc = training_config or {}
        self.epochs = tc.get("epochs", 5)
        self.lr = tc.get("learning_rate", 0.01)
        self.optimizer_name = tc.get("optimizer", "sgd")
        self.momentum = tc.get("momentum", 0.9)
        self.weight_decay = tc.get("weight_decay", 0.0001)
        self.batch_size = tc.get("batch_size", 32)

        # dry_run オーバーライド
        self.max_steps: int | None = None
        if dry_run:
            drc = dry_run_config or {}
            self.epochs = 1
            self.max_steps = drc.get("dummy_steps", 2)

        # データセット名の情報 (チャネル数推定用)
        self.dataset_name = dataset_name
        in_channels = 1 if dataset_name == "mnist" else 3

        # モデル
        self.model = create_model(
            name=model_name,
            num_classes=num_classes,
            in_channels=in_channels,
            dry_run=dry_run,
        )

        # データローダー (遅延ロードも可能だが、PoC段階では即時ロード)
        if dry_run:
            drc = dry_run_config or {}
            self.trainloader = create_dummy_dataloader(
                batch_size=self.batch_size,
                num_samples=drc.get("dummy_data_size", 64),
                in_channels=in_channels,
                num_classes=num_classes,
            )
        else:
            self.trainloader = load_partition(
                dataset_name=dataset_name,
                partition_id=partition_id,
                total_partitions=total_partitions,
                partition_method=partition_method,
                partition_params=partition_params,
                batch_size=self.batch_size,
            )

        logger.info(
            f"[{self.client_id}] Initialized "
            f"(device={self.device}, dry_run={dry_run}, "
            f"partition={partition_id}/{total_partitions}, "
            f"data_size={len(self.trainloader.dataset)})"
        )

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.model)

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """ローカル学習を行い、パラメータと件数を返す。"""
        set_parameters(self.model, parameters)

        total_examples = 0
        total_loss = 0.0

        for epoch in range(self.epochs):
            loss, n = train_one_epoch(
                model=self.model,
                dataloader=self.trainloader,
                device=self.device,
                lr=self.lr,
                optimizer_name=self.optimizer_name,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                max_steps=self.max_steps,
            )
            total_loss += loss * n
            total_examples += n

        avg_loss = total_loss / max(total_examples, 1)

        # contribution_factor を適用した件数を報告
        reported_examples = int(total_examples * self.contribution_factor)
        reported_examples = max(reported_examples, 1)

        logger.info(
            f"[{self.client_id}] fit complete: "
            f"loss={avg_loss:.4f}, examples={total_examples}, "
            f"reported={reported_examples}"
        )

        return (
            get_parameters(self.model),
            reported_examples,
            {"loss": float(avg_loss)},
        )

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        """ローカル評価を行う（サーバーからの要求時）。"""
        set_parameters(self.model, parameters)
        loss, accuracy, num_examples = evaluate_model(
            model=self.model,
            dataloader=self.trainloader,  # PoC段階ではtrainデータで評価
            device=self.device,
        )
        return float(loss), num_examples, {"accuracy": float(accuracy)}
