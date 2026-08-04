"""Edge配下の常駐Flowerサーバーと親子ラウンド同期。"""

from __future__ import annotations

import queue
import time
import traceback
from dataclasses import dataclass
from multiprocessing.queues import Queue
from typing import Any

import flwr as fl
import grpc
from flwr.common import (
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerConfig
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from src.core.client import HFLClient
from src.core.training import evaluate_model
from src.data.loader import create_dummy_dataloader, load_test_data
from src.models.nets import copy_parameters, create_model, get_parameters, set_parameters
from src.strategies.aggregation import WeightedFedAvg
from src.utils.config import resolve_device
from src.utils.logger import get_logger

logger = get_logger("sub_federation")


def _start_client_with_retry(
    *,
    server_address: str,
    client: Any,
    startup_timeout: float,
    retry_interval: float = 1.0,
) -> None:
    """起動直後のUNAVAILABLEを許容してFlower Clientを接続する。"""
    deadline = time.monotonic() + startup_timeout
    attempt = 0
    while True:
        attempt += 1
        try:
            fl.client.start_client(
                server_address=server_address,
                client=client,
                insecure=True,
                max_retries=None,
            )
            return
        except grpc.RpcError as exc:
            remaining = deadline - time.monotonic()
            if exc.code() != grpc.StatusCode.UNAVAILABLE or remaining <= 0:
                raise
            logger.warning(
                "Flower server at %s is not ready; retrying in %.1fs "
                "(attempt=%d, remaining=%.1fs)",
                server_address,
                min(retry_interval, remaining),
                attempt,
                remaining,
            )
            time.sleep(min(retry_interval, remaining))


@dataclass
class ParentModelRequest:
    parent_round: int
    parameters: NDArrays


@dataclass
class ChildModelResult:
    parent_round: int
    child_round: int
    parameters: NDArrays
    num_examples: int
    metrics: dict[str, Scalar]


@dataclass
class WorkerEvent:
    worker: str
    kind: str
    message: str
    parent_round: int = 0
    child_round: int = 0
    connected_clients: int = 0
    expected_clients: int = 0


def parent_round_for_child(child_round: int, sub_rounds: int) -> int:
    if child_round < 1 or sub_rounds < 1:
        raise ValueError("child_round and sub_rounds must be positive")
    return ((child_round - 1) // sub_rounds) + 1


class SynchronousSubFedAvg(WeightedFedAvg):
    """親モデルを世代境界で受け取り、最終子結果を同じ親世代へ返す。"""

    def __init__(
        self,
        *,
        edge_id: str,
        sub_rounds: int,
        expected_clients: int,
        parent_queue: Queue,
        result_queue: Queue,
        event_queue: Queue,
        parent_model_timeout: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(accept_failures=False, **kwargs)
        self.edge_id = edge_id
        self.sub_rounds = sub_rounds
        self.expected_clients = expected_clients
        self.parent_queue = parent_queue
        self.result_queue = result_queue
        self.event_queue = event_queue
        self.parent_model_timeout = parent_model_timeout
        self.current_parent_round = 0
        self.current_parent_parameters: Parameters | None = None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        parent_round = parent_round_for_child(server_round, self.sub_rounds)
        connected = client_manager.num_available()
        first_child_round = (server_round - 1) % self.sub_rounds == 0

        if first_child_round:
            started = time.monotonic()
            try:
                request = self.parent_queue.get(timeout=self.parent_model_timeout)
            except queue.Empty as exc:
                waited = time.monotonic() - started
                message = (
                    f"[{self.edge_id}] Parent model timeout: edge_id={self.edge_id}, "
                    f"parent_round={parent_round}, child_round={server_round}, "
                    f"connected_clients={connected}, expected_clients={self.expected_clients}, "
                    f"waited={waited:.1f}s"
                )
                logger.error(message)
                raise TimeoutError(message) from exc

            if not isinstance(request, ParentModelRequest):
                raise TypeError(f"Unexpected parent queue message: {type(request)!r}")
            if request.parent_round != parent_round:
                raise RuntimeError(
                    f"[{self.edge_id}] Parent/child round mismatch: "
                    f"expected parent_round={parent_round}, got {request.parent_round}, "
                    f"child_round={server_round}"
                )
            if request.parent_round != self.current_parent_round + 1:
                raise RuntimeError(
                    f"[{self.edge_id}] Non-sequential parent round: "
                    f"previous={self.current_parent_round}, got={request.parent_round}"
                )
            self.current_parent_round = request.parent_round
            self.current_parent_parameters = ndarrays_to_parameters(
                copy_parameters(request.parameters)
            )
            parameters = self.current_parent_parameters
        elif parent_round != self.current_parent_round:
            raise RuntimeError(
                f"[{self.edge_id}] Child round {server_round} mapped to parent "
                f"{parent_round}, active parent is {self.current_parent_round}"
            )

        fit_config = {
            "parent_round": parent_round,
            "child_round": server_round,
        }
        sample_size, min_num_clients = self.num_fit_clients(connected)
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )
        self.event_queue.put(
            WorkerEvent(
                worker="sub_server",
                kind="round_started",
                message="Child round configured",
                parent_round=parent_round,
                child_round=server_round,
                connected_clients=connected,
                expected_clients=self.expected_clients,
            )
        )
        return [(client, FitIns(parameters, fit_config)) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        parent_round = parent_round_for_child(server_round, self.sub_rounds)
        if len(results) != self.expected_clients or failures:
            raise RuntimeError(
                f"[{self.edge_id}] Incomplete child round: parent_round={parent_round}, "
                f"child_round={server_round}, results={len(results)}, "
                f"expected={self.expected_clients}, failures={len(failures)}"
            )

        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            raise RuntimeError(
                f"[{self.edge_id}] No aggregate for child_round={server_round}"
            )

        if server_round % self.sub_rounds == 0:
            ndarrays = copy_parameters(parameters_to_ndarrays(aggregated))
            total_examples = sum(fit_res.num_examples for _, fit_res in results)
            self.result_queue.put(
                ChildModelResult(
                    parent_round=parent_round,
                    child_round=server_round,
                    parameters=ndarrays,
                    num_examples=total_examples,
                    metrics=dict(metrics),
                )
            )
            logger.info(
                f"[{self.edge_id}] Parent generation complete: "
                f"parent_round={parent_round}, child_round={server_round}, "
                f"results={len(results)}"
            )
        return aggregated, metrics


def run_sub_server(
    edge_config: dict[str, Any],
    global_config: dict[str, Any],
    defaults_config: dict[str, Any],
    parent_queue: Queue,
    result_queue: Queue,
    event_queue: Queue,
) -> None:
    """spawnされたプロセスのメインスレッドで常駐子サーバーを実行する。"""
    ec = edge_config["edge"]
    edge_id = ec["id"]
    try:
        dry_run = global_config.get("mode") == "dry_run"
        dry_cfg = global_config.get("dry_run", {})
        global_rounds = (
            int(dry_cfg.get("global_rounds", 2))
            if dry_run
            else int(global_config["server"]["num_rounds"])
        )
        sub_rounds = (
            int(dry_cfg.get("sub_rounds", ec.get("sub_rounds", 1)))
            if dry_run
            else int(ec.get("sub_rounds", 1))
        )
        expected_clients = int(ec["min_available_clients"])
        total_child_rounds = global_rounds * sub_rounds

        model_cfg = global_config.get("model", {})
        dataset_cfg = global_config.get("dataset", {})
        dataset_name = dataset_cfg.get("name", "cifar10")
        in_channels = 1 if dataset_name == "mnist" else 3
        model = create_model(
            model_cfg.get("name", "simplecnn"),
            num_classes=model_cfg.get("num_classes", 10),
            in_channels=in_channels,
            dry_run=dry_run,
        )
        device = resolve_device(
            edge_config.get("device", defaults_config.get("device", "auto"))
        )
        if dry_run:
            testloader = create_dummy_dataloader(
                batch_size=32,
                num_samples=dry_cfg.get("dummy_data_size", 64),
                in_channels=in_channels,
                num_classes=model_cfg.get("num_classes", 10),
            )
        else:
            testloader = load_test_data(
                dataset_name,
                batch_size=dataset_cfg.get("test_batch_size", 128),
            )

        def evaluate_fn(
            child_round: int, parameters: NDArrays, config: dict[str, Scalar]
        ) -> tuple[float, dict[str, Scalar]]:
            set_parameters(model, parameters)
            loss, accuracy, sample_count = evaluate_model(model, testloader, device)
            if child_round == 0:
                logger.info(
                    f"[{edge_id}] Initial sub-server evaluation: "
                    f"loss={loss:.4f}, accuracy={accuracy:.4f}, n={sample_count}"
                )
                return float(loss), {"accuracy": float(accuracy)}
            parent_round = parent_round_for_child(child_round, sub_rounds)
            logger.info(
                f"[{edge_id}] Sub-round evaluation: parent_round={parent_round}, "
                f"child_round={child_round}, loss={loss:.4f}, "
                f"accuracy={accuracy:.4f}, n={sample_count}"
            )
            return float(loss), {"accuracy": float(accuracy)}

        strategy = SynchronousSubFedAvg(
            edge_id=edge_id,
            sub_rounds=sub_rounds,
            expected_clients=expected_clients,
            parent_queue=parent_queue,
            result_queue=result_queue,
            event_queue=event_queue,
            parent_model_timeout=float(ec.get("parent_model_timeout", 300.0)),
            evaluate_fn=evaluate_fn,
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=expected_clients,
            min_available_clients=expected_clients,
            min_evaluate_clients=expected_clients,
            initial_parameters=ndarrays_to_parameters(get_parameters(model)),
        )
        logger.info(
            f"[{edge_id}] Persistent sub-server starting on {ec['sub_server_address']} "
            f"(child_rounds={total_child_rounds}, expected_clients={expected_clients})"
        )
        event_queue.put(
            WorkerEvent(
                worker="sub_server",
                kind="started",
                message="Persistent sub-server starting",
                expected_clients=expected_clients,
            )
        )
        fl.server.start_server(
            server_address=ec["sub_server_address"],
            config=ServerConfig(
                num_rounds=total_child_rounds,
                round_timeout=float(ec.get("sub_round_timeout", 120.0)),
            ),
            strategy=strategy,
        )
        event_queue.put(
            WorkerEvent(worker="sub_server", kind="finished", message="Server finished")
        )
    except BaseException:
        event_queue.put(
            WorkerEvent(
                worker="sub_server",
                kind="error",
                message=traceback.format_exc(),
            )
        )
        raise


def run_internal_client(
    edge_config: dict[str, Any],
    global_config: dict[str, Any],
    topology_config: dict[str, Any],
    defaults_config: dict[str, Any],
    event_queue: Queue,
) -> None:
    """Internal Clientを一度だけ接続し、全子Roundに参加させる。"""
    ec = edge_config["edge"]
    edge_id = ec["id"]
    try:
        dry_run = global_config.get("mode") == "dry_run"
        model_cfg = global_config.get("model", {})
        dataset_cfg = global_config.get("dataset", {})
        data_partition = topology_config.get("data_partition", {})
        assignments = data_partition.get("assignments", {})
        training_config = {
            **defaults_config.get("training", {}),
            **edge_config.get("training", {}),
        }
        client = HFLClient(
            client_id=f"{edge_id}_internal",
            model_name=model_cfg.get("name", "simplecnn"),
            num_classes=model_cfg.get("num_classes", 10),
            dataset_name=dataset_cfg.get("name", "cifar10"),
            partition_id=assignments.get(f"{edge_id}_internal", 0),
            total_partitions=data_partition.get("total_partitions", 6),
            partition_method=data_partition.get("method", "dirichlet"),
            partition_params=data_partition.get("params", {}),
            training_config=training_config,
            device=edge_config.get("device", defaults_config.get("device", "auto")),
            dry_run=dry_run,
            dry_run_config=global_config.get("dry_run", {}),
        )
        address = ec["sub_server_address"].replace("0.0.0.0", "127.0.0.1")
        event_queue.put(
            WorkerEvent(
                worker="internal_client",
                kind="started",
                message=f"Connecting to {address}",
            )
        )
        _start_client_with_retry(
            server_address=address,
            client=client.to_client(),
            startup_timeout=float(ec.get("startup_timeout", 60.0)),
        )
        event_queue.put(
            WorkerEvent(
                worker="internal_client",
                kind="finished",
                message="Internal client finished",
            )
        )
    except BaseException:
        event_queue.put(
            WorkerEvent(
                worker="internal_client",
                kind="error",
                message=traceback.format_exc(),
            )
        )
        raise
