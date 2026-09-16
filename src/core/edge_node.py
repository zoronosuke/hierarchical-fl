"""Global Clientとして動作するEdge Nodeと常駐子連合の管理。"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from typing import Any

import flwr as fl
from flwr.common import NDArrays, Scalar

from src.core.sub_federation import (
    ChildModelResult,
    ParentModelRequest,
    WorkerEvent,
    run_internal_client,
    run_sub_server,
)
from src.core.training import evaluate_model
from src.data.loader import create_dummy_dataloader, load_test_data
from src.models.nets import (
    copy_parameters,
    create_model,
    get_parameters,
    set_parameters,
    validate_parameters,
)
from src.utils.config import resolve_device
from src.utils.logger import get_logger

logger = get_logger("edge")


class EdgeRuntime:
    """子サーバーとInternal Clientのspawnプロセスを所有する。"""

    def __init__(
        self,
        edge_config: dict[str, Any],
        global_config: dict[str, Any],
        topology_config: dict[str, Any],
        defaults_config: dict[str, Any],
    ) -> None:
        self.edge_config = edge_config
        self.global_config = global_config
        self.topology_config = topology_config
        self.defaults_config = defaults_config
        self.edge_id = edge_config["edge"]["id"]
        self.context = mp.get_context("spawn")
        self.parent_queue = self.context.Queue(maxsize=1)
        self.result_queue = self.context.Queue(maxsize=1)
        self.event_queue = self.context.Queue()
        self.sub_server_process: mp.Process | None = None
        self.internal_client_process: mp.Process | None = None
        self.last_event: WorkerEvent | None = None

    def start(self) -> None:
        ec = self.edge_config["edge"]
        self.sub_server_process = self.context.Process(
            name=f"{self.edge_id}-sub-server",
            target=run_sub_server,
            args=(
                self.edge_config,
                self.global_config,
                self.defaults_config,
                self.parent_queue,
                self.result_queue,
                self.event_queue,
            ),
        )
        self.sub_server_process.start()

        if ec.get("internal_client", {}).get("enabled", True):
            self.internal_client_process = self.context.Process(
                name=f"{self.edge_id}-internal-client",
                target=run_internal_client,
                args=(
                    self.edge_config,
                    self.global_config,
                    self.topology_config,
                    self.defaults_config,
                    self.event_queue,
                ),
            )
            self.internal_client_process.start()

        self._wait_for_server_start(float(ec.get("startup_timeout", 60.0)))

    def _wait_for_server_start(self, timeout: float) -> None:
        started = time.monotonic()
        while time.monotonic() - started < timeout:
            self.raise_if_failed()
            try:
                event = self.event_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self.last_event = event
            if event.kind == "error":
                raise RuntimeError(f"[{self.edge_id}] {event.worker} failed:\n{event.message}")
            if event.worker == "sub_server" and event.kind == "started":
                logger.info(f"[{self.edge_id}] Child server process is ready.")
                return
        raise TimeoutError(
            f"[{self.edge_id}] Startup timeout: edge_id={self.edge_id}, "
            "parent_round=0, child_round=0, connected_clients=0, "
            f"expected_clients={self.edge_config['edge']['min_available_clients']}, "
            f"waited={timeout:.1f}s"
        )

    def drain_events(self) -> None:
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                return
            self.last_event = event
            if event.kind == "error":
                raise RuntimeError(f"[{self.edge_id}] {event.worker} failed:\n{event.message}")

    def raise_if_failed(self) -> None:
        for process in (self.sub_server_process, self.internal_client_process):
            if process is not None and process.exitcode not in (None, 0):
                raise RuntimeError(
                    f"[{self.edge_id}] Worker {process.name} exited with "
                    f"code {process.exitcode}"
                )
        self.drain_events()

    def shutdown(self, successful: bool) -> None:
        timeout = float(self.edge_config["edge"].get("shutdown_timeout", 30.0))
        processes = [
            process
            for process in (self.internal_client_process, self.sub_server_process)
            if process is not None
        ]
        if successful:
            for process in processes:
                process.join(timeout=timeout)
            alive = [process for process in processes if process.is_alive()]
            failed = [
                process
                for process in processes
                if process.exitcode not in (None, 0)
            ]
            if alive or failed:
                for process in alive:
                    process.terminate()
                for process in alive:
                    process.join(timeout=5)
                details = ", ".join(
                    f"{process.name}:exit={process.exitcode}" for process in alive + failed
                )
                raise RuntimeError(f"[{self.edge_id}] Worker shutdown failed: {details}")
        else:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)

        self.parent_queue.close()
        self.result_queue.close()
        self.event_queue.close()


class EdgeNode(fl.client.NumPyClient):
    """Global Serverのクライアントとして子連合を同期実行する。"""

    def __init__(
        self,
        edge_config: dict[str, Any],
        global_config: dict[str, Any],
        topology_config: dict[str, Any],
        defaults_config: dict[str, Any],
        runtime: EdgeRuntime,
    ) -> None:
        ec = edge_config["edge"]
        self.runtime = runtime
        self.edge_id = ec["id"]
        self.contribution_factor = float(ec.get("contribution_factor", 1.0))
        self.result_timeout = float(ec.get("parent_result_timeout", 180.0))
        self.expected_parent_round = 1
        self.sub_rounds = self._sub_rounds(ec, global_config)
        required_child_time = self.sub_rounds * float(
            ec.get("sub_round_timeout", 120.0)
        )
        global_timeout = float(
            global_config.get("server", {}).get("round_timeout", 300.0)
        )
        if self.result_timeout <= required_child_time:
            raise ValueError(
                f"[{self.edge_id}] parent_result_timeout={self.result_timeout} must "
                f"exceed sub_rounds * sub_round_timeout={required_child_time}"
            )
        if global_timeout <= self.result_timeout:
            raise ValueError(
                f"[{self.edge_id}] Global round_timeout={global_timeout} must exceed "
                f"parent_result_timeout={self.result_timeout}"
            )
        self.expected_clients = len(
            topology_config.get("edges", {}).get(self.edge_id, {}).get("leaf_clients", [])
        ) + int(ec.get("internal_client", {}).get("enabled", True))
        if self.expected_clients != int(ec["min_available_clients"]):
            raise ValueError(
                f"[{self.edge_id}] Configured participants={self.expected_clients}, "
                f"min_available_clients={ec['min_available_clients']}"
            )

        self.dry_run = global_config.get("mode") == "dry_run"
        self.device = resolve_device(
            edge_config.get("device", defaults_config.get("device", "auto"))
        )
        model_cfg = global_config.get("model", {})
        dataset_cfg = global_config.get("dataset", {})
        self.dataset_name = dataset_cfg.get("name", "cifar10")
        in_channels = 1 if self.dataset_name == "mnist" else 3
        self.model = create_model(
            model_cfg.get("name", "simplecnn"),
            num_classes=model_cfg.get("num_classes", 10),
            in_channels=in_channels,
            dry_run=self.dry_run,
        )
        if self.dry_run:
            dry_cfg = global_config.get("dry_run", {})
            self.testloader = create_dummy_dataloader(
                batch_size=32,
                num_samples=dry_cfg.get("dummy_data_size", 64),
                in_channels=in_channels,
                num_classes=model_cfg.get("num_classes", 10),
            )
        else:
            self.testloader = load_test_data(
                self.dataset_name,
                batch_size=dataset_cfg.get("test_batch_size", 128),
            )

        logger.info(
            f"[{self.edge_id}] Initialized persistent Edge client "
            f"(sub_rounds={self.sub_rounds}, expected_clients={self.expected_clients})"
        )

    @staticmethod
    def _sub_rounds(ec: dict[str, Any], global_config: dict[str, Any]) -> int:
        if global_config.get("mode") == "dry_run":
            return int(
                global_config.get("dry_run", {}).get(
                    "sub_rounds", ec.get("sub_rounds", 1)
                )
            )
        return int(ec.get("sub_rounds", 1))

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.model)

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        raw_round = config.get("parent_round")
        if not isinstance(raw_round, int):
            raise ValueError(
                f"[{self.edge_id}] Missing integer parent_round in fit config"
            )
        parent_round = raw_round
        if parent_round != self.expected_parent_round:
            raise RuntimeError(
                f"[{self.edge_id}] Non-sequential Global round: "
                f"expected={self.expected_parent_round}, got={parent_round}"
            )
        validate_parameters(self.model, parameters)
        request = ParentModelRequest(parent_round, copy_parameters(parameters))
        self.runtime.parent_queue.put(request, timeout=5)
        logger.info(
            f"[{self.edge_id}] Parent model submitted: parent_round={parent_round}"
        )

        started = time.monotonic()
        while True:
            self.runtime.raise_if_failed()
            elapsed = time.monotonic() - started
            remaining = self.result_timeout - elapsed
            if remaining <= 0:
                last = self.runtime.last_event
                child_round = last.child_round if last is not None else 0
                connected = last.connected_clients if last is not None else 0
                raise TimeoutError(
                    f"[{self.edge_id}] Child result timeout: edge_id={self.edge_id}, "
                    f"parent_round={parent_round}, child_round={child_round}, "
                    f"connected_clients={connected}, "
                    f"expected_clients={self.expected_clients}, "
                    f"waited={elapsed:.1f}s"
                )
            try:
                result = self.runtime.result_queue.get(timeout=min(0.5, remaining))
            except queue.Empty:
                continue
            break

        if not isinstance(result, ChildModelResult):
            raise TypeError(f"Unexpected result queue message: {type(result)!r}")
        expected_child_round = parent_round * self.sub_rounds
        if (
            result.parent_round != parent_round
            or result.child_round != expected_child_round
        ):
            raise RuntimeError(
                f"[{self.edge_id}] Result round mismatch: expected parent/child "
                f"{parent_round}/{expected_child_round}, got "
                f"{result.parent_round}/{result.child_round}"
            )
        validate_parameters(self.model, result.parameters)
        final_parameters = copy_parameters(result.parameters)
        set_parameters(self.model, final_parameters)
        self.expected_parent_round += 1
        reported_examples = max(
            int(result.num_examples * self.contribution_factor), 1
        )
        loss, accuracy, sample_count = evaluate_model(
            self.model, self.testloader, self.device
        )
        logger.info(
            f"[{self.edge_id}] Returning child aggregate: parent_round={parent_round}, "
            f"child_round={result.child_round}, clients={self.expected_clients}, "
            f"reported_examples={reported_examples}"
        )
        return final_parameters, reported_examples, {
            "edge_id": self.edge_id,
            "parent_round": parent_round,
            "child_round": result.child_round,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "eval_examples": sample_count,
        }

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        set_parameters(self.model, parameters)
        loss, accuracy, num_examples = evaluate_model(
            self.model, self.testloader, self.device
        )
        return float(loss), num_examples, {
            "accuracy": float(accuracy),
            "edge_id": self.edge_id,
        }
