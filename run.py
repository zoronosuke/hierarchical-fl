"""1台のPCでGlobal、Edge、Leafを起動・監視するオーケストレーター。"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

from src.utils.config import load_yaml
from src.utils.logger import get_logger

logger = get_logger("orchestrator")


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen


def _pre_download_dataset(dataset_name: str) -> None:
    ds_mapping = {"cifar10": "uoft-cs/cifar10", "mnist": "ylecun/mnist"}
    hf_name = ds_mapping.get(dataset_name, dataset_name)
    logger.info(f"Pre-downloading dataset '{hf_name}' to local cache...")
    from datasets import load_dataset

    dataset = load_dataset(hf_name)
    logger.info(
        f"Dataset ready: {', '.join(f'{key}: {len(value)}' for key, value in dataset.items())}"
    )


def _stop_all(processes: list[ManagedProcess], grace: float = 10.0) -> None:
    for managed in processes:
        if managed.process.poll() is None:
            managed.process.terminate()
    deadline = time.monotonic() + grace
    for managed in processes:
        if managed.process.poll() is None:
            remaining = max(deadline - time.monotonic(), 0.0)
            try:
                managed.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                managed.process.kill()
                managed.process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description="HFL Orchestrator")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--global-config", default="config/global.yaml")
    parser.add_argument("--topology-config", default="config/topology.yaml")
    parser.add_argument("--defaults-config", default="config/defaults.yaml")
    parser.add_argument(
        "--completion-timeout",
        type=float,
        default=900.0,
        help="全プロセスが終了するまでの上限秒数",
    )
    args = parser.parse_args()

    global_cfg = load_yaml(args.global_config)
    topology_cfg = load_yaml(args.topology_config)
    if not args.dry_run:
        _pre_download_dataset(global_cfg.get("dataset", {}).get("name", "cifar10"))

    dry_run_flag = ["--dry-run"] if args.dry_run else []
    python = sys.executable
    sub_env = {**os.environ, "HF_DATASETS_OFFLINE": "1"}
    processes: list[ManagedProcess] = []
    interrupted = False

    def handle_signal(signum, frame):
        nonlocal interrupted
        interrupted = True
        logger.error(f"Received signal {signum}; stopping all processes.")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    def launch(name: str, command: list[str]) -> None:
        logger.info(f"Starting {name}...")
        process = subprocess.Popen(command, env=sub_env)
        processes.append(ManagedProcess(name, process))

    try:
        launch(
            "global",
            [
                python,
                "-m",
                "src.core.global_server",
                "--config",
                args.global_config,
                *dry_run_flag,
            ],
        )
        time.sleep(2)

        edges = topology_cfg.get("edges", {})
        for edge_id, edge_info in edges.items():
            edge_config_file = edge_info.get(
                "config_file", f"config/edge/{edge_id}.yaml"
            )
            launch(
                edge_id,
                [
                    python,
                    "-m",
                    "src.core.run_edge",
                    "--edge-config",
                    edge_config_file,
                    "--global-config",
                    args.global_config,
                    "--topology-config",
                    args.topology_config,
                    "--defaults-config",
                    args.defaults_config,
                    *dry_run_flag,
                ],
            )

        time.sleep(2)
        assignments = topology_cfg.get("data_partition", {}).get("assignments", {})
        for edge_id, edge_info in edges.items():
            edge_config_file = edge_info.get(
                "config_file", f"config/edge/{edge_id}.yaml"
            )
            edge_cfg = load_yaml(edge_config_file)
            address = edge_cfg["edge"]["sub_server_address"].replace(
                "0.0.0.0", "127.0.0.1"
            )
            for leaf_id in edge_info.get("leaf_clients", []):
                launch(
                    leaf_id,
                    [
                        python,
                        "-m",
                        "src.core.run_leaf",
                        "--client-id",
                        leaf_id,
                        "--edge-address",
                        address,
                        "--partition-id",
                        str(assignments.get(leaf_id, 0)),
                        "--global-config",
                        args.global_config,
                        "--topology-config",
                        args.topology_config,
                        "--defaults-config",
                        args.defaults_config,
                        *dry_run_flag,
                    ],
                )

        logger.info(f"All {len(processes)} processes started.")
        deadline = time.monotonic() + args.completion_timeout
        while True:
            if interrupted:
                raise RuntimeError("Run interrupted by signal")
            failed = [
                managed
                for managed in processes
                if managed.process.poll() not in (None, 0)
            ]
            if failed:
                details = ", ".join(
                    f"{managed.name}={managed.process.returncode}" for managed in failed
                )
                raise RuntimeError(f"Child process failure: {details}")
            if all(managed.process.poll() == 0 for managed in processes):
                break
            if time.monotonic() >= deadline:
                running = [
                    managed.name
                    for managed in processes
                    if managed.process.poll() is None
                ]
                raise TimeoutError(
                    f"Completion timeout after {args.completion_timeout:.1f}s; "
                    f"running={running}"
                )
            time.sleep(0.2)

        summary = ", ".join(
            f"{managed.name}={managed.process.returncode}" for managed in processes
        )
        logger.info(f"All processes completed successfully: {summary}")
        return 0
    except BaseException as exc:
        logger.error(f"HFL run failed: {exc}")
        _stop_all(processes)
        summary = ", ".join(
            f"{managed.name}={managed.process.poll()}" for managed in processes
        )
        logger.error(f"Final process states: {summary}")
        return 1
    finally:
        _stop_all(processes)


if __name__ == "__main__":
    raise SystemExit(main())
