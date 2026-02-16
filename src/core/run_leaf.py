"""
Leaf Client 起動スクリプト。

Usage:
    python -m src.core.run_leaf --client-id leaf_01 --edge-address 127.0.0.1:9001 \
        [--global-config config/global.yaml] [--dry-run]
"""

from __future__ import annotations

import argparse
import time

import flwr as fl
import grpc

from src.core.client import HFLClient
from src.utils.config import load_yaml
from src.utils.logger import get_logger

logger = get_logger("run_leaf")

MAX_RETRIES = 10
RETRY_INTERVAL = 5  # 秒


def main() -> None:
    parser = argparse.ArgumentParser(description="HFL Leaf Client")
    parser.add_argument("--client-id", type=str, required=True)
    parser.add_argument("--edge-address", type=str, required=True,
                        help="接続先 Edge Server アドレス (e.g., 127.0.0.1:9001)")
    parser.add_argument("--partition-id", type=int, required=True)
    parser.add_argument("--global-config", type=str, default="config/global.yaml")
    parser.add_argument("--topology-config", type=str, default="config/topology.yaml")
    parser.add_argument("--defaults-config", type=str, default="config/defaults.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    global_cfg = load_yaml(args.global_config)
    if args.dry_run:
        global_cfg["mode"] = "dry_run"
    dry_run = global_cfg.get("mode", "production") == "dry_run"

    topo_cfg = load_yaml(args.topology_config)
    defaults_cfg = load_yaml(args.defaults_config)
    dp = topo_cfg.get("data_partition", {})

    model_cfg = global_cfg.get("model", {})
    ds_cfg = global_cfg.get("dataset", {})
    training_cfg = defaults_cfg.get("training", {})

    client = HFLClient(
        client_id=args.client_id,
        model_name=model_cfg.get("name", "simplecnn"),
        num_classes=model_cfg.get("num_classes", 10),
        dataset_name=ds_cfg.get("name", "cifar10"),
        partition_id=args.partition_id,
        total_partitions=dp.get("total_partitions", 6),
        partition_method=dp.get("method", "dirichlet"),
        partition_params=dp.get("params"),
        training_config=training_cfg,
        device=defaults_cfg.get("device", "auto"),
        dry_run=dry_run,
        dry_run_config=global_cfg.get("dry_run"),
    )

    # Edge sub-server の起動を待ってからリトライ付きで接続
    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(
            f"Connecting to Edge Server at {args.edge_address}... "
            f"(attempt {attempt}/{MAX_RETRIES})"
        )
        try:
            fl.client.start_client(
                server_address=args.edge_address,
                client=client.to_client(),
                insecure=True,
            )
            logger.info(f"Leaf Client {args.client_id} finished.")
            return
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE and attempt < MAX_RETRIES:
                logger.warning(
                    f"[{args.client_id}] Edge Server not ready, "
                    f"retrying in {RETRY_INTERVAL}s... ({e.details()})"
                )
                time.sleep(RETRY_INTERVAL)
            else:
                raise

    logger.error(f"[{args.client_id}] Failed to connect after {MAX_RETRIES} attempts.")


if __name__ == "__main__":
    main()
