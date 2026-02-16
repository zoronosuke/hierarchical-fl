"""
Edge Node 起動スクリプト。

Usage:
    python -m src.core.run_edge --edge-config config/edge/edge_01.yaml \
        [--global-config config/global.yaml] [--dry-run]
"""

from __future__ import annotations

import argparse

import flwr as fl

from src.core.edge_node import EdgeNode
from src.utils.config import load_yaml
from src.utils.logger import get_logger

logger = get_logger("run_edge")


def main() -> None:
    parser = argparse.ArgumentParser(description="HFL Edge Node")
    parser.add_argument("--edge-config", type=str, required=True)
    parser.add_argument("--global-config", type=str, default="config/global.yaml")
    parser.add_argument("--topology-config", type=str, default="config/topology.yaml")
    parser.add_argument("--defaults-config", type=str, default="config/defaults.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    global_cfg = load_yaml(args.global_config)
    if args.dry_run:
        global_cfg["mode"] = "dry_run"

    edge_cfg = load_yaml(args.edge_config)
    topo_cfg = load_yaml(args.topology_config)
    defaults_cfg = load_yaml(args.defaults_config)

    edge_node = EdgeNode(
        edge_config=edge_cfg,
        global_config=global_cfg,
        topology_config=topo_cfg,
        defaults_config=defaults_cfg,
    )

    # Global Server に接続
    global_addr = edge_cfg["edge"].get("global_server_address", "127.0.0.1:8080")
    logger.info(f"Connecting to Global Server at {global_addr}...")

    fl.client.start_client(
        server_address=global_addr,
        client=edge_node.to_client(),
        insecure=True,
    )
    logger.info("Edge Node finished.")


if __name__ == "__main__":
    main()
