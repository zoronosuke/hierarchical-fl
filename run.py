"""
一括起動スクリプト (PoC用)。

1台のPC上で Global Server, Edge Nodes, Leaf Clients を
サブプロセスとして起動し、連合学習パイプライン全体を実行する。

Usage:
    python run.py [--dry-run]
    python run.py --mode production
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

from src.utils.config import load_yaml
from src.utils.logger import get_logger

logger = get_logger("orchestrator")


def main() -> None:
    parser = argparse.ArgumentParser(description="HFL Orchestrator (全プロセス一括起動)")
    parser.add_argument("--dry-run", action="store_true", help="疎通確認モード")
    parser.add_argument(
        "--global-config", type=str, default="config/global.yaml"
    )
    parser.add_argument(
        "--topology-config", type=str, default="config/topology.yaml"
    )
    parser.add_argument(
        "--defaults-config", type=str, default="config/defaults.yaml"
    )
    args = parser.parse_args()

    global_cfg = load_yaml(args.global_config)
    topo_cfg = load_yaml(args.topology_config)

    dry_run_flag = ["--dry-run"] if args.dry_run else []
    python = sys.executable  # 現在のPythonインタープリタ

    processes: list[subprocess.Popen] = []

    def cleanup(sig=None, frame=None):
        logger.info("Shutting down all processes...")
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            p.wait(timeout=10)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # ==================================================================
        # 1. Global Server 起動
        # ==================================================================
        logger.info("Starting Global Server...")
        p_global = subprocess.Popen(
            [python, "-m", "src.core.global_server",
             "--config", args.global_config] + dry_run_flag,
        )
        processes.append(p_global)
        time.sleep(3)  # サーバー起動待ち

        # ==================================================================
        # 2. Edge Nodes 起動
        # ==================================================================
        edges = topo_cfg.get("edges", {})
        for edge_id, edge_info in edges.items():
            edge_config_file = edge_info.get("config_file", f"config/edge/{edge_id}.yaml")
            logger.info(f"Starting Edge Node: {edge_id} ({edge_config_file})")

            p_edge = subprocess.Popen(
                [python, "-m", "src.core.run_edge",
                 "--edge-config", edge_config_file,
                 "--global-config", args.global_config,
                 "--topology-config", args.topology_config,
                 "--defaults-config", args.defaults_config] + dry_run_flag,
            )
            processes.append(p_edge)
            time.sleep(2)  # Edge の sub-server 起動待ち

        # ==================================================================
        # 3. Leaf Clients 起動
        # ==================================================================
        assignments = topo_cfg.get("data_partition", {}).get("assignments", {})

        for edge_id, edge_info in edges.items():
            # Edge の sub-server アドレスを取得
            edge_cfg = load_yaml(edge_info.get("config_file", f"config/edge/{edge_id}.yaml"))
            sub_addr = edge_cfg["edge"]["sub_server_address"].replace("0.0.0.0", "127.0.0.1")

            leaf_ids = edge_info.get("leaf_clients", [])
            for leaf_id in leaf_ids:
                partition_id = assignments.get(leaf_id, 0)
                logger.info(f"Starting Leaf Client: {leaf_id} -> {sub_addr} (partition={partition_id})")

                p_leaf = subprocess.Popen(
                    [python, "-m", "src.core.run_leaf",
                     "--client-id", leaf_id,
                     "--edge-address", sub_addr,
                     "--partition-id", str(partition_id),
                     "--global-config", args.global_config,
                     "--topology-config", args.topology_config,
                     "--defaults-config", args.defaults_config] + dry_run_flag,
                )
                processes.append(p_leaf)
                time.sleep(0.5)

        # ==================================================================
        # 4. 全プロセスの完了を待機
        # ==================================================================
        logger.info(f"All {len(processes)} processes started. Waiting for completion...")

        # Global Server の終了を待つ (それが終われば全体が終了)
        p_global.wait()
        logger.info(f"Global Server exited with code {p_global.returncode}")

        # 残りのプロセスも回収
        time.sleep(5)
        for p in processes:
            if p.poll() is None:
                p.terminate()
            p.wait(timeout=10)

        logger.info("All processes completed.")

    except Exception as e:
        logger.error(f"Error: {e}")
        cleanup()


if __name__ == "__main__":
    main()
