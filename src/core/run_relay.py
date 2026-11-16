"""
中継ノード起動スクリプト (flat FL: 集約せず子の接続を親へ TCP 転送する)。

Usage:
    python -m src.core.run_relay --relay-config config/jetson-flat/edge_01.yaml \
        [--global-config ...] [--topology-config ...] [--defaults-config ...] [--dry-run]

HFL の run_edge と同じ位置づけだが、sub-server を立てない。
  1. 親 (Global または上位中継) の待受を待つ
  2. listen_address で待ち受け、子の接続を parent_address へそのまま中継する (src/core/tcp_relay.py)
  3. internal_client.enabled なら、自分も HFLClient として親へ直接接続して学習に参加する
  4. 親との接続が終わったら (Global が停止したら)、中継中の接続が無くなるのを待って終了する
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from src.core.client import HFLClient
from src.core.sub_federation import _start_client_with_retry
from src.core.tcp_relay import TcpRelay, relay_summary
from src.utils.config import load_yaml
from src.utils.events import get_event_logger
from src.utils.logger import get_logger
from src.utils.net import install_grpc_noise_filter, wait_for_port
from src.utils.process import run_main

logger = get_logger("run_relay")

PORT_WAIT_TIMEOUT = 600.0  # 親の待受を待つ上限秒数


def run_internal_client(
    relay_config: dict[str, Any],
    global_config: dict[str, Any],
    topology_config: dict[str, Any],
    defaults_config: dict[str, Any],
    parent_address: str,
) -> None:
    """中継ノード自身のクライアント。Edge の Internal Client と同じ client_id / partition。"""
    rc = relay_config["relay"]
    relay_id = rc["id"]
    event_logger = get_event_logger("internal_client")
    started_at = time.monotonic()
    try:
        dry_run = global_config.get("mode") == "dry_run"
        model_cfg = global_config.get("model", {})
        dataset_cfg = global_config.get("dataset", {})
        data_partition = topology_config.get("data_partition", {})
        assignments = data_partition.get("assignments", {})
        training_config = {
            **defaults_config.get("training", {}),
            **relay_config.get("training", {}),
        }
        client = HFLClient(
            client_id=f"{relay_id}_internal",
            model_name=model_cfg.get("name", "simplecnn"),
            num_classes=model_cfg.get("num_classes", 10),
            dataset_name=dataset_cfg.get("name", "cifar10"),
            partition_id=assignments.get(f"{relay_id}_internal", 0),
            total_partitions=data_partition.get("total_partitions", 6),
            partition_method=data_partition.get("method", "dirichlet"),
            partition_params=data_partition.get("params", {}),
            training_config=training_config,
            device=relay_config.get("device", defaults_config.get("device", "auto")),
            dry_run=dry_run,
            dry_run_config=global_config.get("dry_run", {}),
            event_logger=event_logger,
        )
        event_logger.emit(
            "start", argv=sys.argv[1:], client_id=client.client_id,
            address=parent_address, dry_run=dry_run,
        )
        logger.info(f"[{relay_id}] Internal client connecting to {parent_address}...")
        _start_client_with_retry(
            server_address=parent_address,
            client=client.to_client(),
            startup_timeout=float(rc.get("startup_timeout", 60.0)),
        )
        event_logger.emit(
            "stop", reason="finished", elapsed_s=round(time.monotonic() - started_at, 1)
        )
    except BaseException as exc:
        event_logger.emit_error(exc)
        raise
    finally:
        event_logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="flat FL relay node (TCP forwarding)")
    parser.add_argument("--relay-config", type=str, required=True)
    parser.add_argument("--global-config", type=str, default="config/global.yaml")
    parser.add_argument("--topology-config", type=str, default="config/topology.yaml")
    parser.add_argument("--defaults-config", type=str, default="config/defaults.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    event_logger = get_event_logger("relay")
    started_at = time.monotonic()

    try:
        global_cfg = load_yaml(args.global_config)
        if args.dry_run:
            global_cfg["mode"] = "dry_run"
        relay_cfg = load_yaml(args.relay_config)
        topo_cfg = load_yaml(args.topology_config)
        defaults_cfg = load_yaml(args.defaults_config)
        rc = relay_cfg["relay"]
        relay_id = rc["id"]
        listen_address = rc["listen_address"]
        parent_address = rc["parent_address"]
        startup_timeout = float(rc.get("startup_timeout", PORT_WAIT_TIMEOUT))
        shutdown_timeout = float(rc.get("shutdown_timeout", 30.0))
        internal_enabled = bool(rc.get("internal_client", {}).get("enabled", True))
        event_logger.emit(
            "start",
            argv=sys.argv[1:],
            config=args.relay_config,
            listen_address=listen_address,
            parent_address=parent_address,
            internal_client=internal_enabled,
            dry_run=global_cfg.get("mode") == "dry_run",
        )

        # 子は中継の待受を見てから接続してくるので、中継は親の待受を見てから listen する
        install_grpc_noise_filter()
        if not wait_for_port(parent_address, timeout=startup_timeout):
            raise TimeoutError(
                f"[{relay_id}] parent {parent_address} did not accept connections "
                f"within {startup_timeout:.0f}s"
            )
        relay = TcpRelay(
            listen_address, parent_address, node_id=relay_id, event_logger=event_logger
        )
        relay.start()
        try:
            if internal_enabled:
                run_internal_client(relay_cfg, global_cfg, topo_cfg, defaults_cfg, parent_address)
                logger.info(
                    f"[{relay_id}] Internal client finished; waiting for relayed connections"
                )
                idle = relay.wait_idle(shutdown_timeout)
            else:
                # 転送だけのノード: 子が一度つながってから全部切れたら終わり
                idle = relay.wait_idle(None, after_first_connection=True)
            if not idle:
                logger.warning(
                    f"[{relay_id}] {relay.active_connections()} relayed connection(s) still open "
                    f"after {shutdown_timeout:.0f}s; closing them"
                )
        finally:
            relay.stop()
        summary = relay_summary(relay)
        logger.info(
            f"[{relay_id}] Relay finished: connections={summary['connections']}, "
            f"up={summary['bytes_up']} B, down={summary['bytes_down']} B"
        )
        event_logger.emit(
            "stop", reason="finished", elapsed_s=round(time.monotonic() - started_at, 1), **summary
        )
    except BaseException as exc:
        event_logger.emit_error(exc)
        raise
    finally:
        event_logger.close()


if __name__ == "__main__":
    run_main(main)
