"""
Global Server 起動スクリプト。

Usage:
    python -m src.core.global_server [--config config/global.yaml] [--dry-run]
        [--max-hours 240]

--max-hours (または YAML の server.max_duration_hours) を指定すると、その時間を
過ぎた最初のラウンド境界で正常停止する。環境変数 HFL_STOP_FILE のファイルが
作られた場合も同じ経路でラウンド境界停止になる (scripts/hfl.sh stop が使う)。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig, SimpleClientManager

from src.core.stoppable_server import StoppableServer
from src.core.training import evaluate_model
from src.data.loader import create_dummy_dataloader, load_test_data
from src.models.nets import create_model, get_parameters, set_parameters
from src.strategies.aggregation import create_strategy
from src.utils.config import load_yaml, resolve_device
from src.utils.events import EventLogger, get_event_logger
from src.utils.logger import get_logger

logger = get_logger("global_server")


class _RoundClock:
    """ラウンド開始時刻を覚えておき、評価イベントに所要時間を付ける。"""

    def __init__(self, event_logger: EventLogger) -> None:
        self.event_logger = event_logger
        self.started_at: float | None = None

    def round_start(self, server_round: int) -> None:
        self.started_at = time.monotonic()
        self.event_logger.emit(
            "round_start", global_round=server_round, parent_round=server_round
        )

    def round_duration(self) -> float | None:
        if self.started_at is None:
            return None
        return round(time.monotonic() - self.started_at, 3)


def _build_evaluate_fn(model, testloader, device, event_logger, clock):
    """サーバーサイド評価関数を構築する。"""
    def evaluate_fn(server_round, parameters_ndarrays, config):
        set_parameters(model, parameters_ndarrays)
        loss, accuracy, n = evaluate_model(model, testloader, device)
        logger.info(
            f"[Global Eval] Round {server_round}: "
            f"loss={loss:.4f}, accuracy={accuracy:.4f} (n={n})"
        )
        fields = {"loss": float(loss), "accuracy": float(accuracy), "n": n}
        if server_round > 0:
            fields["round_duration_s"] = clock.round_duration()
        event_logger.emit(
            "eval", global_round=server_round, parent_round=server_round, **fields
        )
        return loss, {"accuracy": accuracy}
    return evaluate_fn


def _resolve_max_hours(args_max_hours: float | None, srv_cfg: dict) -> float | None:
    if args_max_hours is not None:
        return args_max_hours
    value = srv_cfg.get("max_duration_hours")
    return float(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description="HFL Global Server")
    parser.add_argument("--config", type=str, default="config/global.yaml")
    parser.add_argument("--dry-run", action="store_true", help="疎通確認モード")
    parser.add_argument(
        "--max-hours", type=float, default=None,
        help="この時間を過ぎたラウンド境界で停止 (YAML server.max_duration_hours を上書き)",
    )
    args = parser.parse_args()
    event_logger = get_event_logger("global")
    started_at = time.monotonic()

    try:
        cfg = load_yaml(args.config)
        if args.dry_run:
            cfg["mode"] = "dry_run"
        dry_run = cfg.get("mode", "production") == "dry_run"

        # --- サーバー設定 ---
        srv_cfg = cfg.get("server", {})
        address = srv_cfg.get("address", "0.0.0.0:8080")
        round_timeout = srv_cfg.get("round_timeout", 300.0)
        dr_cfg = cfg.get("dry_run", {})
        num_rounds = (
            dr_cfg.get("global_rounds", 2)
            if dry_run
            else srv_cfg.get("num_rounds", 10)
        )
        max_hours = _resolve_max_hours(args.max_hours, srv_cfg)
        deadline = started_at + max_hours * 3600.0 if max_hours is not None else None
        stop_file = os.environ.get("HFL_STOP_FILE") or None
        event_logger.emit(
            "start",
            argv=sys.argv[1:],
            config=args.config,
            num_rounds=num_rounds,
            max_hours=max_hours,
            stop_file=stop_file,
            dry_run=dry_run,
        )

        # --- モデル ---
        model_cfg = cfg.get("model", {})
        ds_cfg = cfg.get("dataset", {})
        dataset_name = ds_cfg.get("name", "cifar10")
        in_channels = 1 if dataset_name == "mnist" else 3

        model = create_model(
            name=model_cfg.get("name", "simplecnn"),
            num_classes=model_cfg.get("num_classes", 10),
            in_channels=in_channels,
            dry_run=dry_run,
        )
        initial_params = ndarrays_to_parameters(get_parameters(model))
        device = resolve_device(cfg.get("device", "auto"))
        clock = _RoundClock(event_logger)

        # --- 評価関数 ---
        eval_cfg = cfg.get("evaluation", {})
        evaluate_fn = None
        if eval_cfg.get("enabled", True):
            if dry_run:
                testloader = create_dummy_dataloader(
                    batch_size=32, num_samples=dr_cfg.get("dummy_data_size", 64),
                    in_channels=in_channels,
                    num_classes=model_cfg.get("num_classes", 10),
                )
            else:
                testloader = load_test_data(
                    dataset_name, batch_size=ds_cfg.get("test_batch_size", 128)
                )
            evaluate_fn = _build_evaluate_fn(model, testloader, device, event_logger, clock)

        # --- Strategy ---
        strat_cfg = cfg.get("strategy", {})
        strategy = create_strategy(
            name=strat_cfg.get("name", "fedavg"),
            params=strat_cfg.get("params"),
            initial_parameters=initial_params,
            evaluate_fn=evaluate_fn,
            event_logger=event_logger,
            fraction_fit=srv_cfg.get("fraction_fit", 1.0),
            fraction_evaluate=0.0,  # 分散評価は無効 (evaluate_fn でサーバーサイド評価を行う)
            min_fit_clients=srv_cfg.get("min_fit_clients", 1),
            min_available_clients=srv_cfg.get("min_available_clients", 1),
            min_evaluate_clients=srv_cfg.get("min_available_clients", 1),
            accept_failures=False,
            # global_round は階層を下るときも保持され、全ノードのイベントで同じ軸になる
            on_fit_config_fn=lambda server_round: {
                "parent_round": server_round,
                "global_round": server_round,
            },
        )

        # --- サーバー起動 ---
        server = StoppableServer(
            client_manager=SimpleClientManager(),
            strategy=strategy,
            deadline=deadline,
            stop_file=stop_file,
            on_round_start=clock.round_start,
        )
        logger.info(
            f"Starting Global Server on {address} "
            f"(rounds={num_rounds}, max_hours={max_hours}, stop_file={stop_file}, "
            f"dry_run={dry_run})"
        )
        fl.server.start_server(
            server_address=address,
            server=server,
            config=ServerConfig(num_rounds=num_rounds, round_timeout=round_timeout),
        )
        reason = server.stop_reason or "finished"
        event_logger.emit(
            "stop", reason=reason, elapsed_s=round(time.monotonic() - started_at, 1)
        )
        logger.info(f"Global Server finished (reason={reason}).")
    except BaseException as exc:
        event_logger.emit_error(exc)
        raise
    finally:
        event_logger.close()


if __name__ == "__main__":
    main()
