"""
Global Server 起動スクリプト。

Usage:
    python -m src.core.global_server [--config config/global.yaml] [--dry-run]
        [--max-hours 240] [--checkpoint PATH] [--resume]

--max-hours (または YAML の server.max_duration_hours) を指定すると、その時間を
過ぎた最初のラウンド境界で正常停止する。環境変数 HFL_STOP_FILE のファイルが
作られた場合も同じ経路でラウンド境界停止になる (scripts/hfl.sh stop が使う)。

--checkpoint (または環境変数 HFL_CHECKPOINT) を指定すると、server.checkpoint_every
ラウンドごとと停止時に Global モデルを保存する。--resume を付けるとそのチェックポイントの
続きから再開する。ラウンド番号 (global_round) は通しで続き、時間上限は最初の開始時刻から数える。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerConfig, SimpleClientManager

from src.core.stoppable_server import StoppableServer
from src.core.training import evaluate_model
from src.data.loader import create_dummy_dataloader, load_test_data
from src.models.nets import create_model, get_parameters, set_parameters, validate_parameters
from src.strategies.aggregation import create_strategy
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config import load_yaml, resolve_device
from src.utils.events import EventLogger, get_event_logger
from src.utils.logger import get_logger
from src.utils.process import run_main

logger = get_logger("global_server")


class _RoundClock:
    """ラウンド開始時刻を覚えておき、評価イベントに所要時間を付ける。"""

    def __init__(self, event_logger: EventLogger, offset: int = 0) -> None:
        self.event_logger = event_logger
        self.offset = offset
        self.started_at: float | None = None

    def round_start(self, server_round: int) -> None:
        self.started_at = time.monotonic()
        self.event_logger.emit(
            "round_start", global_round=server_round + self.offset, parent_round=server_round
        )

    def round_duration(self) -> float | None:
        if self.started_at is None:
            return None
        return round(time.monotonic() - self.started_at, 3)


class _Checkpointer:
    """checkpoint_every ラウンドごとと停止時に Global モデルを保存する。"""

    def __init__(
        self,
        path: str | None,
        *,
        every: int,
        offset: int,
        run_started_t: float,
        meta: dict,
        event_logger: EventLogger,
    ) -> None:
        self.path = path
        self.every = max(int(every), 1)
        self.offset = offset
        self.run_started_t = run_started_t
        self.meta = meta
        self.event_logger = event_logger
        self.saved_round = offset

    def save(self, global_round: int, parameters: Parameters) -> None:
        if self.path is None:
            return
        started = time.monotonic()
        save_checkpoint(
            self.path,
            parameters_to_ndarrays(parameters),
            {
                **self.meta,
                "global_round": global_round,
                "run_started_t": self.run_started_t,
                "saved_t": round(time.time(), 3),
            },
        )
        self.saved_round = global_round
        self.event_logger.emit(
            "checkpoint",
            global_round=global_round,
            path=self.path,
            duration_s=round(time.monotonic() - started, 3),
        )

    def on_round_end(self, server_round: int, parameters: Parameters) -> None:
        global_round = server_round + self.offset
        if global_round % self.every == 0:
            self.save(global_round, parameters)


def _build_evaluate_fn(model, testloader, device, event_logger, clock):
    """サーバーサイド評価関数を構築する。"""
    def evaluate_fn(server_round, parameters_ndarrays, config):
        set_parameters(model, parameters_ndarrays)
        loss, accuracy, n = evaluate_model(model, testloader, device)
        global_round = server_round + clock.offset
        logger.info(
            f"[Global Eval] Round {global_round}: "
            f"loss={loss:.4f}, accuracy={accuracy:.4f} (n={n})"
        )
        fields = {"loss": float(loss), "accuracy": float(accuracy), "n": n}
        if server_round > 0:
            fields["round_duration_s"] = clock.round_duration()
        elif clock.offset > 0:
            fields["resumed"] = True  # 再開直後の評価 (チェックポイントのモデル)
        event_logger.emit("eval", global_round=global_round, parent_round=server_round, **fields)
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
    parser.add_argument(
        "--checkpoint", type=str, default=os.environ.get("HFL_CHECKPOINT") or None,
        help="Global モデルの保存先 (.npz)。既定: 環境変数 HFL_CHECKPOINT",
    )
    parser.add_argument("--resume", action="store_true", help="チェックポイントの続きから再開")
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
        stop_file = os.environ.get("HFL_STOP_FILE") or None

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

        # --- チェックポイントからの再開 ---
        offset = 0
        run_started_t = time.time()
        if args.resume:
            if not args.checkpoint or not os.path.exists(args.checkpoint):
                raise FileNotFoundError(
                    f"--resume requires an existing checkpoint (got {args.checkpoint!r})"
                )
            ckpt_params, ckpt_meta = load_checkpoint(args.checkpoint)
            validate_parameters(model, ckpt_params)
            set_parameters(model, ckpt_params)
            offset = int(ckpt_meta["global_round"])
            run_started_t = float(ckpt_meta.get("run_started_t", run_started_t))
            downtime = time.time() - float(ckpt_meta.get("saved_t", time.time()))
            logger.info(
                f"Resuming from checkpoint {args.checkpoint}: global_round={offset}, "
                f"saved {downtime:.0f}s ago"
            )
            event_logger.emit(
                "resume",
                global_round=offset,
                checkpoint=args.checkpoint,
                checkpoint_saved_t=ckpt_meta.get("saved_t"),
                downtime_s=round(downtime, 1),
            )
        remaining_rounds = max(int(num_rounds) - offset, 0)
        # 時間上限は最初の開始時刻 (再開をまたいで共通) から数える
        deadline = None
        if max_hours is not None:
            remaining_s = run_started_t + max_hours * 3600.0 - time.time()
            deadline = time.monotonic() + remaining_s
        event_logger.emit(
            "start",
            argv=sys.argv[1:],
            config=args.config,
            num_rounds=num_rounds,
            max_hours=max_hours,
            stop_file=stop_file,
            checkpoint=args.checkpoint,
            resume_from_round=offset if args.resume else None,
            dry_run=dry_run,
        )

        initial_params = ndarrays_to_parameters(get_parameters(model))
        device = resolve_device(cfg.get("device", "auto"))
        clock = _RoundClock(event_logger, offset=offset)
        checkpointer = _Checkpointer(
            args.checkpoint,
            every=int(srv_cfg.get("checkpoint_every", 1)),
            offset=offset,
            run_started_t=run_started_t,
            meta={"run_id": os.environ.get("HFL_RUN_ID", ""), "config": args.config},
            event_logger=event_logger,
        )

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
            # parent_round は Edge との同期用 (再開ごとに 1 から)。global_round は通し番号で、
            # 階層を下るときも保持され、全ノードのイベントで同じ軸になる
            on_fit_config_fn=lambda server_round: {
                "parent_round": server_round,
                "global_round": server_round + offset,
            },
        )
        strategy.global_round_offset = offset

        # --- サーバー起動 ---
        server = StoppableServer(
            client_manager=SimpleClientManager(),
            strategy=strategy,
            deadline=deadline,
            stop_file=stop_file,
            on_round_start=clock.round_start,
            on_round_end=checkpointer.on_round_end,
        )
        logger.info(
            f"Starting Global Server on {address} "
            f"(rounds={remaining_rounds}, from_round={offset + 1}, max_hours={max_hours}, "
            f"checkpoint={args.checkpoint}, stop_file={stop_file}, dry_run={dry_run})"
        )
        fl.server.start_server(
            server_address=address,
            server=server,
            config=ServerConfig(num_rounds=remaining_rounds, round_timeout=round_timeout),
        )
        last_round = server.completed_rounds + offset
        if checkpointer.saved_round < last_round:
            checkpointer.save(last_round, server.parameters)
        reason = server.stop_reason or "finished"
        event_logger.emit(
            "stop",
            reason=reason,
            global_round=last_round,
            elapsed_s=round(time.monotonic() - started_at, 1),
        )
        logger.info(f"Global Server finished (reason={reason}, global_round={last_round}).")
    except BaseException as exc:
        event_logger.emit_error(exc)
        raise
    finally:
        event_logger.close()


if __name__ == "__main__":
    run_main(main)
