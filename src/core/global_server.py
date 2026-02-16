"""
Global Server 起動スクリプト。

Usage:
    python -m src.core.global_server [--config config/global.yaml] [--dry-run]
"""

from __future__ import annotations

import argparse

import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig

from src.core.training import evaluate_model
from src.data.loader import create_dummy_dataloader, load_test_data
from src.models.nets import create_model, get_parameters, set_parameters
from src.strategies.aggregation import create_strategy
from src.utils.config import load_yaml, resolve_device
from src.utils.logger import get_logger

logger = get_logger("global_server")


def _build_evaluate_fn(model, testloader, device):
    """サーバーサイド評価関数を構築する。"""
    def evaluate_fn(server_round, parameters_ndarrays, config):
        set_parameters(model, parameters_ndarrays)
        loss, accuracy, n = evaluate_model(model, testloader, device)
        logger.info(
            f"[Global Eval] Round {server_round}: "
            f"loss={loss:.4f}, accuracy={accuracy:.4f} (n={n})"
        )
        return loss, {"accuracy": accuracy}
    return evaluate_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="HFL Global Server")
    parser.add_argument("--config", type=str, default="config/global.yaml")
    parser.add_argument("--dry-run", action="store_true", help="疎通確認モード")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.dry_run:
        cfg["mode"] = "dry_run"
    dry_run = cfg.get("mode", "production") == "dry_run"

    # --- サーバー設定 ---
    srv_cfg = cfg.get("server", {})
    address = srv_cfg.get("address", "0.0.0.0:8080")
    round_timeout = srv_cfg.get("round_timeout", 300.0)
    dr_cfg = cfg.get("dry_run", {})
    num_rounds = dr_cfg.get("num_rounds", 1) if dry_run else srv_cfg.get("num_rounds", 10)

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
        evaluate_fn = _build_evaluate_fn(model, testloader, device)

    # --- Strategy ---
    strat_cfg = cfg.get("strategy", {})
    strategy = create_strategy(
        name=strat_cfg.get("name", "fedavg"),
        params=strat_cfg.get("params"),
        initial_parameters=initial_params,
        evaluate_fn=evaluate_fn,
        fraction_fit=srv_cfg.get("fraction_fit", 1.0),
        fraction_evaluate=0.0,  # 分散評価は無効 (evaluate_fn でサーバーサイド評価を行う)
        min_fit_clients=srv_cfg.get("min_fit_clients", 1),
        min_available_clients=srv_cfg.get("min_available_clients", 1),
    )

    # --- サーバー起動 ---
    logger.info(
        f"Starting Global Server on {address} "
        f"(rounds={num_rounds}, dry_run={dry_run})"
    )
    fl.server.start_server(
        server_address=address,
        config=ServerConfig(num_rounds=num_rounds, round_timeout=round_timeout),
        strategy=strategy,
    )
    logger.info("Global Server finished.")


if __name__ == "__main__":
    main()
