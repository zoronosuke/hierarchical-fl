"""
集約戦略モジュール。

FedAvg をベースとし、contribution_factor による重み調整をサポート。
将来的に FedProx, SCAFFOLD 等を追加するための抽象基底クラスを提供。
"""

from __future__ import annotations

from typing import Any, Optional

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy

from src.utils.logger import get_logger

logger = get_logger("strategy")


# =========================================================================
# Strategy レジストリ (名前 → クラスのマッピング)
# =========================================================================
_STRATEGY_REGISTRY: dict[str, type] = {}


def register_strategy(name: str):
    """戦略クラスをレジストリに登録するデコレータ。"""
    def decorator(cls):
        _STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


def create_strategy(
    name: str,
    params: dict[str, Any] | None = None,
    **kwargs,
) -> fl.server.strategy.Strategy:
    """
    名前から Flower Strategy インスタンスを生成する。

    Parameters
    ----------
    name : str
        戦略名 (e.g., "fedavg")
    params : dict
        戦略固有のパラメータ
    **kwargs
        Flower Strategy 共通パラメータ (fraction_fit, min_fit_clients 等)
    """
    params = params or {}
    if name not in _STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: {name}. Available: {list(_STRATEGY_REGISTRY.keys())}"
        )
    return _STRATEGY_REGISTRY[name](strategy_params=params, **kwargs)


# =========================================================================
# FedAvg 実装
# =========================================================================

@register_strategy("fedavg")
class WeightedFedAvg(fl.server.strategy.FedAvg):
    """
    contribution_factor 対応の FedAvg。

    クライアントが返す num_examples に contribution_factor が乗算済みであることを
    前提とし、Flower 標準の FedAvg 集約ロジック（データ件数ベースの加重平均）に委譲する。
    """

    def __init__(
        self,
        strategy_params: dict[str, Any] | None = None,
        initial_parameters: Parameters | None = None,
        evaluate_fn=None,
        **kwargs,
    ) -> None:
        super().__init__(
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
            **kwargs,
        )
        self._strategy_params = strategy_params or {}

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """集約ログを追加した上で親クラスのFedAvg集約に委譲。"""
        if not results:
            logger.warning(f"Round {server_round}: No results received.")
            return None, {}

        total_examples = sum(r.num_examples for _, r in results)
        logger.info(
            f"Round {server_round}: Aggregating {len(results)} results "
            f"({total_examples} total examples, {len(failures)} failures)"
        )
        return super().aggregate_fit(server_round, results, failures)
