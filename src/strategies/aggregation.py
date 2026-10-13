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

from src.utils.events import EventLogger
from src.utils.logger import get_logger

logger = get_logger("strategy")


def participant_ids(results: list[tuple[ClientProxy, FitRes]]) -> list[str]:
    """FitRes.metrics から参加クライアント ID (client_id / edge_id) を集める。"""
    ids = []
    for _, fit_res in results:
        metrics = fit_res.metrics or {}
        client_id = metrics.get("client_id") or metrics.get("edge_id")
        if client_id is not None:
            ids.append(str(client_id))
    return sorted(ids)


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
        event_logger: EventLogger | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
            **kwargs,
        )
        self._strategy_params = strategy_params or {}
        self.event_logger = event_logger or EventLogger(None)
        # チェックポイントから再開した Global では、Flower のラウンド番号は 1 から数え直し、
        # 通しのラウンド番号 (global_round) = server_round + この値 になる
        self.global_round_offset = 0

    def _round_fields(self, server_round: int) -> dict[str, int]:
        """イベント記録用のラウンド番号。sub-server は child_round も付ける。"""
        return {
            "global_round": server_round + self.global_round_offset,
            "parent_round": server_round,
        }

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """集約ログを追加した上で親クラスのFedAvg集約に委譲。"""
        if not results:
            message = f"Round {server_round}: No results received."
            if self.min_fit_clients > 0:
                raise RuntimeError(message)
            logger.warning(message)
            return None, {}

        total_examples = sum(r.num_examples for _, r in results)
        logger.info(
            f"Round {server_round}: Aggregating {len(results)} results "
            f"({total_examples} total examples, {len(failures)} failures)"
        )
        self.event_logger.emit(
            "aggregate",
            **self._round_fields(server_round),
            results=len(results),
            failures=len(failures),
            total_examples=total_examples,
            participants=participant_ids(results),
        )
        if self.min_fit_clients > 0 and len(results) < self.min_fit_clients:
            raise RuntimeError(
                f"Round {server_round}: expected at least {self.min_fit_clients} "
                f"results, got {len(results)}"
            )
        return super().aggregate_fit(server_round, results, failures)
