"""StoppableServer のラウンド境界停止と、sub-server の停止要求ポーリングのテスト。"""

from __future__ import annotations

import queue
import threading
import time

import pytest
from flwr.common import Parameters, ndarrays_to_parameters
from flwr.server import History, SimpleClientManager
from flwr.server.strategy import Strategy

from src.core.stoppable_server import (
    STOP_FILE,
    STOP_MAX_DURATION,
    STOP_REQUESTED,
    StoppableServer,
    StopRun,
)
from src.core.sub_federation import SynchronousSubFedAvg
from src.models.nets import create_model, get_parameters


class _CountingStrategy(Strategy):
    """クライアント無しで fit ループを回す最小 Strategy。configure_fit の回数を数える。"""

    def __init__(self, on_round=None) -> None:
        self.configure_calls = 0
        self.on_round = on_round

    def initialize_parameters(self, client_manager):
        return Parameters(tensors=[], tensor_type="")

    def configure_fit(self, server_round, parameters, client_manager):
        self.configure_calls += 1
        if self.on_round is not None:
            self.on_round(server_round)
        return []

    def aggregate_fit(self, server_round, results, failures):
        return None, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None


def _server(strategy, **kwargs) -> StoppableServer:
    return StoppableServer(client_manager=SimpleClientManager(), strategy=strategy, **kwargs)


def test_runs_all_rounds_without_stop():
    strategy = _CountingStrategy()
    server = _server(strategy)
    history, elapsed = server.fit(num_rounds=3, timeout=None)
    assert strategy.configure_calls == 3
    assert server.stop_reason is None
    assert isinstance(history, History)
    assert elapsed >= 0


def test_deadline_in_the_past_stops_before_first_round():
    reasons = []
    strategy = _CountingStrategy()
    server = _server(strategy, deadline=time.monotonic() - 1, on_stop=reasons.append)
    history, _ = server.fit(num_rounds=1_000_000, timeout=None)
    assert strategy.configure_calls == 0
    assert server.stop_reason == STOP_MAX_DURATION
    assert reasons == [STOP_MAX_DURATION]
    assert isinstance(history, History)


def test_stop_event_is_honoured_at_round_boundary():
    stop_event = threading.Event()
    strategy = _CountingStrategy(on_round=lambda r: stop_event.set() if r == 2 else None)
    server = _server(strategy, stop_event=stop_event)
    server.fit(num_rounds=1_000_000, timeout=None)
    assert strategy.configure_calls == 2
    assert server.stop_reason == STOP_REQUESTED


def test_stop_file_is_honoured(tmp_path):
    stop_file = tmp_path / "STOP"
    strategy = _CountingStrategy(on_round=lambda r: stop_file.touch() if r == 1 else None)
    server = _server(strategy, stop_file=str(stop_file))
    server.fit(num_rounds=1_000_000, timeout=None)
    assert strategy.configure_calls == 1
    assert server.stop_reason == STOP_FILE


def test_round_start_callback_receives_round_number():
    rounds = []
    server = _server(_CountingStrategy(), on_round_start=rounds.append)
    server.fit(num_rounds=2, timeout=None)
    assert rounds == [1, 2]


def test_sub_server_parent_wait_raises_stop_run_when_stop_event_set():
    model = create_model("simplecnn", dry_run=True)
    stop_event = threading.Event()
    stop_event.set()
    strategy = SynchronousSubFedAvg(
        edge_id="edge_test",
        sub_rounds=1,
        expected_clients=1,
        parent_queue=queue.Queue(),
        result_queue=queue.Queue(),
        event_queue=queue.Queue(),
        parent_model_timeout=30.0,
        stop_event=stop_event,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=1,
        min_available_clients=1,
        min_evaluate_clients=1,
        initial_parameters=ndarrays_to_parameters(get_parameters(model)),
    )

    class _Manager:
        def num_available(self):
            return 1

        def sample(self, num_clients, min_num_clients):
            return [object()] * num_clients

    started = time.monotonic()
    with pytest.raises(StopRun) as raised:
        strategy.configure_fit(1, ndarrays_to_parameters(get_parameters(model)), _Manager())
    assert raised.value.reason == STOP_REQUESTED
    assert time.monotonic() - started < 5.0  # parent_model_timeout を待たずに抜ける
