"""常駐子連合の同期・安全性テスト。"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters

from src.core.sub_federation import (
    ParentModelRequest,
    SynchronousSubFedAvg,
    parent_round_for_child,
)
from src.models.nets import (
    copy_parameters,
    create_model,
    get_parameters,
    set_parameters,
)


@pytest.mark.parametrize(
    ("child_round", "sub_rounds", "parent_round"),
    [(1, 1, 1), (2, 1, 2), (1, 3, 1), (3, 3, 1), (4, 3, 2), (6, 3, 2)],
)
def test_parent_child_round_mapping(child_round, sub_rounds, parent_round):
    assert parent_round_for_child(child_round, sub_rounds) == parent_round


def test_parameter_copy_is_independent():
    original = [np.arange(4, dtype=np.float32)]
    copied = copy_parameters(original)
    original[0][0] = 99
    assert copied[0][0] == 0


def test_parameter_count_shape_and_dtype_are_validated():
    model = create_model("simplecnn", dry_run=True)
    parameters = get_parameters(model)

    with pytest.raises(ValueError, match="count mismatch"):
        set_parameters(model, parameters[:-1])

    wrong_shape = copy_parameters(parameters)
    wrong_shape[0] = wrong_shape[0].reshape(-1)
    with pytest.raises(ValueError, match="shape mismatch"):
        set_parameters(model, wrong_shape)

    wrong_dtype = copy_parameters(parameters)
    wrong_dtype[0] = wrong_dtype[0].astype(np.float64)
    with pytest.raises(ValueError, match="dtype mismatch"):
        set_parameters(model, wrong_dtype)


def _strategy(expected_clients=3, sub_rounds=1, timeout=0.05):
    model = create_model("simplecnn", dry_run=True)
    return SynchronousSubFedAvg(
        edge_id="edge_test",
        sub_rounds=sub_rounds,
        expected_clients=expected_clients,
        parent_queue=queue.Queue(),
        result_queue=queue.Queue(),
        event_queue=queue.Queue(),
        parent_model_timeout=timeout,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=expected_clients,
        min_available_clients=expected_clients,
        min_evaluate_clients=expected_clients,
        initial_parameters=ndarrays_to_parameters(get_parameters(model)),
    )


class _ClientManager:
    def num_available(self):
        return 3

    def sample(self, num_clients, min_num_clients):
        return [object()] * num_clients


def test_parent_model_timeout_contains_diagnostics():
    strategy = _strategy()
    model = create_model("simplecnn", dry_run=True)
    with pytest.raises(TimeoutError) as raised:
        strategy.configure_fit(
            1,
            ndarrays_to_parameters(get_parameters(model)),
            _ClientManager(),
        )
    message = str(raised.value)
    for expected in (
        "edge_id=edge_test",
        "parent_round=1",
        "child_round=1",
        "connected_clients=3",
        "expected_clients=3",
        "waited=",
    ):
        assert expected in message


def test_round_mismatch_is_rejected():
    strategy = _strategy()
    model = create_model("simplecnn", dry_run=True)
    strategy.parent_queue.put(ParentModelRequest(2, get_parameters(model)))
    with pytest.raises(RuntimeError, match="round mismatch"):
        strategy.configure_fit(
            1,
            ndarrays_to_parameters(get_parameters(model)),
            _ClientManager(),
        )


def test_incomplete_child_round_is_rejected():
    strategy = _strategy(expected_clients=3)
    model = create_model("simplecnn", dry_run=True)
    fit_res = FitRes(
        status=Status(Code.OK, ""),
        parameters=ndarrays_to_parameters(get_parameters(model)),
        num_examples=10,
        metrics={},
    )
    with pytest.raises(RuntimeError, match="Incomplete child round"):
        strategy.aggregate_fit(1, [(object(), fit_res), (object(), fit_res)], [])


def test_final_result_only_after_last_sub_round():
    strategy = _strategy(expected_clients=1, sub_rounds=2)
    model = create_model("simplecnn", dry_run=True)
    fit_res = FitRes(
        status=Status(Code.OK, ""),
        parameters=ndarrays_to_parameters(get_parameters(model)),
        num_examples=10,
        metrics={},
    )
    strategy.aggregate_fit(1, [(object(), fit_res)], [])
    with pytest.raises(queue.Empty):
        strategy.result_queue.get_nowait()

    strategy.aggregate_fit(2, [(object(), fit_res)], [])
    result = strategy.result_queue.get_nowait()
    assert result.parent_round == 1
    assert result.child_round == 2


def test_waiting_thread_does_not_finish_before_result():
    result_queue: queue.Queue[int] = queue.Queue()
    finished = threading.Event()

    def wait_for_result():
        result_queue.get(timeout=1)
        finished.set()

    thread = threading.Thread(target=wait_for_result)
    thread.start()
    time.sleep(0.05)
    assert not finished.is_set()
    result_queue.put(1)
    thread.join(timeout=1)
    assert finished.is_set()
