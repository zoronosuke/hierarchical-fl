"""チェックポイント保存・読み込みと、プロセス終了処理のテスト。"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
from flwr.common import ndarrays_to_parameters
from flwr.server import SimpleClientManager

from src.core.stoppable_server import StoppableServer
from src.models.nets import create_model, get_parameters
from src.utils.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path):
    model = create_model("simplecnn", dry_run=True)
    parameters = get_parameters(model)
    path = tmp_path / "run" / "checkpoint.npz"
    save_checkpoint(path, parameters, {"global_round": 1182, "run_started_t": 1.5, "run_id": "r"})

    loaded, meta = load_checkpoint(path)
    assert meta == {"global_round": 1182, "run_started_t": 1.5, "run_id": "r"}
    assert len(loaded) == len(parameters)
    for original, restored in zip(parameters, loaded):
        assert original.dtype == restored.dtype and original.shape == restored.shape
        np.testing.assert_array_equal(original, restored)
    assert not (tmp_path / "run" / "checkpoint.npz.tmp").exists()


def test_checkpoint_overwrite_keeps_latest(tmp_path):
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(path, [np.zeros(3, dtype=np.float32)], {"global_round": 1})
    save_checkpoint(path, [np.ones(3, dtype=np.float32)], {"global_round": 2})
    loaded, meta = load_checkpoint(path)
    assert meta["global_round"] == 2
    np.testing.assert_array_equal(loaded[0], np.ones(3, dtype=np.float32))


def test_round_end_callback_receives_aggregated_parameters(monkeypatch):
    from flwr.server import Server

    from tests.test_stoppable_server import _CountingStrategy

    params = ndarrays_to_parameters([np.full(2, 7.0, dtype=np.float32)])
    # クライアント通信 (Flower の fit_round 本体) を、集約済みパラメータを返すだけに差し替える
    monkeypatch.setattr(
        Server, "fit_round", lambda self, server_round, timeout: (params, {}, ([], []))
    )

    seen = []
    server = StoppableServer(
        client_manager=SimpleClientManager(),
        strategy=_CountingStrategy(),
        on_round_end=lambda r, p: seen.append((r, p)),
    )
    server.fit(num_rounds=3, timeout=None)
    assert [r for r, _ in seen] == [1, 2, 3]
    assert all(p is params for _, p in seen)
    assert server.completed_rounds == 3
    assert server.parameters is params


def test_run_main_exits_even_with_non_daemon_threads():
    """例外で終わった main は、残ったスレッドがあっても exit 1 で終了する。"""
    script = textwrap.dedent(
        """
        import threading, time
        from src.utils.process import run_main

        def main():
            threading.Thread(target=time.sleep, args=(60,)).start()  # gRPC の残りスレッド相当
            raise RuntimeError("round failed")

        run_main(main)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 1
    assert "RuntimeError: round failed" in completed.stderr
