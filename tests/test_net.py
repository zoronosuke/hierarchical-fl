"""待受確認と gRPC ノイズ抑制のテスト。"""

from __future__ import annotations

import socket
import threading
import time

from src.utils import net


def test_wait_for_port_returns_true_when_listening():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        started = time.monotonic()
        assert net.wait_for_port(f"0.0.0.0:{port}", timeout=5.0)
        assert time.monotonic() - started < 2.0
    finally:
        server.close()


def test_wait_for_port_times_out_on_closed_port():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()  # 閉じたので誰も待受していない
    started = time.monotonic()
    assert not net.wait_for_port(f"127.0.0.1:{port}", timeout=0.6, interval=0.1)
    assert 0.5 <= time.monotonic() - started < 3.0


def test_grpc_noise_filter_only_swallows_channel_closed(monkeypatch):
    seen = []
    monkeypatch.setattr(threading, "excepthook", lambda args: seen.append(args.exc_type))
    monkeypatch.setattr(net, "_filter_installed", False)
    net.install_grpc_noise_filter()

    def raise_noise():
        raise ValueError("Cannot monitor channel state: Channel closed!")

    def raise_other():
        raise RuntimeError("real problem")

    for target in (raise_noise, raise_other):
        thread = threading.Thread(target=target)
        thread.start()
        thread.join()
    assert seen == [RuntimeError]
