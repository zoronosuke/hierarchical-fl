"""flat FL の中継 (src/core/tcp_relay.py) のテスト。flwr は使わない。"""

from __future__ import annotations

import json
import socket
import threading
import time

from src.core.tcp_relay import TcpRelay, relay_summary
from src.utils.events import EventLogger


def _echo_server() -> tuple[socket.socket, int, threading.Thread]:
    """受けたバイトをそのまま返し、クライアントが半閉じしたら自分も閉じる。"""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    listener.settimeout(5)

    def echo(conn: socket.socket) -> None:
        with conn:
            try:
                while True:
                    data = conn.recv(65536)
                    if not data:
                        break
                    conn.sendall(data)
            except OSError:
                pass  # 中継側が閉じた

    def serve():
        while True:
            try:
                conn, _ = listener.accept()
            except (socket.timeout, OSError):
                return
            threading.Thread(target=echo, args=(conn,), daemon=True).start()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return listener, listener.getsockname()[1], thread


def _events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_relay_forwards_both_directions_and_counts_bytes(tmp_path):
    listener, upstream_port, _ = _echo_server()
    events_path = tmp_path / "events.jsonl"
    event_logger = EventLogger(str(events_path), node_id="edge_01", proc="relay")
    relay = TcpRelay(
        "127.0.0.1:0", f"127.0.0.1:{upstream_port}", node_id="edge_01", event_logger=event_logger
    )
    relay.start()
    try:
        payload = bytes(range(256)) * 400  # 102400 B (バッファより大きい)
        with socket.create_connection(("127.0.0.1", relay.bound_port), timeout=5) as client:
            client.sendall(payload)
            received = b""
            while len(received) < len(payload):
                chunk = client.recv(65536)
                assert chunk, "relay closed early"
                received += chunk
            assert received == payload
            assert relay.active_connections() == 1
            client.shutdown(socket.SHUT_WR)  # 半閉じ → echo が閉じる → 接続が終わる
            assert client.recv(1) == b""
        assert relay.wait_idle(5.0)
        assert relay.total_connections == 1
        assert relay.total_bytes_up == len(payload)
        assert relay.total_bytes_down == len(payload)
    finally:
        relay.stop()
        listener.close()
        event_logger.close()

    kinds = [e["event"] for e in _events(events_path)]
    assert kinds == ["relay_connect", "relay_disconnect"]
    disconnect = _events(events_path)[-1]
    assert disconnect["bytes_up"] == len(payload) and disconnect["bytes_down"] == len(payload)
    assert disconnect["connections"] == 0 and disconnect["duration_s"] >= 0
    assert relay_summary(relay) == {
        "connections": 1, "bytes_up": len(payload), "bytes_down": len(payload)
    }


def test_relay_handles_concurrent_connections_and_stop(tmp_path):
    listener, upstream_port, _ = _echo_server()
    relay = TcpRelay("127.0.0.1:0", f"127.0.0.1:{upstream_port}", node_id="leaf_01")
    relay.start()
    clients = []
    try:
        for i in range(3):
            client = socket.create_connection(("127.0.0.1", relay.bound_port), timeout=5)
            client.sendall(f"hello {i}".encode())
            assert client.recv(64) == f"hello {i}".encode()
            clients.append(client)
        deadline = time.monotonic() + 5
        while relay.active_connections() < 3 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert relay.active_connections() == 3
        assert not relay.wait_idle(0.3)  # まだ 3 本開いている
        # wait_idle(after_first_connection=True) は接続が来る前には idle と言わない
        fresh = TcpRelay("127.0.0.1:0", f"127.0.0.1:{upstream_port}")
        fresh.start()
        assert not fresh.wait_idle(0.2, after_first_connection=True)
        fresh.stop()
    finally:
        relay.stop()  # 残っている接続を閉じる
        for client in clients:
            try:
                assert client.recv(1) == b""  # 中継が閉じたのでクライアント側も EOF
            except OSError:
                pass
            client.close()
        listener.close()
    assert relay.active_connections() == 0
    assert relay.total_connections == 3


def test_relay_closes_child_when_upstream_is_unreachable(tmp_path):
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    closed_port = probe.getsockname()[1]
    probe.close()
    events_path = tmp_path / "events.jsonl"
    event_logger = EventLogger(str(events_path), node_id="edge_02", proc="relay")
    relay = TcpRelay("127.0.0.1:0", f"127.0.0.1:{closed_port}", event_logger=event_logger)
    relay.start()
    try:
        with socket.create_connection(("127.0.0.1", relay.bound_port), timeout=5) as client:
            client.settimeout(5)
            try:
                assert client.recv(1) == b""
            except ConnectionError:
                pass
    finally:
        relay.stop()
        event_logger.close()
    assert [e["event"] for e in _events(events_path)] == ["relay_connect_failed"]
    assert relay.total_connections == 0
