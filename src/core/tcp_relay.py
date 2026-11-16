"""
子ノードの TCP 接続を親へそのまま中継する (flat FL 用の中継ノード)。

HFL の中継ノードは Flower の sub-server で子を集約してから親へ 1 モデルを送るが、
flat FL の中継は子ごとの Flower gRPC 接続を中身を読まずに親へ通す。Global と各クライアントは
Flower で直接やりとりしている状態になり、物理的には同じ親子リンクを経由する。

接続ごとに子→親 (bytes_up) と親→子 (bytes_down) のバイト数を数え、イベント
(relay_connect / relay_disconnect) として記録する。TCP の中身なので、Jetson 側の記録
(scripts/hfl_sampler.py の links) と同じ尺度になる。
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from src.utils.events import EventLogger
from src.utils.logger import get_logger

logger = get_logger("tcp_relay")

_BUFFER_SIZE = 65536
_ACCEPT_POLL_S = 0.5


def split_address(address: str) -> tuple[str, int]:
    host, port_text = address.rsplit(":", 1)
    host = host.strip("[]")
    return host, int(port_text)


@dataclass
class RelayConnection:
    peer: str
    child: socket.socket
    upstream: socket.socket
    started: float = field(default_factory=time.monotonic)
    bytes_up: int = 0
    bytes_down: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def close(self) -> None:
        for sock in (self.child, self.upstream):
            try:
                sock.close()
            except OSError:
                pass


class TcpRelay:
    """listen_address で受けた接続を upstream_address へ中継する。"""

    def __init__(
        self,
        listen_address: str,
        upstream_address: str,
        *,
        node_id: str = "",
        event_logger: EventLogger | None = None,
        connect_timeout: float = 10.0,
    ) -> None:
        self.listen_address = listen_address
        self.upstream_address = upstream_address
        self.node_id = node_id
        self.event_logger = event_logger or EventLogger(None)
        self.connect_timeout = connect_timeout
        self._listener: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self._lock = threading.Lock()
        self._connections: dict[int, RelayConnection] = {}
        self._workers: list[threading.Thread] = []
        self.total_connections = 0
        self.total_bytes_up = 0
        self.total_bytes_down = 0

    # ------------------------------------------------------------------ 公開 API
    def start(self) -> None:
        host, port = split_address(self.listen_address)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(64)
        listener.settimeout(_ACCEPT_POLL_S)
        self._listener = listener
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name=f"{self.node_id}-relay-accept", daemon=True
        )
        self._accept_thread.start()
        logger.info(
            f"[{self.node_id}] TCP relay listening on {self.listen_address} "
            f"-> {self.upstream_address}"
        )

    @property
    def bound_port(self) -> int:
        if self._listener is None:
            raise RuntimeError("relay is not started")
        return self._listener.getsockname()[1]

    def active_connections(self) -> int:
        with self._lock:
            return len(self._connections)

    def wait_idle(self, timeout: float | None, *, after_first_connection: bool = False) -> bool:
        """中継中の接続が無くなるまで待つ。after_first_connection なら最初の接続も待つ。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                idle = not self._connections and (
                    not after_first_connection or self.total_connections > 0
                )
            if idle:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.1)

    def stop(self) -> None:
        """待受を閉じ、残っている接続を閉じる。"""
        self._stopped.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=5)
        with self._lock:
            connections = list(self._connections.values())
        for conn in connections:
            conn.close()
        for worker in list(self._workers):
            worker.join(timeout=5)

    # ------------------------------------------------------------------ 内部
    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._stopped.is_set():
            try:
                child, addr = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stopped.is_set():
                    return
                logger.warning(f"[{self.node_id}] accept failed; retrying")
                time.sleep(_ACCEPT_POLL_S)
                continue
            peer = f"{addr[0]}:{addr[1]}"
            worker = threading.Thread(
                target=self._serve, args=(child, peer),
                name=f"{self.node_id}-relay-{peer}", daemon=True,
            )
            self._workers.append(worker)
            worker.start()

    def _serve(self, child: socket.socket, peer: str) -> None:
        host, port = split_address(self.upstream_address)
        try:
            upstream = socket.create_connection((host, port), timeout=self.connect_timeout)
        except OSError as exc:
            logger.warning(
                f"[{self.node_id}] cannot reach upstream {self.upstream_address} for {peer}: {exc}"
            )
            self.event_logger.emit("relay_connect_failed", peer=peer, message=str(exc)[:200])
            child.close()
            return
        upstream.settimeout(None)
        child.settimeout(None)
        for sock in (child, upstream):
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
        conn = RelayConnection(peer=peer, child=child, upstream=upstream)
        with self._lock:
            self._connections[id(conn)] = conn
            self.total_connections += 1
            count = len(self._connections)
        logger.info(f"[{self.node_id}] relay connection from {peer} (active={count})")
        self.event_logger.emit("relay_connect", peer=peer, connections=count)

        up = threading.Thread(
            target=self._pump, args=(conn, child, upstream, "up"), daemon=True,
            name=f"{self.node_id}-relay-up-{peer}",
        )
        down = threading.Thread(
            target=self._pump, args=(conn, upstream, child, "down"), daemon=True,
            name=f"{self.node_id}-relay-down-{peer}",
        )
        up.start()
        down.start()
        up.join()
        down.join()
        conn.close()
        with self._lock:
            self._connections.pop(id(conn), None)
            self.total_bytes_up += conn.bytes_up
            self.total_bytes_down += conn.bytes_down
            count = len(self._connections)
        duration = time.monotonic() - conn.started
        logger.info(
            f"[{self.node_id}] relay connection closed {peer}: up={conn.bytes_up} B, "
            f"down={conn.bytes_down} B, {duration:.1f}s (active={count})"
        )
        self.event_logger.emit(
            "relay_disconnect",
            peer=peer,
            bytes_up=conn.bytes_up,
            bytes_down=conn.bytes_down,
            duration_s=round(duration, 3),
            connections=count,
        )

    @staticmethod
    def _pump(
        conn: RelayConnection, src: socket.socket, dst: socket.socket, direction: str
    ) -> None:
        """src から読んで dst へ書く。EOF で dst を半閉じし、エラーなら両方閉じる。"""
        try:
            while True:
                data = src.recv(_BUFFER_SIZE)
                if not data:
                    try:
                        dst.shutdown(socket.SHUT_WR)
                    except OSError:
                        pass
                    return
                dst.sendall(data)
                with conn.lock:
                    if direction == "up":
                        conn.bytes_up += len(data)
                    else:
                        conn.bytes_down += len(data)
        except OSError:
            conn.close()  # 片側が落ちたら反対側のポンプも recv が失敗して終わる


def relay_summary(relay: TcpRelay) -> dict[str, Any]:
    return {
        "connections": relay.total_connections,
        "bytes_up": relay.total_bytes_up,
        "bytes_down": relay.total_bytes_down,
    }
