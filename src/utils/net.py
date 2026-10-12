"""接続先の待受確認と、gRPC 接続再試行時のノイズ抑制。"""

from __future__ import annotations

import socket
import threading
import time

from src.utils.logger import get_logger

logger = get_logger("net")


def wait_for_port(address: str, timeout: float, interval: float = 0.5) -> bool:
    """host:port が TCP 接続を受け付けるまで待つ。timeout 内に開けば True。

    Flower Client は接続先が未起動だと UNAVAILABLE で失敗し、そのリトライ時に
    gRPC の接続監視スレッドが Traceback を出す。先にポートを待てば通常はこの経路に入らない。
    """
    host, port_text = address.rsplit(":", 1)
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    deadline = time.monotonic() + timeout
    attempt = 0
    while True:
        attempt += 1
        try:
            with socket.create_connection((host, int(port_text)), timeout=2.0):
                if attempt > 1:
                    logger.info(f"{address} is now accepting connections (attempt={attempt})")
                return True
        except OSError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(f"{address} did not accept connections within {timeout:.0f}s")
                return False
            if attempt == 1:
                logger.info(f"Waiting for {address} to accept connections (up to {timeout:.0f}s)")
            time.sleep(min(interval, remaining))


_GRPC_NOISE = "Cannot monitor channel state"
_filter_installed = False


def install_grpc_noise_filter() -> None:
    """接続失敗後にチャネルを閉じたときの gRPC 監視スレッドの ValueError を握りつぶす。

    grpc._channel._poll_connectivity が daemon スレッドで
    "ValueError: Cannot monitor channel state: Channel closed!" を出すが、動作に影響はない。
    Traceback がログに残ると成功判定 (Traceback なし) を誤らせるので、この1種類だけ抑制する。
    """
    global _filter_installed
    if _filter_installed:
        return
    previous = threading.excepthook

    def hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is ValueError and _GRPC_NOISE in str(args.exc_value):
            logger.debug(f"Ignored gRPC connectivity-watch noise in {args.thread}")
            return
        previous(args)

    threading.excepthook = hook
    _filter_installed = True
