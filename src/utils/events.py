"""
JSONL イベント記録。

長時間実験の証拠と可視化のために、各プロセスが「いつ・どのラウンドで・何をしたか」を
1行1イベントの JSON Lines で追記する。電源断に備えて1行ごとに flush + fsync する。

環境変数で有効化する (未設定なら no-op):
    HFL_EVENT_LOG   出力ファイル (例: logs/<run_id>/events-202.jsonl)
    HFL_RUN_ID      実験 ID
    HFL_NODE_ID     ノード ID (global / edge_01 / leaf_02 ...)
    HFL_NODE_NUM    ノード番号 (201..207)

spawn された子プロセスは環境変数を継承するので、各プロセスで get_event_logger() を
呼ぶだけで同じファイルに追記できる。Edge では3プロセスが同じファイルに書くため、
行が混ざらないよう書き込み中はファイルロックを取る (Linux: flock, Windows: msvcrt)。
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

from src.utils.logger import get_logger

logger = get_logger("events")

if sys.platform == "win32":
    import msvcrt

    @contextmanager
    def _locked(fd: int) -> Iterator[None]:
        # Windows の O_APPEND は他プロセスとの同時書き込みで混ざるので、先頭1バイトをロックに使う
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    @contextmanager
    def _locked(fd: int) -> Iterator[None]:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)

ENV_PATH = "HFL_EVENT_LOG"
ENV_RUN_ID = "HFL_RUN_ID"
ENV_NODE_ID = "HFL_NODE_ID"
ENV_NODE_NUM = "HFL_NODE_NUM"

_TRACEBACK_LIMIT = 2000


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return str(value)


class EventLogger:
    """1プロセス分のイベント記録。path が None なら何もしない。"""

    def __init__(
        self,
        path: str | None,
        *,
        run_id: str = "",
        node_id: str = "",
        node_num: int | None = None,
        proc: str = "",
    ) -> None:
        self.path = path
        self.run_id = run_id
        self.node_id = node_id
        self.node_num = node_num
        self.proc = proc
        self.host = socket.gethostname()
        self._fd: int | None = None
        if path:
            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory, exist_ok=True)
            self._fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)

    @property
    def enabled(self) -> bool:
        return self._fd is not None

    def emit(self, event: str, **fields: Any) -> None:
        """イベントを1行追記する。失敗しても学習を止めない。"""
        if self._fd is None:
            return
        now = time.time()
        record: dict[str, Any] = {
            "ts": datetime.fromtimestamp(now).astimezone().isoformat(timespec="milliseconds"),
            "t": round(now, 3),
            "run_id": self.run_id,
            "node": self.node_id,
            "node_num": self.node_num,
            "host": self.host,
            "proc": self.proc,
            "pid": os.getpid(),
            "event": event,
        }
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
        try:
            with _locked(self._fd):
                os.write(self._fd, line.encode("utf-8"))  # O_APPEND なので常に末尾に書く
                os.fsync(self._fd)
        except OSError as exc:
            logger.warning(f"Failed to write event '{event}' to {self.path}: {exc}")

    def emit_error(self, exc: BaseException, **fields: Any) -> None:
        """例外を error イベントとして記録する (traceback は末尾のみ)。"""
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.emit(
            "error",
            exc_type=type(exc).__name__,
            message=str(exc)[:500],
            traceback=tb[-_TRACEBACK_LIMIT:],
            **fields,
        )

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            finally:
                self._fd = None


def get_event_logger(proc: str) -> EventLogger:
    """環境変数から EventLogger を生成する。HFL_EVENT_LOG が無ければ no-op。"""
    path = os.environ.get(ENV_PATH) or None
    node_num_text = os.environ.get(ENV_NODE_NUM, "")
    node_num = int(node_num_text) if node_num_text.isdigit() else None
    return EventLogger(
        path,
        run_id=os.environ.get(ENV_RUN_ID, ""),
        node_id=os.environ.get(ENV_NODE_ID, ""),
        node_num=node_num,
        proc=proc,
    )
