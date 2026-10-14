"""
ラウンド境界で正常停止できる Flower レガシー Server。

Flower の Server.fit() は num_rounds 回ループするだけで途中停止の手段がない。
このクラスは各ラウンド開始前に「時間上限」「停止イベント」「停止ファイル」を確認し、
該当すれば StopRun を投げて fit() から正常に抜ける。start_server() はその後
通常どおり disconnect_all_clients() を呼ぶので、クライアント側は reconnect
メッセージで正常終了する (Traceback を出さない)。

シグナルを使わないのは、start_server() が Flower 自身の SIGTERM/SIGINT ハンドラを
登録して即時終了してしまうため。手動停止は停止ファイル (touch) で行う。

Global Server と Edge の sub-server の両方で使う。
"""

from __future__ import annotations

import math
import os
import time
import timeit
from typing import Any, Callable

from flwr.common import Parameters
from flwr.server import History, Server
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy

from src.utils.logger import get_logger

logger = get_logger("stoppable_server")

STOP_MAX_DURATION = "max_duration"
STOP_REQUESTED = "stop_requested"
STOP_FILE = "stop_file"


class StopRun(Exception):
    """ラウンド境界での正常停止要求。"""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def next_slot(now: float, interval: float, last_start: float, align_to_clock: bool) -> float:
    """次のラウンドの開始時刻 (UNIX 時刻)。

    align_to_clock=True なら interval の倍数の時刻 (600 秒なら毎時 0, 10, 20 … 分) にそろえる。
    ラウンドが interval より長引いた場合は、次の区切りまで待つ。
    False なら「前のラウンド開始 + interval」。
    """
    if align_to_clock:
        # 前のラウンドが区切りちょうどに始まっていても、同じ区切りで2回始めない
        return max(math.ceil(now / interval), math.floor(last_start / interval) + 1) * interval
    return last_start + interval


class StoppableServer(Server):
    """deadline (time.monotonic 基準) または stop_event で停止する Server。

    round_interval_s を指定すると、2ラウンド目以降は次の開始時刻まで待ってから始める
    (学習を一定間隔にして通信量を抑える)。待機中も1秒ごとに停止要求と時間上限を確認する。
    """

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Strategy,
        deadline: float | None = None,
        stop_event: Any = None,
        stop_file: str | None = None,
        on_stop: Callable[[str], None] | None = None,
        on_round_start: Callable[[int], None] | None = None,
        on_round_end: Callable[[int, Parameters], None] | None = None,
        round_interval_s: float | None = None,
        align_to_clock: bool = True,
        on_wait: Callable[[int, float, float], None] | None = None,
        wall_clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.deadline = deadline
        self.stop_event = stop_event  # threading.Event / multiprocessing.Event
        self.stop_file = stop_file or None
        self.on_stop = on_stop
        self.on_round_start = on_round_start
        self.on_round_end = on_round_end  # 集約後のパラメータを受け取る (チェックポイント用)
        self.round_interval_s = round_interval_s if round_interval_s else None
        self.align_to_clock = align_to_clock
        self.on_wait = on_wait  # (server_round, 次の開始時刻, 待ち秒数)
        self._wall = wall_clock
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_round_start: float | None = None
        self.stop_reason: str | None = None
        self.completed_rounds = 0

    def should_stop(self) -> str | None:
        if self.stop_event is not None and self.stop_event.is_set():
            return STOP_REQUESTED
        if self.stop_file is not None and os.path.exists(self.stop_file):
            return STOP_FILE
        if self.deadline is not None and self._monotonic() >= self.deadline:
            return STOP_MAX_DURATION
        return None

    def _wait_for_slot(self, server_round: int) -> None:
        """次のラウンドの開始時刻まで待つ (1ラウンド目と、間隔未設定のときは待たない)。"""
        if self.round_interval_s is None or self._last_round_start is None:
            return
        interval = self.round_interval_s
        now = self._wall()
        wait = next_slot(now, interval, self._last_round_start, self.align_to_clock) - now
        if wait <= 0:
            return
        wait = min(wait, interval)  # 時計が飛んでも1区間より長くは待たない
        if self.deadline is not None and self._monotonic() + wait >= self.deadline:
            raise StopRun(STOP_MAX_DURATION)  # 次のラウンドの前に時間上限が来る
        if self.on_wait is not None:
            self.on_wait(server_round, now + wait, wait)
        end = self._monotonic() + wait
        while True:
            reason = self.should_stop()
            if reason is not None:
                raise StopRun(reason)
            remaining = end - self._monotonic()
            if remaining <= 0:
                return
            self._sleep(min(1.0, remaining))

    def fit_round(self, server_round: int, timeout: float | None):
        reason = self.should_stop()
        if reason is not None:
            raise StopRun(reason)
        self._wait_for_slot(server_round)
        reason = self.should_stop()
        if reason is not None:
            raise StopRun(reason)
        self._last_round_start = self._wall()
        if self.on_round_start is not None:
            self.on_round_start(server_round)
        result = super().fit_round(server_round, timeout)
        self.completed_rounds = server_round
        if self.on_round_end is not None and result is not None and result[0] is not None:
            self.on_round_end(server_round, result[0])
        return result

    def fit(self, num_rounds: int, timeout: float | None) -> tuple[History, float]:
        started = timeit.default_timer()
        try:
            return super().fit(num_rounds, timeout)
        except StopRun as exc:
            self.stop_reason = exc.reason
            logger.info(f"Stopping at round boundary: reason={exc.reason}")
            if self.on_stop is not None:
                self.on_stop(exc.reason)
            # 記録は events.jsonl が正なので History は空で返す
            return History(), timeit.default_timer() - started
