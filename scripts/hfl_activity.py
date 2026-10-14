"""
各ノードが「今なにをしているか」と、ラウンドごとのタイムライン (ガント) をイベントから作る。
監視役 (hfl_monitor.py) と再生ページ用データ (build_replay.py) で共用する。

イベントは src/utils/events.py が記録するもの (proc ごと):
    global:          start / resume / round_start / aggregate / checkpoint / eval / wait /
                     stop / error
    edge:            start / parent_fit_start / parent_fit_end / stop / error
    sub_server:      start / parent_model_received / round_start / aggregate / child_result_sent /
                     eval / stop / error
    internal_client: start / fit_start / fit_end / stop / error
    leaf:            start / fit_start / fit_end / stop / error
"""

from __future__ import annotations

import time
from typing import Any

# 画面で色分けに使う段階
PHASE_IDLE = "idle"          # 待機 (次のモデル・次のラウンド待ち)
PHASE_TRAIN = "train"        # 学習中 / 子の学習待ち
PHASE_AGGREGATE = "aggregate"  # 集約・評価・返送
PHASE_STARTING = "starting"  # 起動・接続待ち
PHASE_STOPPED = "stopped"
PHASE_ERROR = "error"

# タイムラインに出す区間: (proc, 開始イベント, 終了イベント, 表示名)
TIMELINE_SPANS = [
    ("global", "round_start", "eval", "ラウンド (配布〜集約・評価)"),
    ("edge", "parent_fit_start", "parent_fit_end", "子へ配布〜集約して返送"),
    ("internal_client", "fit_start", "fit_end", "学習 (Internal)"),
    ("leaf", "fit_start", "fit_end", "学習"),
]


def is_activity_event(event: dict[str, Any]) -> bool:
    """「今なにをしているか」の判定に使うイベントか。

    sub-server の eval は、子の結果を親へ返した後に記録される確認用の評価なので使わない
    (使うと、返送後の待機中も「集約して親へ返送」と表示されてしまう)。
    """
    return not (event.get("proc") == "sub_server" and event.get("event") == "eval")


def _hhmm(t: float | None) -> str:
    return time.strftime("%H:%M", time.localtime(t)) if t else "--:--"


def _finished(event: dict[str, Any] | None) -> dict[str, str] | None:
    if event is None:
        return None
    if event.get("event") == "error":
        return {"label": "異常終了", "phase": PHASE_ERROR}
    if event.get("event") == "stop":
        return {"label": f"停止 ({event.get('reason', '')})", "phase": PHASE_STOPPED}
    return None


def describe_global(event: dict[str, Any] | None) -> dict[str, str]:
    if event is None:
        return {"label": "未起動", "phase": PHASE_STOPPED}
    done = _finished(event)
    if done:
        return done
    kind, rnd = event.get("event"), event.get("global_round")
    if kind == "wait":
        return {"label": f"待機 (次のラウンド {_hhmm(event.get('next_round_t'))})",
                "phase": PHASE_IDLE}
    if kind == "round_start":
        return {"label": f"ラウンド {rnd}: 下位の学習待ち", "phase": PHASE_TRAIN}
    if kind in ("aggregate", "checkpoint"):
        return {"label": f"ラウンド {rnd}: 集約・保存", "phase": PHASE_AGGREGATE}
    if kind == "eval":
        if event.get("resumed"):
            return {"label": f"再開 (ラウンド {rnd} から)", "phase": PHASE_STARTING}
        return {"label": f"ラウンド {rnd}: 評価完了", "phase": PHASE_IDLE}
    if kind in ("start", "resume"):
        return {"label": "起動・接続待ち", "phase": PHASE_STARTING}
    return {"label": str(kind), "phase": PHASE_IDLE}


def describe_edge(edge: dict[str, Any] | None, sub: dict[str, Any] | None) -> dict[str, str]:
    """Edge / 中継ノード: 親とのやりとり (edge) と子への配布 (sub_server) の新しい方で判定。"""
    events = [e for e in (edge, sub) if e is not None]
    if not events:
        return {"label": "未起動", "phase": PHASE_STOPPED}
    latest = max(events, key=lambda e: e.get("t", 0))
    done = _finished(latest)
    if done:
        return done
    kind, rnd = latest.get("event"), latest.get("global_round")
    if kind in ("parent_fit_start", "parent_model_received", "round_start"):
        return {"label": f"ラウンド {rnd}: 子へ配布・学習待ち", "phase": PHASE_TRAIN}
    if kind in ("aggregate", "child_result_sent", "eval"):
        return {"label": f"ラウンド {rnd}: 集約して親へ返送", "phase": PHASE_AGGREGATE}
    if kind == "parent_fit_end":
        return {"label": "待機 (次のモデル待ち)", "phase": PHASE_IDLE}
    if kind == "start":
        return {"label": "起動・接続待ち", "phase": PHASE_STARTING}
    return {"label": str(kind), "phase": PHASE_IDLE}


def describe_client(event: dict[str, Any] | None) -> dict[str, str]:
    """Leaf / Internal Client。"""
    if event is None:
        return {"label": "未起動", "phase": PHASE_STOPPED}
    done = _finished(event)
    if done:
        return done
    kind, rnd = event.get("event"), event.get("global_round")
    if kind == "fit_start":
        return {"label": f"ラウンド {rnd}: 学習中", "phase": PHASE_TRAIN}
    if kind == "fit_end":
        return {"label": "待機 (次のモデル待ち)", "phase": PHASE_IDLE}
    if kind == "start":
        return {"label": "起動・接続待ち", "phase": PHASE_STARTING}
    return {"label": str(kind), "phase": PHASE_IDLE}


def describe(role: str, proc_events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """ノード全体の活動 (+ Edge なら Internal Client の活動)。"""
    if role == "global":
        return describe_global(proc_events.get("global"))
    if role == "edge":
        result = describe_edge(proc_events.get("edge"), proc_events.get("sub_server"))
        result["internal"] = describe_client(proc_events.get("internal_client"))
        return result
    return describe_client(proc_events.get("leaf"))


class RoundTimeline:
    """直近のラウンドについて、ノード・プロセスごとの開始〜終了時刻を集める。"""

    def __init__(self, keep_rounds: int = 3) -> None:
        self.keep_rounds = keep_rounds
        # global_round → {(node_num, proc): {"t0": ..., "t1": ...}}
        self.rounds: dict[int, dict[tuple[int, str], dict[str, float]]] = {}

    def consume(self, num: int, event: dict[str, Any]) -> None:
        rnd, proc, kind, t = (
            event.get("global_round"), event.get("proc"), event.get("event"), event.get("t")
        )
        if not isinstance(rnd, int) or rnd <= 0 or not isinstance(t, (int, float)):
            return
        if event.get("resumed"):
            return
        for span_proc, start, end, _ in TIMELINE_SPANS:
            if proc != span_proc or kind not in (start, end):
                continue
            bars = self.rounds.setdefault(rnd, {})
            bar = bars.setdefault((num, proc), {})
            bar["t0" if kind == start else "t1"] = float(t)
        while len(self.rounds) > self.keep_rounds:
            del self.rounds[min(self.rounds)]

    def latest(self, nodes: dict[int, dict[str, Any]],
               offsets: dict[int, float] | None = None) -> dict[str, Any] | None:
        """Global の評価まで終わった最新のラウンド。時刻は各ノードの時計ずれを補正する。"""
        offsets = offsets or {}
        complete = [
            r for r, bars in self.rounds.items()
            if any(proc == "global" and "t1" in bar for (_, proc), bar in bars.items())
        ]
        if not complete:
            return None
        rnd = max(complete)
        labels = {proc: label for proc, _, _, label in TIMELINE_SPANS}
        bars = []
        for (num, proc), bar in sorted(self.rounds[rnd].items()):
            if "t0" not in bar:
                continue
            shift = offsets.get(num, 0.0)
            info = nodes.get(num, {})
            bars.append({
                "num": num, "id": info.get("id", str(num)), "proc": proc,
                "label": labels[proc],
                "t0": round(bar["t0"] - shift, 3),
                "t1": round(bar["t1"] - shift, 3) if "t1" in bar else None,
            })
        if not bars:
            return None
        return {
            "round": rnd,
            "t0": min(b["t0"] for b in bars),
            "t1": max((b["t1"] or b["t0"]) for b in bars),
            "bars": bars,
        }
