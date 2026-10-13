"""可視化スクリプト (scripts/plot_run.py) の停止区間・線の区切りのテスト。"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("plot_run", REPO / "scripts" / "plot_run.py")
plot_run = importlib.util.module_from_spec(spec)
sys.modules["plot_run"] = plot_run
spec.loader.exec_module(plot_run)


def _ev(t, event, proc="global", **fields):
    return {"t": t, "event": event, "proc": proc, "node": "global", "node_num": 201, **fields}


def test_outage_spans_from_last_eval_to_first_eval_after_resume():
    events = [
        _ev(100, "eval", global_round=4),
        _ev(118, "eval", global_round=5),
        _ev(300, "error"),
        _ev(900, "resume", global_round=5),
        _ev(905, "eval", global_round=5, resumed=True),
        _ev(940, "eval", global_round=6),
    ]
    spans = plot_run.outages(events)
    assert len(spans) == 1
    assert spans[0]["from_t"] == 118 and spans[0]["to_t"] == 940
    assert spans[0]["resumed_from_round"] == 5
    summary = plot_run.summarize(events)
    assert summary["global_rounds_completed"] == 6
    assert summary["total_downtime_h"] == round((940 - 118) / 60 / 60, 3)


def test_line_breaks_at_exit_and_after_silence():
    points = [(0, 1), (10, 2), (20, 3), (1000, 4), (1010, 5)]
    # 終了イベントがある場合はその時刻で切る
    xs, ys = plot_run._with_breaks(points, [500])
    assert xs[3] == 500 and math.isnan(ys[3])
    # 終了イベントが無くても、max_gap 以上の空白の直前で切る
    xs, ys = plot_run._with_breaks(points, [], max_gap=300)
    assert xs[3] == 21 and math.isnan(ys[3])
    # 通常の間隔では切らない
    xs, ys = plot_run._with_breaks(points[:3], [], max_gap=300)
    assert not any(math.isnan(y) for y in ys)
