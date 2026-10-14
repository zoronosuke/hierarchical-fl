"""活動の判定・タイムライン (scripts/hfl_activity.py)、通信量の集計、再生範囲の選び方のテスト。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hfl_activity = _load("hfl_activity")
plot_run = _load("plot_run")
build_replay = _load("build_replay")
hfl_monitor = _load("hfl_monitor")


def _ev(t, proc, event, **fields):
    return {"t": t, "proc": proc, "event": event, **fields}


def test_describe_global_edge_and_clients():
    d = hfl_activity.describe
    started = d("global", {"global": _ev(1, "global", "round_start", global_round=5)})
    assert started["phase"] == "train"
    waiting = d("global", {"global": _ev(1, "global", "wait", global_round=6, next_round_t=0)})
    assert waiting["phase"] == "idle" and "待機" in waiting["label"]
    assert d("global", {"global": _ev(1, "global", "checkpoint", global_round=5)})["phase"] == \
        "aggregate"

    edge = d("edge", {
        "edge": _ev(1, "edge", "parent_fit_start", global_round=5),
        "sub_server": _ev(2, "sub_server", "child_result_sent", global_round=5),
        "internal_client": _ev(2, "internal_client", "fit_end", global_round=5),
    })
    assert edge["phase"] == "aggregate"  # 新しい方 (sub_server) で判定
    assert edge["internal"]["phase"] == "idle"
    assert d("edge", {"edge": _ev(3, "edge", "parent_fit_end")})["phase"] == "idle"

    assert d("leaf", {"leaf": _ev(1, "leaf", "fit_start", global_round=5)})["phase"] == "train"
    assert d("leaf", {"leaf": _ev(1, "leaf", "error")})["phase"] == "error"
    assert d("leaf", {})["phase"] == "stopped"


def test_round_timeline_keeps_latest_complete_round_and_corrects_clocks():
    timeline = hfl_activity.RoundTimeline(keep_rounds=2)
    for rnd, base in ((1, 100.0), (2, 200.0), (3, 300.0)):
        timeline.consume(201, _ev(base, "global", "round_start", global_round=rnd))
        timeline.consume(205, _ev(base + 2, "leaf", "fit_start", global_round=rnd))
        timeline.consume(205, _ev(base + 8, "leaf", "fit_end", global_round=rnd))
        if rnd < 3:  # ラウンド 3 はまだ評価されていない
            timeline.consume(201, _ev(base + 18, "global", "eval", global_round=rnd))
    assert sorted(timeline.rounds) == [2, 3]
    latest = timeline.latest({201: {"id": "global"}, 205: {"id": "leaf_02"}}, {205: -1.0})
    assert latest["round"] == 2
    leaf = next(b for b in latest["bars"] if b["proc"] == "leaf")
    assert (leaf["t0"], leaf["t1"]) == (203.0, 209.0)  # 時計ずれ -1 秒を補正


def test_monitor_uses_sampler_records_for_power_and_network():
    state = hfl_monitor.RunState()
    records = [
        {"t": 1000.0 + 5 * i, "power_mw": {"VDD_IN": 5000.0}, "temp_c": {"tj": 50.0},
         "net": {"tx_bytes": 1_000_000 * i, "rx_bytes": 2_000_000 * i,
                 "links": {"parent": {"sent": 900_000 * i, "recv": 0}}}}
        for i in range(30)
    ]
    state.consume_sampler_records(202, records)
    assert state.node(202).has_sampler
    tx_rate, rx_rate = state.net_rates(202)
    assert round(tx_rate) == 200_000 and round(rx_rate) == 400_000  # 1 MB / 5 s
    assert len(state.metric_series["202"]) == 3  # 1分に1点 (145 秒分)
    assert state.metric_series["202"][-1][3] == 24_000_000


def test_network_summary_separates_learning_and_other_traffic():
    records = [
        {"t": 1.0, "node_num": 202, "node": "edge_01",
         "net": {"tx_bytes": 10_000, "rx_bytes": 9_000, "links": {}}},
        {"t": 2.0, "node_num": 202, "node": "edge_01",
         "net": {"tx_bytes": 3_000_000, "rx_bytes": 3_100_000, "iface": "enP8p1s0",
                 "links": {"parent": {"sent": 750_000, "recv": 760_000},
                           "child:192.168.10.204": {"sent": 760_000, "recv": 750_000},
                           "child:192.168.10.205": {"sent": 760_000, "recv": 750_000}}}},
    ]
    network = plot_run.summarize_network(records, rounds=2)
    node = network["nodes"]["202 edge_01"]
    assert node["fl_sent_bytes"] == 2_270_000
    assert node["other_tx_bytes"] == 730_000
    assert network["total_tx_bytes_per_round"] == 1_500_000


def test_silent_gap_follows_round_interval():
    events = [{"proc": "global", "event": "start", "round_interval_s": 600}]
    assert plot_run.silent_gap_s(events) == 1200
    assert plot_run.silent_gap_s([]) == plot_run.SILENT_GAP_S


def test_replay_range_stops_once_accuracy_flattens():
    accs = [0.3, 0.5, 0.6, 0.65, 0.7, 0.72, 0.74, 0.75, 0.755, 0.76] + [0.76] * 60
    evals = [{"accuracy": a, "global_round": i + 1} for i, a in enumerate(accs)]
    start, end = build_replay.choose_range(evals)
    assert start == 0
    assert build_replay.MIN_FRAMES - 1 <= end < 30  # 横ばいになったら止める (最低 20 コマ)


def test_sub_server_eval_after_return_does_not_look_like_aggregation():
    state = hfl_monitor.RunState()
    state.consume_events(203, "edge", [
        _ev(1, "sub_server", "child_result_sent", global_round=7),
        _ev(2, "edge", "parent_fit_end", global_round=7),
        _ev(3, "sub_server", "eval", global_round=7),  # 返送後の確認用の評価
    ])
    assert hfl_activity.describe("edge", state.node(203).proc_events)["phase"] == "idle"
