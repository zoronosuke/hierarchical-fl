#!/usr/bin/env python
"""
再生ページ (動画用) のデータを作る。
学習が大きく変わる区間を 10〜20 秒で再生し、画面録画で動画にする。

Usage (PC で):
    uv run python scripts/build_replay.py logs/<run_id>/monitor        # 監視役のフォルダ
    uv run python scripts/build_replay.py collected/<run_id>           # collect したフォルダ
        [--profile config/jetson-5tier-paced] [--from-round N] [--to-round M] [--seconds 15]

出力 (既定: <入力>/replay/): replay.html, viz.js, replay-data.js。replay.html をブラウザで開く
(ファイルを直接開いてよい)。URL に ?autoplay=1 (自動再生)、?clean=1 (操作パネルを隠す)、
?seconds=20 (再生時間) を付けられる。

範囲を指定しないときは、ラウンド 1 から「精度 (5ラウンド移動平均) が最高値との差 0.01 以内に
初めて届くラウンド」までを再生する (最低 20 ラウンド)。1 コマ = 1 ラウンドで、各ラウンドの中は
配布 → 学習 → 集約の流れを 4 段階で見せる。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hfl_activity import describe, is_activity_event  # noqa: E402
from hfl_nodes import load_inventory  # noqa: E402
from plot_run import (  # noqa: E402
    _global_evals,
    _tkey,
    apply_clock_offsets,
    load_events,
    load_metrics,
    outages,
    round_interval_s,
)

MIN_FRAMES = 20
MAX_FRAMES = 1000
STEP_FRACTIONS = (0.1, 0.4, 0.7, 1.0)  # ラウンドの中の見せる時点 (開始〜評価の割合)
MAX_SERIES_POINTS = 600
ROLLING = 5
PLATEAU_MARGIN = 0.01


def choose_range(evals: list[dict[str, Any]]) -> tuple[int, int]:
    """学習が大きく変わる区間: 先頭から、移動平均が最高値 − 0.01 に届くまで (添字)。"""
    if not evals:
        return 0, -1
    accs = [e.get("accuracy") or 0.0 for e in evals]
    rolling = [
        sum(accs[max(0, i - ROLLING + 1):i + 1]) / len(accs[max(0, i - ROLLING + 1):i + 1])
        for i in range(len(accs))
    ]
    target = max(rolling) - PLATEAU_MARGIN
    end = next(i for i, value in enumerate(rolling) if value >= target)
    end = min(max(end, MIN_FRAMES - 1), len(evals) - 1)
    return 0, end


def _downsample(points: list[list[Any]], limit: int = MAX_SERIES_POINTS) -> list[list[Any]]:
    if len(points) <= limit:
        return points
    step = len(points) / limit
    return [points[int(i * step)] for i in range(limit)] + [points[-1]]


def build(root: Path, profile: str, from_round: int | None, to_round: int | None,
          seconds: float) -> dict[str, Any]:
    events = load_events(root)
    apply_clock_offsets(events, root)
    metrics = load_metrics(root)
    inventory = load_inventory(profile)
    roles = {n.num: n.role for n in inventory.nodes}
    evals = _global_evals(events)
    if not evals:
        raise SystemExit(f"Global の評価イベントがありません: {root}")

    start_i, end_i = choose_range(evals)
    if from_round is not None:
        start_i = next((i for i, e in enumerate(evals) if e["global_round"] >= from_round), 0)
    if to_round is not None:
        end_i = max(i for i, e in enumerate(evals) if e["global_round"] <= to_round)
    selected = evals[start_i:end_i + 1]
    stride = max(1, -(-len(selected) // MAX_FRAMES))
    selected = selected[::stride]
    t_from, t_to = _tkey(selected[0]), _tkey(selected[-1])

    # ラウンドの開始時刻 (Global の round_start)
    round_start = {}
    for e in events:
        if e.get("proc") == "global" and e.get("event") == "round_start":
            round_start[e["global_round"]] = _tkey(e)

    # 時刻順にイベント・記録をたどり、各時点のノードの状態を作る
    timeline = sorted(events, key=_tkey)
    metric_timeline = sorted(metrics, key=lambda r: r["t"])
    proc_events: dict[int, dict[str, dict[str, Any]]] = {n.num: {} for n in inventory.nodes}
    last_round: dict[int, int] = {}
    last_metric: dict[int, dict[str, Any]] = {}
    ei = mi = 0

    def advance(t: float) -> None:
        nonlocal ei, mi
        while ei < len(timeline) and _tkey(timeline[ei]) <= t:
            e = timeline[ei]
            num = e.get("node_num")
            if num in proc_events and e.get("proc") and is_activity_event(e):
                proc_events[num][e["proc"]] = e
                rnd = e.get("global_round")
                if isinstance(rnd, int) and rnd > 0 and not e.get("resumed"):
                    last_round[num] = rnd
            ei += 1
        while mi < len(metric_timeline) and metric_timeline[mi]["t"] <= t:
            r = metric_timeline[mi]
            last_metric[r.get("node_num")] = r
            mi += 1

    frames = []
    for ev in selected:
        rnd = ev["global_round"]
        t_end = _tkey(ev)
        t_begin = round_start.get(rnd, t_end - 1.0)
        steps = []
        for fraction in STEP_FRACTIONS:
            advance(t_begin + (t_end - t_begin) * fraction)
            steps.append({
                str(num): describe(roles[num], proc_events[num]) for num in proc_events
            })
        nodes = {}
        for num in proc_events:
            m = last_metric.get(num) or {}
            net = m.get("net") or {}
            power = (m.get("power_mw") or {}).get("VDD_IN")
            nodes[str(num)] = {
                "round": last_round.get(num),
                "power_w": round(power / 1000, 2) if power is not None else None,
                "temp_c": (m.get("temp_c") or {}).get("tj"),
                "tx_bytes": net.get("tx_bytes"),
                "links": net.get("links"),
            }
        frames.append({
            "round": rnd, "t": round(t_end, 3), "accuracy": ev.get("accuracy"),
            "loss": ev.get("loss"), "steps": steps, "nodes": nodes,
        })

    # 図の時系列 (再生範囲のみ。ページ側で現在時刻までを描く)
    def in_range(t: float) -> bool:
        return t_from - 60 <= t <= t_to + 1

    series_rounds: dict[str, list[list[Any]]] = {}
    for e in events:
        num, rnd = e.get("node_num"), e.get("global_round")
        if num in roles and isinstance(rnd, int) and rnd > 0 and in_range(_tkey(e)):
            points = series_rounds.setdefault(str(num), [])
            if not points or points[-1][1] != rnd:
                points.append([round(_tkey(e), 1), rnd])
    series_metrics: dict[str, list[list[Any]]] = {}
    for r in metrics:
        if r.get("node_num") in roles and in_range(r["t"]):
            power = (r.get("power_mw") or {}).get("VDD_IN")
            series_metrics.setdefault(str(r["node_num"]), []).append([
                round(r["t"], 1),
                round(power / 1000, 2) if power is not None else None,
                (r.get("temp_c") or {}).get("tj"),
                (r.get("net") or {}).get("tx_bytes"),
            ])
    return {
        "run_id": evals[0].get("run_id"),
        "seconds": seconds,
        "round_interval_s": round_interval_s(events),
        "range": {
            "from_round": selected[0]["global_round"], "to_round": selected[-1]["global_round"],
            "t_from": t_from, "t_to": t_to, "stride": stride,
        },
        "nodes": [
            {"num": n.num, "id": n.id, "role": n.role, "parent": n.parent_num}
            for n in inventory.nodes
        ],
        "frames": frames,
        "global_series": [[round(_tkey(e), 1), e.get("accuracy"), e.get("loss"), e["global_round"]]
                          for e in evals if in_range(_tkey(e))],
        "round_series": {k: _downsample(v) for k, v in series_rounds.items()},
        "metric_series": {k: _downsample(v) for k, v in series_metrics.items()},
        "outages": [[s["from_t"], s["to_t"]] for s in outages(events)
                    if s["to_t"] >= t_from and s["from_t"] <= t_to],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the replay page (for a short video)")
    parser.add_argument("root", help="logs/<run_id>/monitor または collected/<run_id>")
    parser.add_argument("--profile", default="config/jetson-5tier-paced",
                        help="トポロジー (nodes.yaml) を読む profile")
    parser.add_argument("--from-round", type=int, default=None)
    parser.add_argument("--to-round", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=15.0, help="再生時間 (秒)")
    parser.add_argument("--out", default=None, help="出力先 (既定: <root>/replay)")
    args = parser.parse_args(argv)

    root = Path(args.root)
    data = build(root, args.profile, args.from_round, args.to_round, args.seconds)
    out = Path(args.out) if args.out else root / "replay"
    out.mkdir(parents=True, exist_ok=True)
    page_dir = Path(__file__).resolve().parent / "monitor"
    shutil.copyfile(page_dir / "replay.html", out / "replay.html")
    shutil.copyfile(page_dir / "viz.js", out / "viz.js")
    (out / "replay-data.js").write_text(
        "window.REPLAY = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8"
    )
    rng = data["range"]
    print(
        f"replay: rounds {rng['from_round']}-{rng['to_round']} ({len(data['frames'])} frames, "
        f"stride {rng['stride']}), {args.seconds:.0f} s -> {out / 'replay.html'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
