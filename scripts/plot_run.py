#!/usr/bin/env python
"""
回収したイベント (events-<num>.jsonl) を1本の CSV にまとめ、可視化 PNG を出力する。

Usage (PC で実行。matplotlib が必要: uv sync --extra analysis):
    uv run python scripts/plot_run.py collected/<run_id> [--out collected/<run_id>/plots]

入力ディレクトリ直下または配下の events-*.jsonl をすべて読む。
clock_offsets.json (scripts/hfl.sh collect が書く) があれば t_corrected = t - offset を付ける。

出力:
    events.csv                   全イベントをフラット化 (可視化ツールに読ませる用)
    summary.json                 ラウンド数・稼働時間・精度・停止理由などの要約
    timeline_rounds.png          各ノードの Global ラウンド番号 vs 時刻 (止まった端末は線が途切れる)
    global_metrics.png           Global accuracy / loss vs 時刻
    global_metrics_vs_round.png  同 vs ラウンド
    participation.png            参加マトリクス (ノード × ラウンド)
    fit_duration.png             各ノードの fit 所要時間 vs 時刻
    round_duration.png           Global ラウンド所要時間 vs 時刻
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# 検証済みパレット (dataviz: categorical slots, 固定順で割り当て)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SEQ_LIGHT, SEQ_DARK = "#cde2fb", "#0d366b"

LEADING_COLUMNS = [
    "ts", "t", "t_corrected", "run_id", "node", "node_num", "host", "proc", "pid", "event",
    "global_round", "parent_round", "child_round", "duration_s", "wait_s", "round_duration_s",
    "loss", "accuracy", "n", "examples", "reported_examples", "total_examples",
    "results", "failures", "participants", "client_id", "reason", "exc_type", "message",
]

NodeKey = tuple[Any, str]


def load_events(root: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(root.rglob("events-*.jsonl")):
        num_text = path.stem.split("-", 1)[1]
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("node_num") is None:
                # run.py 経由 (1PC) ではファイル名が events-<node_id>.jsonl で番号が無い
                record["node_num"] = int(num_text) if num_text.isdigit() else None
                record.setdefault("node", None if num_text.isdigit() else num_text)
            events.append(record)
    events.sort(key=lambda e: (e.get("t", 0), str(e.get("node_num"))))
    return events


def apply_clock_offsets(events: list[dict[str, Any]], root: Path) -> None:
    path = root / "clock_offsets.json"
    offsets: dict[str, float] = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            offsets = {k: v for k, v in raw.items() if v is not None}
        except (OSError, json.JSONDecodeError):
            offsets = {}
    for event in events:
        offset = offsets.get(str(event.get("node_num")), 0.0)
        if isinstance(event.get("t"), (int, float)):
            event["t_corrected"] = round(float(event["t"]) - float(offset), 3)


def write_csv(events: list[dict[str, Any]], path: Path) -> None:
    keys = list(LEADING_COLUMNS)
    extra = sorted({k for e in events for k in e} - set(keys))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys + extra, extrasaction="ignore")
        writer.writeheader()
        for event in events:
            row = dict(event)
            if isinstance(row.get("participants"), list):
                row["participants"] = "|".join(map(str, row["participants"]))
            writer.writerow(row)


def node_key(event: dict[str, Any]) -> NodeKey:
    return (event.get("node_num"), event.get("node") or "")


def node_label(key: NodeKey) -> str:
    return " ".join(str(part) for part in key if part not in (None, "")) or "?"


def _tkey(event: dict[str, Any]) -> float:
    return float(event.get("t_corrected", event.get("t", 0)))


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "max": None}
    return {
        "mean": round(statistics.fmean(values), 2),
        "median": round(statistics.median(values), 2),
        "max": round(max(values), 2),
    }


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t).isoformat()


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_node: dict[NodeKey, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_node[node_key(event)].append(event)
    global_evals = [
        e for e in events
        if e.get("proc") == "global" and e.get("event") == "eval" and e.get("global_round")
    ]
    round_durations = [
        e["round_duration_s"] for e in global_evals
        if isinstance(e.get("round_duration_s"), (int, float))
    ]
    starts = [e["t"] for e in events if e.get("event") == "start"]
    stops = [e["t"] for e in events if e.get("event") in ("stop", "error")]
    summary: dict[str, Any] = {
        "run_id": events[0].get("run_id") if events else None,
        "events": len(events),
        "first_event": _iso(events[0]["t"]) if events else None,
        "last_event": _iso(events[-1]["t"]) if events else None,
        "wall_clock_h": round((events[-1]["t"] - events[0]["t"]) / 3600, 3) if events else None,
        "global_rounds_completed": max((e["global_round"] for e in global_evals), default=0),
        "final_accuracy": global_evals[-1].get("accuracy") if global_evals else None,
        "best_accuracy": max((e.get("accuracy") or 0 for e in global_evals), default=None),
        "round_duration_s": _stats(round_durations),
        "first_start": _iso(min(starts)) if starts else None,
        "last_stop": _iso(max(stops)) if stops else None,
        "nodes": {},
    }
    if len(global_evals) >= 3:
        gaps = [b["t"] - a["t"] for a, b in zip(global_evals, global_evals[1:])]
        median_gap = statistics.median(gaps)
        summary["gaps_over_2x_median"] = [
            {"after_round": a["global_round"], "gap_s": round(g, 1), "at": _iso(a["t"])}
            for a, g in zip(global_evals, gaps)
            if g > 2 * median_gap and g > 30
        ][:50]
    for key in sorted(by_node, key=lambda k: str(k[0])):
        items = by_node[key]
        fits = [e for e in items if e.get("event") in ("fit_end", "parent_fit_end")]
        durations = [e["duration_s"] for e in fits if isinstance(e.get("duration_s"), (int, float))]
        exits = [e for e in items if e.get("event") in ("stop", "error")]
        summary["nodes"][node_label(key)] = {
            "events": len(items),
            "procs": sorted({e.get("proc") for e in items if e.get("proc")}),
            "fits": len(fits),
            "fit_duration_s": _stats(durations),
            "last_round": max((e.get("global_round") or 0 for e in items), default=0),
            "exit": [
                {
                    "proc": e.get("proc"),
                    "event": e.get("event"),
                    "reason": e.get("reason") or e.get("exc_type"),
                    "at": _iso(e["t"]),
                }
                for e in exits
            ],
        }
    return summary


def _participation_matrix(events: list[dict[str, Any]]):
    """aggregate.participants から (行ラベル, 行列[参加率], 最大ラウンド, ビン数) を作る。"""
    import numpy as np

    participation: dict[str, set[int]] = defaultdict(set)
    for e in events:
        if e.get("event") != "aggregate" or not e.get("global_round"):
            continue
        for p in e.get("participants") or []:
            participation[str(p)].add(int(e["global_round"]))
    max_round = max((max(r) for r in participation.values() if r), default=0)
    if not max_round:
        return None
    rows = sorted(participation)
    bins = max_round if max_round <= 2000 else 500
    width = max_round / bins
    matrix = np.zeros((len(rows), bins))
    counts = np.zeros(bins)
    for b in range(bins):
        lo, hi = int(b * width) + 1, int((b + 1) * width)
        counts[b] = max(hi - lo + 1, 1)
    for i, row in enumerate(rows):
        for r in participation[row]:
            matrix[i, min(int((r - 1) / width), bins - 1)] += 1
    return rows, matrix / counts, max_round, bins


def make_plots(events: list[dict[str, Any]], out: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("matplotlib がありません: uv sync --extra analysis を実行", file=sys.stderr)
        return []

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10, "text.color": INK,
        "axes.labelcolor": INK2, "axes.edgecolor": GRID, "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.grid": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 2,
    })
    written: list[Path] = []
    nodes = sorted({node_key(e) for e in events}, key=lambda k: str(k[0]))
    color_of = {n: SERIES[i % len(SERIES)] for i, n in enumerate(nodes)}
    date_fmt = mdates.DateFormatter("%m/%d %H:%M")

    def to_dt(values):
        return [datetime.fromtimestamp(v) for v in values]

    def save(fig, name: str) -> None:
        path = out / name
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)

    def node_points(n: NodeKey, event_name: str | None, value_key: str):
        return sorted(
            (_tkey(e), e[value_key]) for e in events
            if node_key(e) == n
            and (event_name is None or e.get("event") == event_name)
            and isinstance(e.get(value_key), (int, float)) and e[value_key] > 0
        )

    # 1. ラウンド番号 vs 時刻 (ノードごと)
    fig, ax = plt.subplots(figsize=(11, 5))
    for n in nodes:
        pts = node_points(n, None, "global_round")
        if not pts:
            continue
        xs, ys = to_dt([p[0] for p in pts]), [p[1] for p in pts]
        ax.step(xs, ys, where="post", color=color_of[n], label=node_label(n))
        ax.plot(xs[-1:], ys[-1:], "o", ms=7, color=color_of[n], mec=SURFACE, mew=1.5)
    ax.set_ylabel("Global round")
    ax.set_title("Global round progress per node")
    ax.xaxis.set_major_formatter(date_fmt)
    ax.legend(ncol=4, fontsize=9)
    save(fig, "timeline_rounds.png")

    # 2. Global accuracy / loss vs 時刻, 3. vs ラウンド, 6. ラウンド所要時間
    evals = sorted(
        (
            e for e in events
            if e.get("proc") == "global" and e.get("event") == "eval" and e.get("global_round")
        ),
        key=_tkey,
    )
    if evals:
        for name, xs, xlabel in (
            ("global_metrics.png", to_dt([_tkey(e) for e in evals]), "time"),
            ("global_metrics_vs_round.png", [e["global_round"] for e in evals], "Global round"),
        ):
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            a1.plot(xs, [e.get("accuracy") for e in evals], color=SERIES[0])
            a1.set_ylabel("accuracy")
            a1.set_ylim(0, 1)
            a1.set_title("Global evaluation (CIFAR-10 test)")
            a2.plot(xs, [e.get("loss") for e in evals], color=SERIES[0])
            a2.set_ylabel("loss")
            a2.set_xlabel(xlabel)
            if xlabel == "time":
                a2.xaxis.set_major_formatter(date_fmt)
            save(fig, name)

        durations = [
            (e, e["round_duration_s"]) for e in evals
            if isinstance(e.get("round_duration_s"), (int, float))
        ]
        if durations:
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(to_dt([_tkey(e) for e, _ in durations]), [d for _, d in durations],
                    color=SERIES[0])
            ax.set_ylabel("seconds")
            ax.set_title("Global round duration")
            ax.xaxis.set_major_formatter(date_fmt)
            save(fig, "round_duration.png")

    # 4. 参加マトリクス
    matrix_info = _participation_matrix(events)
    if matrix_info is not None:
        rows, matrix, max_round, bins = matrix_info
        fig, ax = plt.subplots(figsize=(11, 0.45 * len(rows) + 1.8))
        cmap = LinearSegmentedColormap.from_list("blue", [SEQ_LIGHT, SEQ_DARK])
        im = ax.imshow(
            matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
            extent=(0.5, max_round + 0.5, len(rows) - 0.5, -0.5),
        )
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows)
        ax.grid(False)
        ax.set_xlabel("Global round")
        ax.set_title("Participation" + (" (fraction per bin)" if bins != max_round else ""))
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="participation")
        save(fig, "participation.png")

    # 5. fit 所要時間
    fig, ax = plt.subplots(figsize=(11, 5))
    plotted = False
    for n in nodes:
        pts = node_points(n, "fit_end", "duration_s")
        if not pts:
            continue
        plotted = True
        ax.plot(to_dt([p[0] for p in pts]), [p[1] for p in pts], ".", ms=5, alpha=0.8,
                color=color_of[n], label=node_label(n))
    if plotted:
        ax.set_ylabel("seconds")
        ax.set_title("fit duration per client (training + local evaluation)")
        ax.xaxis.set_major_formatter(date_fmt)
        ax.legend(ncol=4, fontsize=9, markerscale=2)
        save(fig, "fit_duration.png")
    else:
        plt.close(fig)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HFL run plots")
    parser.add_argument("root", help="collected/<run_id> (events-*.jsonl を含むディレクトリ)")
    parser.add_argument("--out", default=None, help="出力先 (既定: <root>/plots)")
    parser.add_argument("--no-plots", action="store_true", help="CSV と summary だけ出す")
    args = parser.parse_args(argv)

    root = Path(args.root)
    events = load_events(root)
    if not events:
        print(f"events-*.jsonl が見つかりません: {root}", file=sys.stderr)
        return 1
    apply_clock_offsets(events, root)
    out = Path(args.out) if args.out else root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(events, out / "events.csv")
    summary = summarize(events)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written = [] if args.no_plots else make_plots(events, out)

    print(f"events: {len(events)}  ->  {out / 'events.csv'}")
    print(
        f"global rounds: {summary['global_rounds_completed']}  "
        f"wall clock: {summary['wall_clock_h']} h  "
        f"final acc: {summary['final_accuracy']}  best acc: {summary['best_accuracy']}  "
        f"round median: {summary['round_duration_s']['median']} s"
    )
    for label, info in summary["nodes"].items():
        exits = ", ".join(f"{x['proc']}:{x['event']}:{x['reason']}" for x in info["exit"]) or "-"
        print(
            f"  {label:<14} fits={info['fits']:<6} last_round={info['last_round']:<7} exit={exits}"
        )
    for path in written:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
