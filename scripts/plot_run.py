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

監視役 (scripts/hfl_monitor.py) のフォルダ logs/<run_id>/monitor を渡すと、ミラーした events と
metrics.jsonl (電力・温度) も読み、次も出力する:
    metrics.csv                  電力・温度・メモリをフラット化
    power.png                    各ノードの消費電力 (VDD_IN) vs 時刻
    temperature.png              各ノードの温度 (tj) vs 時刻
    summary.json の "power"       ノードごとの平均・最大電力 (W)、消費電力量 (Wh)、最高温度
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


def _global_evals(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """学習で得た Global 評価 (再開直後のチェックポイント評価は除く)。"""
    return sorted(
        (
            e for e in events
            if e.get("proc") == "global" and e.get("event") == "eval"
            and e.get("global_round") and not e.get("resumed")
        ),
        key=_tkey,
    )


def outages(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Global の resume ごとに、学習が止まっていた区間を返す。

    区間 = 停止前の最後の評価 〜 再開後の最初の評価 (学習が実際に進まなかった時間)。
    """
    evals = _global_evals(events)
    resumes = sorted(
        (e for e in events if e.get("proc") == "global" and e.get("event") == "resume"),
        key=_tkey,
    )
    spans = []
    for resume in resumes:
        tr = _tkey(resume)
        before = [_tkey(e) for e in evals if _tkey(e) < tr]
        after = [_tkey(e) for e in evals if _tkey(e) > tr]
        start = before[-1] if before else tr
        end = after[0] if after else tr
        spans.append({
            "from_t": start,
            "to_t": end,
            "from": _iso(start),
            "to": _iso(end),
            "minutes": round((end - start) / 60, 1),
            "resumed_from_round": resume.get("global_round"),
        })
    return spans


# これ以上イベントが無ければ、その間ノードが動いていた証拠は無いとみなして線を切る
SILENT_GAP_S = 300.0


def _with_breaks(
    points: list[tuple[float, float]], breaks: list[float], max_gap: float | None = None
):
    """break の時刻をまたぐ所に NaN を挟み、線を途切れさせる。

    max_gap を指定すると、break が無いまま max_gap 秒以上イベントが途切れた所でも、
    直前の点の直後で線を切る (電源断や強制終了では終了イベントが残らないため)。
    """
    xs: list[float] = []
    ys: list[float] = []
    pending = sorted(breaks)
    for t, y in points:
        if xs and pending and xs[-1] < pending[0] <= t:
            xs.append(pending[0])
            ys.append(float("nan"))
        elif xs and max_gap is not None and t - xs[-1] > max_gap:
            xs.append(xs[-1] + 1.0)
            ys.append(float("nan"))
        while pending and pending[0] <= t:
            pending.pop(0)
        xs.append(t)
        ys.append(y)
    return xs, ys


def _session_breaks(events: list[dict[str, Any]], key: NodeKey) -> list[float]:
    """そのノードの線を切る時刻: プロセスの終了 (stop / error)。

    step 図は次の点まで水平に伸びるので、終了時刻で切れば「その時刻まで動いていた」になり、
    再起動までの空白が残る。
    """
    return sorted(
        _tkey(e) for e in events if node_key(e) == key and e.get("event") in ("stop", "error")
    )


# metrics.jsonl のサンプル間隔がこれより空いたら、その間は電力量に数えない (監視停止・到達不能)
METRIC_MAX_GAP_S = 120.0


def load_metrics(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("metrics.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record.get("t"), (int, float)):
                records.append(record)
    records.sort(key=lambda r: (r["t"], str(r.get("node_num"))))
    return records


def metric_key(record: dict[str, Any]) -> NodeKey:
    return (record.get("node_num"), record.get("node") or "")


def _vdd_in_w(record: dict[str, Any]) -> float | None:
    milliwatts = (record.get("power_mw") or {}).get("VDD_IN")
    return None if milliwatts is None else milliwatts / 1000.0


def summarize_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """ノードごとの平均・最大電力、消費電力量 (台形則)、最高温度。"""
    by_node: dict[NodeKey, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_node[metric_key(record)].append(record)
    nodes: dict[str, Any] = {}
    total_wh = 0.0
    for key in sorted(by_node, key=lambda k: str(k[0])):
        samples = [(r["t"], _vdd_in_w(r)) for r in by_node[key] if _vdd_in_w(r) is not None]
        energy_wh = 0.0
        for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
            if 0 < t1 - t0 <= METRIC_MAX_GAP_S:
                energy_wh += (p0 + p1) / 2 * (t1 - t0) / 3600.0
        temps = [
            (r.get("temp_c") or {}).get("tj") for r in by_node[key]
            if (r.get("temp_c") or {}).get("tj") is not None
        ]
        powers = [p for _, p in samples]
        nodes[node_label(key)] = {
            "samples": len(by_node[key]),
            "mean_power_w": round(statistics.fmean(powers), 2) if powers else None,
            "max_power_w": round(max(powers), 2) if powers else None,
            "energy_wh": round(energy_wh, 2),
            "max_temp_c": max(temps) if temps else None,
        }
        total_wh += energy_wh
    return {
        "first": _iso(records[0]["t"]) if records else None,
        "last": _iso(records[-1]["t"]) if records else None,
        "total_energy_wh": round(total_wh, 2),
        "nodes": nodes,
    }


def write_metrics_csv(records: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for record in records:
        row = {k: v for k, v in record.items() if not isinstance(v, dict)}
        for group in ("power_mw", "temp_c"):
            for name, value in (record.get(group) or {}).items():
                row[f"{group}.{name}"] = value
        rows.append(row)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_node: dict[NodeKey, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_node[node_key(event)].append(event)
    global_evals = _global_evals(events)
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
        "outages": [
            {k: v for k, v in span.items() if not k.endswith("_t")} for span in outages(events)
        ],
        "total_downtime_h": round(sum(span["minutes"] for span in outages(events)) / 60, 3),
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


def make_plots(
    events: list[dict[str, Any]], out: Path, metrics: list[dict[str, Any]] | None = None
) -> list[Path]:
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

    spans = outages(events)

    def shade_outages(ax) -> None:
        for i, span in enumerate(spans):
            ax.axvspan(
                datetime.fromtimestamp(span["from_t"]), datetime.fromtimestamp(span["to_t"]),
                color=MUTED, alpha=0.15, lw=0, label="stopped" if i == 0 else None,
            )

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
        ts, ys = _with_breaks(pts, _session_breaks(events, n), max_gap=SILENT_GAP_S)
        xs = to_dt(ts)
        ax.step(xs, ys, where="post", color=color_of[n], label=node_label(n))
        ax.plot(xs[-1:], ys[-1:], "o", ms=7, color=color_of[n], mec=SURFACE, mew=1.5)
    shade_outages(ax)
    ax.set_ylabel("Global round")
    ax.set_title("Global round progress per node")
    ax.xaxis.set_major_formatter(date_fmt)
    ax.legend(ncol=4, fontsize=9)
    save(fig, "timeline_rounds.png")

    # 2. Global accuracy / loss vs 時刻, 3. vs ラウンド, 6. ラウンド所要時間
    evals = _global_evals(events)
    resume_times = [s["to_t"] for s in spans]
    if evals:
        for name, by_time in (("global_metrics.png", True), ("global_metrics_vs_round.png", False)):
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            for ax, key in ((a1, "accuracy"), (a2, "loss")):
                pts = [(_tkey(e), e.get(key)) for e in evals]
                ts, ys = _with_breaks(pts, resume_times)
                if by_time:
                    ax.plot(to_dt(ts), ys, color=SERIES[0])
                    shade_outages(ax)
                else:
                    # 再開でやり直したラウンドは重なって描かれる
                    round_at = {_tkey(e): e["global_round"] for e in evals}
                    xs = [round_at.get(t, float("nan")) for t in ts]
                    ax.plot(xs, ys, color=SERIES[0])
                ax.set_ylabel(key)
            a1.set_ylim(0, 1)
            a1.set_title("Global evaluation (CIFAR-10 test)")
            a2.set_xlabel("time" if by_time else "Global round")
            if by_time:
                a2.xaxis.set_major_formatter(date_fmt)
                if spans:
                    a1.legend(loc="lower right", fontsize=9)
            save(fig, name)

        durations = [
            (e, e["round_duration_s"]) for e in evals
            if isinstance(e.get("round_duration_s"), (int, float))
        ]
        if durations:
            fig, ax = plt.subplots(figsize=(11, 4))
            ts, ys = _with_breaks([(_tkey(e), d) for e, d in durations], resume_times)
            ax.plot(to_dt(ts), ys, color=SERIES[0])
            shade_outages(ax)
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

    # 7, 8. 電力・温度 (監視役の metrics.jsonl があるとき)
    if metrics:
        metric_nodes = sorted({metric_key(r) for r in metrics}, key=lambda k: str(k[0]))
        for name, title, ylabel, value in (
            ("power.png", "Power per node (VDD_IN)", "W", _vdd_in_w),
            ("temperature.png", "Temperature per node (tj)", "deg C",
             lambda r: (r.get("temp_c") or {}).get("tj")),
        ):
            fig, ax = plt.subplots(figsize=(11, 5))
            for i, key in enumerate(metric_nodes):
                pts = [
                    (r["t"], value(r)) for r in metrics
                    if metric_key(r) == key and value(r) is not None
                ]
                if not pts:
                    continue
                ts, ys = _with_breaks(pts, [], max_gap=SILENT_GAP_S)
                color = color_of.get(key, SERIES[i % len(SERIES)])
                ax.plot(to_dt(ts), ys, color=color, lw=1.2, label=node_label(key))
            shade_outages(ax)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.xaxis.set_major_formatter(date_fmt)
            ax.legend(ncol=4, fontsize=9)
            save(fig, name)
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
    metrics = load_metrics(root)
    if metrics:
        write_metrics_csv(metrics, out / "metrics.csv")
        summary["power"] = summarize_metrics(metrics)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written = [] if args.no_plots else make_plots(events, out, metrics)

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
    if metrics:
        power = summary["power"]
        print(f"power: {len(metrics)} samples, total energy {power['total_energy_wh']} Wh")
        for label, info in power["nodes"].items():
            print(
                f"  {label:<14} mean={info['mean_power_w']} W max={info['max_power_w']} W "
                f"energy={info['energy_wh']} Wh max_temp={info['max_temp_c']} C"
            )
    for path in written:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
