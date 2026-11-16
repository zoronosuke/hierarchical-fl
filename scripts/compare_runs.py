#!/usr/bin/env python
"""
2 つの実験 (例: 5 段 HFL と flat FL) の通信量と精度を比べる。

Usage (PC で。matplotlib が必要: uv sync --extra analysis):
    uv run python scripts/compare_runs.py collected/<hfl-run> collected/<flat-run> \
        [--labels HFL FlatFL] [--out compare/<a>_vs_<b>] [--profile-a DIR --profile-b DIR]
        [--acc-threshold 0.7] [--no-plots]

入力は scripts/hfl.sh collect のフォルダ (events-<NUM>.jsonl, metrics-<NUM>.jsonl, nodes.json) か
監視役のフォルダ logs/<run-id>/monitor (この場合は --profile-a/--profile-b で台帳を渡す)。

比べ方:
  - リンク = 台帳の親子関係 (子→親)。量は子ノードの Jetson 側記録 net.links.parent
    (sent = 上り: 子→親、recv = 下り: 親→子) の最終累計を Global のラウンド数で割る。
    学習の接続の中身だけを数えるので、ラウンド間隔 (10 分待つ / 連続) に左右されない。
  - 期待値 = 1 ラウンド片方向にそのリンクを通るモデル数。HFL は集約するので全リンク 1、
    flat FL は子の部分木にいるクライアント数 (fit を記録したクライアント ID の数)。
  - 精度は Global の eval をラウンド軸で重ねる (時間軸は補助。間隔が違えば注記する)。

出力 (<out>/):
    comparison.json              リンク・ノード・全体・精度の比較 (B/A の比も)
    links.csv, nodes.csv         表計算用
    link_bytes_per_round.png     リンクごとの上り・下り MB/ラウンド (A と B の棒)
    node_tx_per_round.png        ノードごとの送信 MB/ラウンド (学習分とその他の積み上げ)
    cumulative_tx_vs_round.png   全ノード合計の学習送信量 (累計) vs ラウンド
    accuracy_vs_round.png        Global accuracy vs ラウンド
    accuracy_vs_time.png         Global accuracy vs 開始からの経過時間
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_run import (  # noqa: E402
    INK2,
    MUTED,
    RCPARAMS,
    SERIES,
    _global_evals,
    _tkey,
    apply_clock_offsets,
    load_events,
    load_metrics,
    round_interval_s,
    summarize,
    summarize_network,
)

MB = 1e6
ROUNDS_PER_DAY_PACED = 144  # 10 分間隔での 1 日のラウンド数 (換算用)


@dataclass
class NodeInfo:
    num: int
    id: str
    role: str
    parent_num: int | None


@dataclass
class Run:
    label: str
    root: Path
    nodes: list[NodeInfo]
    events: list[dict[str, Any]]
    metrics: list[dict[str, Any]]
    summary: dict[str, Any]
    network: dict[str, Any] | None
    evals: list[dict[str, Any]]
    clients_on_node: dict[int, set[str]] = field(default_factory=dict)

    @property
    def rounds(self) -> int:
        return int(self.summary.get("global_rounds_completed") or 0)

    @property
    def mode(self) -> str:
        return "flat" if any(n.role == "relay" for n in self.nodes) else "hfl"

    def by_num(self, num: int) -> NodeInfo | None:
        return next((n for n in self.nodes if n.num == num), None)

    def children(self, num: int) -> list[NodeInfo]:
        return [n for n in self.nodes if n.parent_num == num]

    def subtree_clients(self, num: int) -> set[str]:
        clients = set(self.clients_on_node.get(num, set()))
        for child in self.children(num):
            clients |= self.subtree_clients(child.num)
        return clients


# ------------------------------------------------------------------ 読み込み
def load_nodes(root: Path, profile: str | None) -> list[NodeInfo]:
    """collect の nodes.json、無ければ --profile の台帳から。"""
    if profile:
        from hfl_nodes import load_inventory

        inventory = load_inventory(profile)
        return [NodeInfo(n.num, n.id, n.role, n.parent_num) for n in inventory.nodes]
    path = root / "nodes.json"
    if not path.exists():
        raise SystemExit(f"{path} がありません (--profile-a/--profile-b で台帳を指定)")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        NodeInfo(int(n["num"]), str(n.get("id") or n["role"]), str(n["role"]), n.get("parent_num"))
        for n in raw.get("nodes", [])
    ]


def load_run(label: str, root: Path, profile: str | None) -> Run:
    events = load_events(root)
    if not events:
        raise SystemExit(f"events-*.jsonl が見つかりません: {root}")
    apply_clock_offsets(events, root)
    summary = summarize(events)
    metrics = load_metrics(root)
    network = summarize_network(metrics, summary["global_rounds_completed"]) if metrics else None
    run = Run(
        label=label, root=root, nodes=load_nodes(root, profile), events=events,
        metrics=metrics, summary=summary, network=network, evals=_global_evals(events),
    )
    for e in events:
        if e.get("event") == "fit_start" and e.get("client_id") and e.get("node_num") is not None:
            run.clients_on_node.setdefault(int(e["node_num"]), set()).add(str(e["client_id"]))
    return run


# ------------------------------------------------------------------ 集計
def _last_net(run: Run) -> dict[int, dict[str, Any]]:
    last: dict[int, dict[str, Any]] = {}
    for record in run.metrics:
        net = record.get("net") or {}
        if isinstance(net.get("tx_bytes"), int) and record.get("node_num") is not None:
            last[int(record["node_num"])] = net
    return last


def link_rows(run: Run) -> list[dict[str, Any]]:
    """子→親の各リンク: 実測 (子の links.parent) と期待モデル数。"""
    last = _last_net(run)
    rounds = run.rounds or None
    rows = []
    for node in run.nodes:
        if node.parent_num is None:
            continue
        parent = run.by_num(node.parent_num)
        link = ((last.get(node.num) or {}).get("links") or {}).get("parent") or {}
        up, down = link.get("sent"), link.get("recv")
        expected = 1 if run.mode == "hfl" else len(run.subtree_clients(node.num))
        rows.append({
            "link": f"{node.id} -> {parent.id if parent else node.parent_num}",
            "child": node.id, "child_num": node.num,
            "parent": parent.id if parent else None, "parent_num": node.parent_num,
            "expected_models_per_round": expected,
            "up_bytes": up, "down_bytes": down,
            "up_mb_per_round": round(up / MB / rounds, 3) if up is not None and rounds else None,
            "down_mb_per_round": (
                round(down / MB / rounds, 3) if down is not None and rounds else None
            ),
        })
    return rows


def node_rows(run: Run) -> list[dict[str, Any]]:
    rows = []
    nodes = (run.network or {}).get("nodes") or {}
    rounds = run.rounds or None
    for node in run.nodes:
        info = next((v for k, v in nodes.items() if k.startswith(f"{node.num} ")), None)
        if info is None:
            info = nodes.get(str(node.num)) or nodes.get(node.id) or {}
        tx, rx = info.get("tx_bytes"), info.get("rx_bytes")
        fl_tx, other = info.get("fl_sent_bytes"), info.get("other_tx_bytes")

        def per(value: int | None) -> float | None:
            return round(value / MB / rounds, 3) if value is not None and rounds else None

        rows.append({
            "num": node.num, "id": node.id, "role": node.role,
            "tx_bytes": tx, "rx_bytes": rx, "fl_sent_bytes": fl_tx, "other_tx_bytes": other,
            "tx_mb_per_round": per(tx), "rx_mb_per_round": per(rx),
            "fl_tx_mb_per_round": per(fl_tx), "other_tx_mb_per_round": per(other),
        })
    return rows


def accuracy_summary(run: Run, threshold: float) -> dict[str, Any]:
    accs = [
        (e["global_round"], e.get("accuracy")) for e in run.evals if e.get("accuracy") is not None
    ]
    reached = next((r for r, a in accs if a >= threshold), None)
    return {
        "rounds": run.rounds,
        "final_accuracy": run.summary.get("final_accuracy"),
        "best_accuracy": run.summary.get("best_accuracy"),
        "threshold": threshold,
        "rounds_to_threshold": reached,
        "round_duration_median_s": (run.summary.get("round_duration_s") or {}).get("median"),
        "round_interval_s": round_interval_s(run.events),
        "wall_clock_h": run.summary.get("wall_clock_h"),
    }


def cumulative_fl_tx_by_round(run: Run) -> list[tuple[int, float]]:
    """各 Global 評価の時点での、全ノードの学習送信 (links の sent 合計) の累計 (MB)。"""
    by_node: dict[int, tuple[list[float], list[int]]] = defaultdict(lambda: ([], []))
    for record in sorted(run.metrics, key=lambda r: r["t"]):
        links = (record.get("net") or {}).get("links")
        if record.get("node_num") is None or not isinstance(links, dict):
            continue
        ts, values = by_node[int(record["node_num"])]
        ts.append(float(record["t"]))
        values.append(sum(int(v.get("sent", 0)) for v in links.values()))
    points = []
    for e in run.evals:
        t = _tkey(e)
        total = 0
        for ts, values in by_node.values():
            i = bisect.bisect_right(ts, t) - 1
            if i >= 0:
                total += values[i]
        points.append((int(e["global_round"]), total / MB))
    return points


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a == 0:
        return None
    return round(b / a, 3)


def compare(run_a: Run, run_b: Run, threshold: float) -> dict[str, Any]:
    links_a, links_b = link_rows(run_a), link_rows(run_b)
    links = []
    for la in links_a:
        lb = next((x for x in links_b if x["child"] == la["child"]), None)
        if lb is None:
            continue
        links.append({
            "link": la["link"],
            run_a.label: la, run_b.label: lb,
            "up_ratio": _ratio(la["up_mb_per_round"], lb["up_mb_per_round"]),
            "down_ratio": _ratio(la["down_mb_per_round"], lb["down_mb_per_round"]),
            "expected_ratio": _ratio(
                la["expected_models_per_round"], lb["expected_models_per_round"]
            ),
        })

    def totals(run: Run, rows: list[dict[str, Any]]) -> dict[str, Any]:
        up = [r["up_mb_per_round"] for r in rows if r["up_mb_per_round"] is not None]
        down = [r["down_mb_per_round"] for r in rows if r["down_mb_per_round"] is not None]
        link_total = round(sum(up) + sum(down), 3) if up or down else None
        tx_total = (run.network or {}).get("total_tx_bytes_per_round")
        return {
            "mode": run.mode,
            "rounds": run.rounds,
            "expected_models_per_round": sum(r["expected_models_per_round"] for r in rows),
            "links_up_mb_per_round": round(sum(up), 3) if up else None,
            "links_down_mb_per_round": round(sum(down), 3) if down else None,
            "links_total_mb_per_round": link_total,
            "links_total_gb_per_day_at_144_rounds": (
                round(link_total * ROUNDS_PER_DAY_PACED / 1000, 3) if link_total else None
            ),
            "all_nodes_tx_mb_per_round": round(tx_total / MB, 3) if tx_total else None,
        }

    tot_a, tot_b = totals(run_a, links_a), totals(run_b, links_b)
    return {
        "runs": {
            run_a.label: {"root": str(run_a.root), "run_id": run_a.summary.get("run_id")},
            run_b.label: {"root": str(run_b.root), "run_id": run_b.summary.get("run_id")},
        },
        "note": (
            "リンクの量は子ノードの links.parent (学習の接続の中身) をラウンド数で割ったもの。"
            "ラウンド間隔の違いには左右されない。ratio は B/A"
        ),
        "links": links,
        "nodes": {run_a.label: node_rows(run_a), run_b.label: node_rows(run_b)},
        "total": {
            run_a.label: tot_a, run_b.label: tot_b,
            "links_total_ratio": _ratio(
                tot_a["links_total_mb_per_round"], tot_b["links_total_mb_per_round"]
            ),
            "expected_ratio": _ratio(
                tot_a["expected_models_per_round"], tot_b["expected_models_per_round"]
            ),
            "all_nodes_tx_ratio": _ratio(
                tot_a["all_nodes_tx_mb_per_round"], tot_b["all_nodes_tx_mb_per_round"]
            ),
        },
        "accuracy": {
            run_a.label: accuracy_summary(run_a, threshold),
            run_b.label: accuracy_summary(run_b, threshold),
        },
    }


# ------------------------------------------------------------------ 出力
def write_csvs(result: dict[str, Any], labels: tuple[str, str], out: Path) -> None:
    with open(out / "links.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "link", "run", "expected_models_per_round", "up_mb_per_round", "down_mb_per_round",
            "up_bytes", "down_bytes",
        ])
        for row in result["links"]:
            for label in labels:
                r = row[label]
                writer.writerow([
                    row["link"], label, r["expected_models_per_round"], r["up_mb_per_round"],
                    r["down_mb_per_round"], r["up_bytes"], r["down_bytes"],
                ])
    with open(out / "nodes.csv", "w", newline="", encoding="utf-8") as f:
        columns = [
            "run", "num", "id", "role", "tx_mb_per_round", "rx_mb_per_round",
            "fl_tx_mb_per_round", "other_tx_mb_per_round", "tx_bytes", "rx_bytes",
        ]
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for label in labels:
            for row in result["nodes"][label]:
                writer.writerow({"run": label, **row})


def make_plots(run_a: Run, run_b: Run, result: dict[str, Any], out: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib がありません: uv sync --extra analysis を実行", file=sys.stderr)
        return []
    plt.rcParams.update(RCPARAMS)
    written: list[Path] = []
    runs = (run_a, run_b)
    colors = (SERIES[0], SERIES[1])

    def save(fig, name: str) -> None:
        path = out / name
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)

    def interval_note() -> str:
        parts = []
        for run in runs:
            interval = round_interval_s(run.events)
            pacing = f"{interval / 60:.0f}-min interval" if interval else "continuous"
            parts.append(f"{run.label}: {pacing}")
        return " / ".join(parts)

    # 1. リンクごとの上り・下り
    links = result["links"]
    if links and any(r[run.label]["up_mb_per_round"] is not None for r in links for run in runs):
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        x = np.arange(len(links))
        width = 0.38
        for ax, key, title in (
            (axes[0], "up_mb_per_round", "uplink (child -> parent)"),
            (axes[1], "down_mb_per_round", "downlink (parent -> child)"),
        ):
            for i, run in enumerate(runs):
                values = [r[run.label][key] or 0 for r in links]
                bars = ax.bar(
                    x + (i - 0.5) * width, values, width, color=colors[i], label=run.label
                )
                for bar, r in zip(bars, links):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"x{r[run.label]['expected_models_per_round']}",
                        ha="center", va="bottom", fontsize=8, color=INK2,
                    )
            ax.set_axisbelow(True)
            ax.set_ylabel("MB / round")
            ax.set_title(f"Bytes per round per link: {title}  (xN = expected model transfers)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([r["link"] for r in links], rotation=15, ha="right")
        axes[0].legend()
        save(fig, "link_bytes_per_round.png")

    # 2. ノードごとの送信 (学習分 + その他)
    rows = {run.label: result["nodes"][run.label] for run in runs}
    if any(r["tx_mb_per_round"] is not None for label in rows for r in rows[label]):
        fig, ax = plt.subplots(figsize=(11, 5))
        ids = [r["id"] for r in rows[run_a.label]]
        x = np.arange(len(ids))
        width = 0.38
        for i, run in enumerate(runs):
            fl = np.array([r["fl_tx_mb_per_round"] or 0 for r in rows[run.label]])
            other = np.array([r["other_tx_mb_per_round"] or 0 for r in rows[run.label]])
            pos = x + (i - 0.5) * width
            ax.bar(pos, fl, width, color=colors[i], label=f"{run.label}: learning")
            ax.bar(pos, other, width, bottom=fl, color=colors[i], alpha=0.35,
                   label=f"{run.label}: other (SSH, headers)")
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(ids, rotation=15, ha="right")
        ax.set_ylabel("MB sent / round")
        ax.set_title(f"Bytes sent per node per round  ({interval_note()})")
        ax.legend(ncol=2, fontsize=9)
        save(fig, "node_tx_per_round.png")

    # 3. 累計の学習送信量 vs ラウンド
    curves = {run.label: cumulative_fl_tx_by_round(run) for run in runs}
    if any(curves.values()):
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, run in enumerate(runs):
            pts = curves[run.label]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colors[i], label=run.label)
        ax.set_xlabel("Global round")
        ax.set_ylabel("MB (all nodes, learning traffic sent)")
        ax.set_title("Cumulative learning traffic vs round")
        ax.legend()
        save(fig, "cumulative_tx_vs_round.png")

    # 4, 5. 精度
    if any(run.evals for run in runs):
        for name, by_time in (("accuracy_vs_round.png", False), ("accuracy_vs_time.png", True)):
            fig, ax = plt.subplots(figsize=(11, 4.5))
            for i, run in enumerate(runs):
                if not run.evals:
                    continue
                t0 = _tkey(run.evals[0])
                xs = [(_tkey(e) - t0) / 3600 if by_time else e["global_round"] for e in run.evals]
                ys = [e.get("accuracy") for e in run.evals]
                ax.plot(xs, ys, color=colors[i], label=run.label)
            ax.set_ylim(0, 1)
            ax.set_ylabel("accuracy")
            ax.set_xlabel("hours since first evaluation" if by_time else "Global round")
            title = "Global accuracy (CIFAR-10 test)"
            if by_time:
                title += f"  ({interval_note()})"
            ax.set_title(title)
            ax.legend(loc="lower right")
            if by_time:
                ax.text(0.01, 0.02,
                        "time axis depends on the round interval; compare by round",
                        transform=ax.transAxes, fontsize=8, color=MUTED)
            save(fig, name)
    return written


def print_summary(result: dict[str, Any], labels: tuple[str, str]) -> None:
    a, b = labels
    print(f"link (child -> parent)          {a:>16} {b:>16}   ratio  expected")
    for row in result["links"]:
        ra, rb = row[a], row[b]
        up = (ra["up_mb_per_round"], rb["up_mb_per_round"])
        down = (ra["down_mb_per_round"], rb["down_mb_per_round"])
        fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
        print(
            f"  {row['link']:<28} up   {fmt(up[0]):>12} MB {fmt(up[1]):>12} MB  "
            f"{fmt(row['up_ratio']):>6}  x{ra['expected_models_per_round']}->"
            f"x{rb['expected_models_per_round']}"
        )
        print(
            f"  {'':<28} down {fmt(down[0]):>12} MB {fmt(down[1]):>12} MB  "
            f"{fmt(row['down_ratio']):>6}"
        )
    total = result["total"]
    print(
        f"total learning bytes on links / round: {a}={total[a]['links_total_mb_per_round']} MB, "
        f"{b}={total[b]['links_total_mb_per_round']} MB, ratio={total['links_total_ratio']} "
        f"(expected models {total[a]['expected_models_per_round']} -> "
        f"{total[b]['expected_models_per_round']}, ratio {total['expected_ratio']})"
    )
    for label in labels:
        acc = result["accuracy"][label]
        print(
            f"{label}: rounds={acc['rounds']} final acc={acc['final_accuracy']} "
            f"best={acc['best_accuracy']} reached {acc['threshold']} at round "
            f"{acc['rounds_to_threshold']} round median={acc['round_duration_median_s']} s "
            f"interval={acc['round_interval_s']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two HFL/flat FL runs")
    parser.add_argument("run_a", help="collected/<run-id> (基準。例: HFL)")
    parser.add_argument("run_b", help="collected/<run-id> (比較対象。例: flat FL)")
    parser.add_argument("--labels", nargs=2, default=("HFL", "FlatFL"), metavar=("A", "B"))
    parser.add_argument("--profile-a", default=None, help="nodes.json が無いときの台帳")
    parser.add_argument("--profile-b", default=None, help="nodes.json が無いときの台帳")
    parser.add_argument("--out", default=None, help="出力先 (既定: compare/<A>_vs_<B>)")
    parser.add_argument("--acc-threshold", type=float, default=0.7)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    labels = (args.labels[0], args.labels[1])
    if labels[0] == labels[1]:
        parser.error("--labels must be different")
    run_a = load_run(labels[0], Path(args.run_a), args.profile_a)
    run_b = load_run(labels[1], Path(args.run_b), args.profile_b)
    result = compare(run_a, run_b, args.acc_threshold)
    default_out = Path("compare") / f"{Path(args.run_a).name}_vs_{Path(args.run_b).name}"
    out = Path(args.out) if args.out else default_out
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csvs(result, labels, out)
    written = [] if args.no_plots else make_plots(run_a, run_b, result, out)
    print_summary(result, labels)
    print(f"wrote {out / 'comparison.json'}")
    for path in written:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
