"""2 run の通信量・精度比較 (scripts/compare_runs.py) のテスト。合成データで比と表を検証する。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("compare_runs", REPO / "scripts" / "compare_runs.py")
compare_runs = importlib.util.module_from_spec(spec)
sys.modules["compare_runs"] = compare_runs
spec.loader.exec_module(compare_runs)

MODEL = 755_000  # 1 モデルのバイト数
# num: (id, role, parent)
HFL_NODES = {
    201: ("global", "global", None), 202: ("edge_01", "edge", 201), 203: ("edge_02", "edge", 201),
    204: ("leaf_01", "edge", 202), 205: ("leaf_02", "leaf", 202), 206: ("leaf_03", "edge", 204),
    207: ("leaf_04", "leaf", 206),
}
# flat: 子の部分木のクライアント数 = 期待モデル数
FLAT_CLIENTS = {202: 5, 203: 1, 204: 3, 205: 1, 206: 2, 207: 1}
INTERNAL = {202: "edge_01_internal", 203: "edge_02_internal", 204: "leaf_01_internal",
            206: "leaf_03_internal", 205: "leaf_02", 207: "leaf_04"}


def _jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _make_run(root: Path, run_id: str, mode: str, rounds: int, accs: list[float]) -> None:
    nodes = []
    for num, (node_id, role, parent) in HFL_NODES.items():
        if mode == "flat" and role == "edge":
            role = "relay"
        nodes.append({"num": num, "id": node_id, "role": role, "parent_num": parent,
                      "ip": f"192.168.10.{num}", "port": 8080 if num == 201 else None})
    (root / "nodes.json").write_text(json.dumps({"nodes": nodes}), encoding="utf-8")
    t0 = 1_000_000.0
    interval = 600.0 if mode == "hfl" else 20.0
    global_events = [{"t": t0, "proc": "global", "event": "start", "node": "global",
                      "node_num": 201, "run_id": run_id,
                      "round_interval_s": 600 if mode == "hfl" else None}]
    for r in range(1, rounds + 1):
        t = t0 + r * interval
        global_events.append({"t": t - 10, "proc": "global", "event": "round_start",
                              "node": "global", "node_num": 201, "global_round": r})
        global_events.append({"t": t, "proc": "global", "event": "eval", "node": "global",
                              "node_num": 201, "global_round": r, "accuracy": accs[r - 1],
                              "loss": 1.0, "round_duration_s": 10.0})
    (root / "201").mkdir()
    _jsonl(root / "201" / "events-201.jsonl", global_events)
    for num, (node_id, role, parent) in HFL_NODES.items():
        if num == 201:
            continue
        d = root / str(num)
        d.mkdir()
        proc = "leaf" if role == "leaf" else "internal_client"
        fits = [{"t": t0 + r * interval - 5, "proc": proc, "event": "fit_start", "node": node_id,
                 "node_num": num, "global_round": r, "client_id": INTERNAL[num]}
                for r in range(1, rounds + 1)]
        _jsonl(d / f"events-{num}.jsonl", fits)
        models = 1 if mode == "hfl" else FLAT_CLIENTS[num]
        records = []
        for r in range(0, rounds + 1):
            sent = models * MODEL * r
            records.append({
                "t": t0 + r * interval + 1, "node": node_id, "node_num": num,
                "net": {"tx_bytes": sent + 100_000 * (r + 1), "rx_bytes": sent,
                        "links": {"parent": {"sent": sent, "recv": sent}}},
            })
        _jsonl(d / f"metrics-{num}.jsonl", records)


def test_compare_links_and_totals(tmp_path):
    hfl, flat = tmp_path / "hfl", tmp_path / "flat"
    hfl.mkdir()
    flat.mkdir()
    accs_hfl = [0.3, 0.5, 0.6, 0.65, 0.7, 0.72, 0.74, 0.75, 0.76, 0.76]
    accs_flat = [0.3, 0.5, 0.6, 0.66, 0.69, 0.71, 0.73, 0.75, 0.76, 0.77]
    _make_run(hfl, "hfl-01", "hfl", 10, accs_hfl)
    _make_run(flat, "flat-01", "flat", 10, accs_flat)
    out = tmp_path / "out"
    assert compare_runs.main([str(hfl), str(flat), "--out", str(out), "--no-plots"]) == 0
    result = json.loads((out / "comparison.json").read_text(encoding="utf-8"))

    links = {row["link"]: row for row in result["links"]}
    assert set(links) == {
        "edge_01 -> global", "edge_02 -> global", "leaf_01 -> edge_01", "leaf_02 -> edge_01",
        "leaf_03 -> leaf_01", "leaf_04 -> leaf_03",
    }
    l01 = links["leaf_01 -> edge_01"]
    assert l01["HFL"]["expected_models_per_round"] == 1
    assert l01["FlatFL"]["expected_models_per_round"] == 3
    assert l01["HFL"]["up_mb_per_round"] == 0.755
    assert l01["FlatFL"]["up_mb_per_round"] == round(3 * 0.755, 3)
    assert l01["up_ratio"] == 3.0 and l01["down_ratio"] == 3.0 and l01["expected_ratio"] == 3.0
    assert links["edge_01 -> global"]["FlatFL"]["expected_models_per_round"] == 5

    total = result["total"]
    assert total["HFL"]["expected_models_per_round"] == 6
    assert total["FlatFL"]["expected_models_per_round"] == 13
    assert total["HFL"]["links_total_mb_per_round"] == round(6 * 2 * 0.755, 3)
    assert total["links_total_ratio"] == round(13 / 6, 3)
    assert total["expected_ratio"] == round(13 / 6, 3)
    assert total["HFL"]["mode"] == "hfl" and total["FlatFL"]["mode"] == "flat"

    acc = result["accuracy"]
    assert acc["HFL"]["rounds_to_threshold"] == 5 and acc["FlatFL"]["rounds_to_threshold"] == 6
    assert acc["HFL"]["round_interval_s"] == 600 and acc["FlatFL"]["round_interval_s"] is None
    assert acc["FlatFL"]["final_accuracy"] == 0.77

    nodes = {row["id"]: row for row in result["nodes"]["FlatFL"]}
    assert nodes["leaf_01"]["fl_tx_mb_per_round"] == round(3 * 0.755, 3)
    assert nodes["leaf_01"]["other_tx_mb_per_round"] == round(1_100_000 / 1e6 / 10, 3)

    links_csv = (out / "links.csv").read_text(encoding="utf-8").splitlines()
    assert links_csv[0].startswith("link,run,expected_models_per_round")
    assert len(links_csv) == 1 + 6 * 2
    assert (out / "nodes.csv").exists()


def test_cumulative_learning_traffic_follows_rounds(tmp_path):
    root = tmp_path / "flat"
    root.mkdir()
    _make_run(root, "flat-02", "flat", 4, [0.3, 0.4, 0.5, 0.6])
    run = compare_runs.load_run("FlatFL", root, None)
    points = compare_runs.cumulative_fl_tx_by_round(run)
    assert [p[0] for p in points] == [1, 2, 3, 4]
    # 各評価時点では、その直前の記録 (= r-1 ラウンド分) が最新
    expected_per_round = sum(FLAT_CLIENTS.values()) * MODEL / 1e6
    assert abs(points[1][1] - 1 * expected_per_round) < 1e-6
    assert points[3][1] > points[2][1] > points[1][1]
