"""1PC・7プロセスの同期HFL疎通試験。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_persistent_hfl_two_global_rounds():
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "run.py",
            "--dry-run",
            "--completion-timeout",
            "120",
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=150,
        check=False,
    )
    log = completed.stdout + completed.stderr
    assert completed.returncode == 0, log
    assert log.count("Persistent sub-server starting on") == 2
    assert "Connection refused" not in log
    assert "All processes completed successfully" in log
    assert "global=0, edge_01=0, edge_02=0" in log

    for client_id in (
        "leaf_01",
        "leaf_02",
        "leaf_03",
        "leaf_04",
        "edge_01_internal",
        "edge_02_internal",
    ):
        assert log.count(f"[{client_id}] fit complete:") == 2

    for edge_id in ("edge_01", "edge_02"):
        assert log.count(f"[{edge_id}] Parent generation complete:") == 2
        assert (
            f"[{edge_id}] Parent generation complete: "
            "parent_round=2, child_round=2, results=3"
        ) in log

    final_child_completion = max(
        log.index(
            f"[{edge_id}] Parent generation complete: "
            "parent_round=2, child_round=2, results=3"
        )
        for edge_id in ("edge_01", "edge_02")
    )
    first_disconnect = log.find("Disconnect and shut down")
    assert first_disconnect == -1 or first_disconnect > final_child_completion


def test_max_hours_stops_all_processes_cleanly(tmp_path):
    """実質無限ラウンドを時間上限 (0.03 h = 108 s) で止め、全プロセスが正常終了する。"""
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "run.py",
            "--dry-run",
            "--global-config",
            "config/local-5tier/global_longrun.yaml",
            "--topology-config",
            "config/local-5tier/topology.yaml",
            "--run-id",
            "maxhours",
            "--log-dir",
            str(tmp_path),
            "--completion-timeout",
            "300",
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=360,
        check=False,
    )
    log = completed.stdout + completed.stderr
    assert completed.returncode == 0, log
    assert "Stopping at round boundary: reason=max_duration" in log
    assert "Traceback" not in log
    assert "All processes completed successfully" in log
    # 少なくとも数ラウンドは回っている (dry-run の1ラウンドは数秒)
    assert log.count("[Global Eval] Round") >= 2

    run_dir = tmp_path / "maxhours"
    stops = {}
    for name in ("global", "edge_01", "edge_02", "leaf_01", "leaf_03", "leaf_02", "leaf_04"):
        events_file = run_dir / f"events-{name}.jsonl"
        assert events_file.exists(), f"missing {events_file}"
        lines = events_file.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
        assert not [r for r in records if r["event"] == "error"], name
        stops[name] = {(r["proc"], r.get("reason")) for r in records if r["event"] == "stop"}
    assert ("global", "max_duration") in stops["global"]
    assert ("leaf", "finished") in stops["leaf_02"] and ("leaf", "finished") in stops["leaf_04"]
    for edge in ("edge_01", "edge_02", "leaf_01", "leaf_03"):
        assert ("edge", "finished") in stops[edge], stops[edge]
        assert ("sub_server", "stop_requested") in stops[edge], stops[edge]
        assert ("internal_client", "finished") in stops[edge], stops[edge]
    # Global の eval イベントに global_round と所要時間が記録されている
    global_evals = [
        json.loads(line)
        for line in (run_dir / "events-global.jsonl").read_text(encoding="utf-8").splitlines()
        if '"event": "eval"' in line
    ]
    assert any(
        r.get("global_round", 0) >= 3 and r.get("round_duration_s") is not None
        for r in global_evals
    )


def test_persistent_hfl_5tier_two_global_rounds():
    """Global - Edge 01 - L01 - L03 - L04 の5段構成を1PCで2ラウンド実行する。"""
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "run.py",
            "--dry-run",
            "--topology-config",
            "config/local-5tier/topology.yaml",
            "--completion-timeout",
            "150",
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=200,
        check=False,
    )
    log = completed.stdout + completed.stderr
    assert completed.returncode == 0, log
    assert log.count("Persistent sub-server starting on") == 4
    assert "All processes completed successfully" in log

    for client_id in (
        "edge_01_internal",
        "leaf_01_internal",
        "leaf_02",
        "edge_02_internal",
        "leaf_03_internal",
        "leaf_04",
    ):
        assert log.count(f"[{client_id}] fit complete:") == 2

    for edge_id, results in (
        ("edge_01", 3),
        ("edge_02", 1),
        ("leaf_01", 2),
        ("leaf_03", 2),
    ):
        assert log.count(f"[{edge_id}] Parent generation complete:") == 2
        assert (
            f"[{edge_id}] Parent generation complete: "
            f"parent_round=2, child_round=2, results={results}"
        ) in log
