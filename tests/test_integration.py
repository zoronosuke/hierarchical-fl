"""1PC・7プロセスの同期HFL疎通試験。"""

from __future__ import annotations

import json
import subprocess
import sys
import time
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


def _events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # 書き込み途中の最終行
    return records


def _run_until_round(repository: Path, args: list[str], run_dir: Path, until_round: int,
                     log_path: Path, timeout: float = 300.0) -> int:
    """run.py を起動し、Global が until_round まで評価したら STOP を作って終了を待つ。"""
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "run.py", *args], cwd=repository, stdout=log, stderr=subprocess.STDOUT
        )
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline and process.poll() is None:
                rounds = [
                    r.get("global_round", 0)
                    for r in _events(run_dir / "events-global.jsonl")
                    if r["event"] == "eval" and not r.get("resumed")
                ]
                if rounds and max(rounds) >= until_round:
                    (run_dir / "STOP").touch()
                    break
                time.sleep(1)
            return process.wait(timeout=max(deadline - time.monotonic(), 30))
        finally:
            if process.poll() is None:
                process.kill()


def test_stop_and_resume_from_checkpoint(tmp_path):
    """停止 → 同じ run-id で --resume → 通しのラウンド番号で続きから学習する。"""
    repository = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "resume"
    common = [
        "--dry-run",
        "--global-config", "config/local-5tier/global_longrun.yaml",
        "--topology-config", "config/local-5tier/topology.yaml",
        "--run-id", "resume", "--log-dir", str(tmp_path),
        "--max-hours", "1",
    ]

    code = _run_until_round(repository, common, run_dir, 3, tmp_path / "run1.log")
    log1 = (tmp_path / "run1.log").read_text(encoding="utf-8", errors="replace")
    assert code == 0, log1
    stop1 = [r for r in _events(run_dir / "events-global.jsonl") if r["event"] == "stop"]
    assert stop1 and stop1[-1]["reason"] == "stop_file", stop1
    first_last = stop1[-1]["global_round"]
    assert first_last >= 3

    from src.utils.checkpoint import load_checkpoint

    _, meta = load_checkpoint(run_dir / "checkpoint.npz")
    assert meta["global_round"] == first_last

    code = _run_until_round(
        repository, [*common, "--resume"], run_dir, first_last + 3, tmp_path / "run2.log"
    )
    log2 = (tmp_path / "run2.log").read_text(encoding="utf-8", errors="replace")
    assert code == 0, log2
    assert "Traceback" not in log1 + log2

    global_events = _events(run_dir / "events-global.jsonl")
    resumes = [r for r in global_events if r["event"] == "resume"]
    assert len(resumes) == 1 and resumes[0]["global_round"] == first_last
    after = global_events[global_events.index(resumes[0]):]
    evals = [r["global_round"] for r in after if r["event"] == "eval" and not r.get("resumed")]
    assert evals[0] == first_last + 1 and evals == sorted(evals)
    stops = [r for r in after if r["event"] == "stop"]
    assert stops[-1]["global_round"] >= first_last + 3

    # 下位ノードも通しのラウンド番号で記録される
    leaf_fits = [
        r["global_round"] for r in _events(run_dir / "events-leaf_04.jsonl")
        if r["event"] == "fit_end"
    ]
    assert first_last + 1 in leaf_fits and max(leaf_fits) >= first_last + 3
    for name in ("global", "edge_01", "edge_02", "leaf_01", "leaf_03", "leaf_02", "leaf_04"):
        assert not [r for r in _events(run_dir / f"events-{name}.jsonl") if r["event"] == "error"]
