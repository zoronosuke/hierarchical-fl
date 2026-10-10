"""1PC・7プロセスの同期HFL疎通試験。"""

from __future__ import annotations

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
