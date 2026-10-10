"""JSONL イベント記録のテスト。"""

from __future__ import annotations

import json

import numpy as np

from src.utils import events
from src.utils.events import EventLogger, get_event_logger


def test_disabled_logger_is_noop(tmp_path, monkeypatch):
    monkeypatch.delenv(events.ENV_PATH, raising=False)
    logger = get_event_logger("leaf")
    assert not logger.enabled
    logger.emit("fit_end", loss=1.0)  # 何も起きない
    logger.emit_error(RuntimeError("x"))
    logger.close()
    assert list(tmp_path.iterdir()) == []


def test_events_are_appended_as_json_lines(tmp_path, monkeypatch):
    path = tmp_path / "sub" / "events-202.jsonl"
    monkeypatch.setenv(events.ENV_PATH, str(path))
    monkeypatch.setenv(events.ENV_RUN_ID, "run-1")
    monkeypatch.setenv(events.ENV_NODE_ID, "edge_01")
    monkeypatch.setenv(events.ENV_NODE_NUM, "202")

    logger = get_event_logger("edge")
    assert logger.enabled
    logger.emit("start", argv=["--edge-config", "x.yaml"])
    logger.emit(
        "fit_end", global_round=3, loss=np.float32(0.5), accuracy=0.75, participants=["a", "b"]
    )
    try:
        raise ValueError("boom")
    except ValueError as exc:
        logger.emit_error(exc, global_round=3)
    logger.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]
    for record in records:
        assert record["run_id"] == "run-1"
        assert record["node"] == "edge_01"
        assert record["node_num"] == 202
        assert record["proc"] == "edge"
        assert isinstance(record["t"], float)
        assert record["ts"][-6] in "+-"  # タイムゾーンオフセット付き ISO 8601
        assert record["pid"] > 0 and record["host"]
    assert records[0]["event"] == "start"
    assert records[1]["loss"] == 0.5 and records[1]["participants"] == ["a", "b"]
    assert records[2]["event"] == "error"
    assert records[2]["exc_type"] == "ValueError"
    assert "boom" in records[2]["traceback"]


def test_second_logger_appends_to_existing_file(tmp_path):
    path = tmp_path / "events.jsonl"
    first = EventLogger(str(path), run_id="r", node_id="n", node_num=1, proc="a")
    first.emit("start")
    first.close()
    second = EventLogger(str(path), run_id="r", node_id="n", node_num=1, proc="b")
    second.emit("stop")
    second.close()
    procs = [json.loads(line)["proc"] for line in path.read_text().splitlines()]
    assert procs == ["a", "b"]
