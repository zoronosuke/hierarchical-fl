"""監視役 (scripts/hfl_monitor.py) のミラー・解析・集計のテスト。SSH は使わない。"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("hfl_monitor", REPO / "scripts" / "hfl_monitor.py")
hfl_monitor = importlib.util.module_from_spec(spec)
sys.modules["hfl_monitor"] = hfl_monitor
spec.loader.exec_module(hfl_monitor)

from hfl_nodes import load_inventory  # noqa: E402


def _event(t, event, **fields):
    return json.dumps({"t": t, "event": event, **fields}) + "\n"


def test_parse_probe_output_reads_exact_byte_chunks():
    body = b'{"a": 1}\n__FILE__ looks like a header but is data\n'
    raw = (
        b"__TIME__ 100.5\n__ALIVE__ 1\n__PID__ \n"
        b"__PWR__ VDD_IN 5000 1100\n__TEMP__ tj-thermal 47125\n"
        + f"__FILE__ events-202.jsonl 10 {len(body)} 200\n".encode() + body
        + b"__FILE__ hfl-202.log 0 0 -1\n__END__\n"
    )
    probe = hfl_monitor.parse_probe_output(raw)
    assert probe.complete
    assert probe.fields["TIME"] == "100.5" and probe.fields["PID"] == ""
    chunk = probe.files["events-202.jsonl"]
    assert (chunk.offset, chunk.data, chunk.size) == (10, body, 200)
    assert probe.files["hfl-202.log"].size == -1
    metrics = hfl_monitor.parse_metrics(probe)
    assert metrics["power_mw"] == {"VDD_IN": 5500.0}
    assert metrics["temp_c"] == {"tj": 47.1}


def test_mirror_appends_complete_lines_and_handles_truncation(tmp_path):
    store = hfl_monitor.MirrorStore(tmp_path)
    chunk = hfl_monitor.FileChunk
    assert store.apply(202, "e.jsonl", chunk(0, b"line1\nline2\npart", 17)) == b"line1\nline2\n"
    assert store.offset(202, "e.jsonl") == 12
    # 想定外の offset は取り込まない
    assert store.apply(202, "e.jsonl", chunk(5, b"x\n", 17)) == b""
    assert store.apply(202, "e.jsonl", chunk(12, b"partial3\n", 21)) == b"partial3\n"
    # 元ファイルが作り直された (offset 0 から送られてきた) ら、前のコピーを残して新しく始める
    assert store.apply(202, "e.jsonl", chunk(0, b"new\n", 4)) == b"new\n"
    assert store.path(202, "e.jsonl").read_bytes() == b"new\n"
    assert len(list((tmp_path / "202").glob("e.jsonl.prev-*"))) == 1


def test_run_state_records_one_point_per_round_and_resets_tracebacks():
    state = hfl_monitor.RunState()
    events = [
        {"t": 1, "event": "fit_start", "global_round": 1},
        {"t": 2, "event": "fit_end", "global_round": 1, "loss": 1.0, "accuracy": 0.5},
        {"t": 3, "event": "eval", "global_round": 1, "loss": 2.0, "accuracy": 0.3},
        {"t": 4, "event": "fit_start", "global_round": 2},
        {"t": 5, "event": "resume", "global_round": 1, "downtime_s": 60},
        {"t": 6, "event": "eval", "global_round": 1, "resumed": True, "loss": 2, "accuracy": 0.3},
        {"t": 7, "event": "eval", "global_round": 2, "loss": 1.5, "accuracy": 0.4},
    ]
    state.consume_events(201, "global", events)
    assert state.round_series["201"] == [[1, 1], [4, 2], [5, 1], [7, 2]]
    assert [p["round"] for p in state.global_series] == [1, 2]
    assert state.resumes == [{"t": 5, "round": 1, "downtime_s": 60}]

    state.consume_log(202, b"Traceback (most recent call last)\nboom\n")
    assert state.node(202).traceback_count == 1
    state.consume_log(202, b"===== [hfl] start 2026-10-13 node=202 resume=1 =====\nok\n")
    assert state.node(202).traceback_count == 0
    assert state.node(202).log_tail[-1] == "ok"


class _FakeJetsons:
    """各ノードの記録ファイルを持ち、調査スクリプトに対する出力を作る。"""

    def __init__(self) -> None:
        self.files: dict[tuple[int, str], bytes] = {}

    def __call__(self, node, script):
        specs = re.search(r"for spec in (.*); do", script).group(1).split()
        out = ["__TIME__ 1000.0\n__ALIVE__ 1\n__PID__ 42\n__EXIT__ \n__PORT__ -\n"
               "__PWR__ VDD_IN 5000 1000\n__TEMP__ tj-thermal 50000\n".encode()]
        for item in specs:
            name, offset = item.rsplit(":", 1)
            data = self.files.get((node.num, name))
            if data is None:
                out.append(f"__FILE__ {name} 0 0 -1\n".encode())
                continue
            offset = int(offset)
            if offset < 0:
                offset = max(len(data) + offset, 0)
            chunk = data[offset:]
            out.append(f"__FILE__ {name} {offset} {len(chunk)} {len(data)}\n".encode() + chunk)
        out.append(b"__END__\n")
        return b"".join(out), "", True


def test_monitor_mirrors_increments_and_rebuilds_after_restart(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    inventory = load_inventory("config/jetson-5tier-longrun")
    jetsons = _FakeJetsons()
    jetsons.files[(201, "events-201.jsonl")] = (
        _event(10, "eval", global_round=1, loss=2.0, accuracy=0.3)
        + _event(20, "eval", global_round=2, loss=1.5, accuracy=0.5)
    ).encode()
    jetsons.files[(201, "hfl-201.log")] = b"===== [hfl] start =====\nhello\n"

    monitor = hfl_monitor.Monitor(
        inventory, "run", tmp_path, interval=15, identity=None, runner=jetsons
    )
    status = monitor.poll_once()
    assert [p["round"] for p in status["global_series"]] == [1, 2]
    assert status["summary"]["total_power_w"] == 35.0  # 7台 × 5 W
    assert (tmp_path / "status.json").exists() and (tmp_path / "metrics.jsonl").exists()

    # 記録が増えた分だけ取り込む
    jetsons.files[(201, "events-201.jsonl")] += _event(30, "eval", global_round=3, loss=1.2,
                                                       accuracy=0.6).encode()
    status = monitor.poll_once()
    assert [p["round"] for p in status["global_series"]] == [1, 2, 3]
    mirrored = (tmp_path / "mirror" / "201" / "events-201.jsonl").read_bytes()
    assert mirrored == jetsons.files[(201, "events-201.jsonl")]

    # 監視役を作り直しても、手元のミラーから最初の記録まで復元する
    restarted = hfl_monitor.Monitor(
        inventory, "run", tmp_path, interval=15, identity=None, runner=jetsons
    )
    assert [p["round"] for p in restarted.state.global_series] == [1, 2, 3]
    assert len(restarted.state.metric_series["201"]) == 1  # 1分に1点


def test_once_mode_reads_tail_without_writing(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    inventory = load_inventory("config/jetson-5tier-longrun")
    jetsons = _FakeJetsons()
    jetsons.files[(201, "events-201.jsonl")] = _event(10, "eval", global_round=7, loss=1,
                                                      accuracy=0.7).encode()
    monitor = hfl_monitor.Monitor(
        inventory, "run", tmp_path, interval=15, identity=None, mirror=False, runner=jetsons
    )
    status = monitor.poll_once()
    assert status["summary"]["global_round"] == 7
    assert not any(tmp_path.iterdir())
