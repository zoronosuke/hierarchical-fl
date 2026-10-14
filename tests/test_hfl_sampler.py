"""Jetson 側の記録 (scripts/hfl_sampler.py) のテスト。/sys と ss は偽物を使う。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("hfl_sampler", REPO / "scripts" / "hfl_sampler.py")
hfl_sampler = importlib.util.module_from_spec(spec)
sys.modules["hfl_sampler"] = hfl_sampler
spec.loader.exec_module(hfl_sampler)

# .202 (Edge 01) で取得した `ss -tinH state established` の出力 (一部を短縮)
SS_EDGE01 = """\
0      0      [::ffff:192.168.10.202]:9001  [::ffff:192.168.10.204]:50418
\t cubic wscale:7,7 bytes_sent:419223509 bytes_acked:419223509 bytes_received:418536727 segs_out:9
0      0      [::ffff:192.168.10.202]:40624 [::ffff:192.168.10.201]:8080
\t cubic wscale:7,7 rto:204 bytes_sent:418536702 bytes_acked:418536703 bytes_received:419212475
0      0      192.168.10.202:22 192.168.10.133:55677
\t cubic wscale:7,7 rto:204 bytes_sent:5120 bytes_received:3000
0      0      [::ffff:127.0.0.1]:9001 [::ffff:127.0.0.1]:53312
\t cubic bytes_sent:999 bytes_received:999
0      0      [::ffff:192.168.10.202]:9001  [::ffff:192.168.10.205]:45498
\t cubic wscale:7,7 rto:204 mss:1448 segs_out:3
"""

PARENT = ("192.168.10.201", 8080)


def test_parse_ss_normalizes_addresses_and_missing_fields():
    connections = hfl_sampler.parse_ss(SS_EDGE01)
    assert len(connections) == 5
    local, peer, sent, received = connections[0]
    assert local == ("192.168.10.202", 9001) and peer == ("192.168.10.204", 50418)
    assert (sent, received) == (419223509, 418536727)
    assert connections[4][2:] == (0, 0)  # 値が 0 の項目は ss が省略する


def test_link_counter_baseline_deltas_and_reconnects():
    counter = hfl_sampler.LinkCounter(PARENT, 9001)
    counter.update(hfl_sampler.parse_ss(SS_EDGE01))  # 記録開始時の接続は基準にするだけ
    assert counter.totals == {}

    later = (
        SS_EDGE01.replace("bytes_sent:419223509", "bytes_sent:420000000")
        .replace("bytes_received:419212475", "bytes_received:419312475")
    )
    counter.update(hfl_sampler.parse_ss(later))
    assert counter.totals["child:192.168.10.204"] == {"sent": 420000000 - 419223509, "recv": 0}
    assert counter.totals["parent"] == {"sent": 0, "recv": 100000}
    assert not any(k.startswith("child:127") for k in counter.totals)  # 同じ Jetson 内は数えない
    assert "child:192.168.10.133" not in counter.totals  # SSH は数えない

    # resume で親との接続を張り直した: 新しい接続は 0 から数える
    reconnected = later.replace(":40624", ":40999").replace(
        "bytes_sent:418536702", "bytes_sent:2000"
    )
    counter.update(hfl_sampler.parse_ss(reconnected))
    assert counter.totals["parent"]["sent"] == 2000


def _fake_sys(root: Path, tx: int, rx: int) -> None:
    hwmon = root / "class/hwmon/hwmon1"
    hwmon.mkdir(parents=True)
    (hwmon / "name").write_text("ina3221\n")
    for i, (label, mv, ma) in enumerate(
        [("VDD_IN", 5000, 1100), ("VDD_CPU_GPU_CV", 5000, 300), ("VDD_SOC", 5000, 350)], 1
    ):
        (hwmon / f"in{i}_label").write_text(label)
        (hwmon / f"in{i}_input").write_text(str(mv))
        (hwmon / f"curr{i}_input").write_text(str(ma))
    zone = root / "class/thermal/thermal_zone8"
    zone.mkdir(parents=True)
    (zone / "type").write_text("tj-thermal")
    (zone / "temp").write_text("47125")
    stats = root / "class/net/enP8p1s0/statistics"
    stats.mkdir(parents=True)
    (stats / "tx_bytes").write_text(str(tx))
    (stats / "rx_bytes").write_text(str(rx))
    gpu = root / "devices/platform/gpu.0"
    gpu.mkdir(parents=True)
    (gpu / "load").write_text("723")


def test_sensor_readers(tmp_path):
    _fake_sys(tmp_path, tx=100, rx=200)
    assert hfl_sampler.read_power(tmp_path) == {
        "VDD_IN": 5500.0, "VDD_CPU_GPU_CV": 1500.0, "VDD_SOC": 1750.0,
    }
    assert hfl_sampler.read_temps(tmp_path) == {"tj": 47.1}
    assert hfl_sampler.read_gpu_load(tmp_path) == 72.3
    assert hfl_sampler.read_iface_counters(tmp_path) == {"enP8p1s0": [200, 100]}


def _sampler(tmp_path, monkeypatch, raw, boot="boot-a"):
    monkeypatch.setattr(hfl_sampler, "iface_for_ip", lambda ip: "enP8p1s0")
    monkeypatch.setattr(hfl_sampler, "read_iface_counters", lambda: {"enP8p1s0": list(raw)})
    monkeypatch.setattr(hfl_sampler, "read_boot_id", lambda: boot)
    monkeypatch.setattr(hfl_sampler, "read_connections", lambda: [])
    return hfl_sampler.Sampler(
        run_id="r", node_id="edge_01", node_num=202, node_ip="192.168.10.202",
        parent=PARENT, own_port=9001,
        out_path=tmp_path / "metrics-202.jsonl", pid_path=tmp_path / "hfl-202.pid",
    )


def test_totals_continue_across_restart_and_reboot(tmp_path, monkeypatch):
    raw = [1000, 5000]  # [rx, tx]
    sampler = _sampler(tmp_path, monkeypatch, raw)
    sampler.write(sampler.sample())  # 基準
    raw[:] = [1500, 7000]
    record = sampler.sample()
    sampler.write(record)
    assert (record["net"]["rx_bytes"], record["net"]["tx_bytes"]) == (500, 2000)

    # 同じ起動中に記録プロセスだけ再起動: ファイル末尾から累計を引き継ぐ
    raw[:] = [1600, 7500]
    restarted = _sampler(tmp_path, monkeypatch, raw)
    record = restarted.sample()
    restarted.write(record)
    assert (record["net"]["rx_bytes"], record["net"]["tx_bytes"]) == (600, 2500)

    # 電源断で Jetson が再起動 (boot_id が変わり、カウンターは 0 から): 起動後の分を足す
    raw[:] = [40, 90]
    rebooted = _sampler(tmp_path, monkeypatch, raw, boot="boot-b")
    record = rebooted.sample()
    assert (record["net"]["rx_bytes"], record["net"]["tx_bytes"]) == (640, 2590)


def test_half_written_last_line_is_closed(tmp_path, monkeypatch):
    path = tmp_path / "metrics-202.jsonl"
    good = json.dumps({"t": 1, "net": {"tx_bytes": 10, "rx_bytes": 20, "iface_raw": [1, 2]},
                       "boot_id": "boot-a"})
    path.write_text(good + "\n" + '{"t": 2, "net": {"tx_b')  # 電源断で途中まで
    sampler = _sampler(tmp_path, monkeypatch, [1, 2])
    sampler.write(sampler.sample())
    lines = path.read_text().splitlines()
    assert lines[1].startswith('{"t": 2')  # 途中の行は残すが閉じる
    last = json.loads(lines[2])
    assert last["net"]["tx_bytes"] == 10  # 最後の完全な行から引き継いだ


def test_unreadable_sensor_is_skipped(tmp_path, monkeypatch):
    """Jetson の cv*-thermal は読むと EAGAIN (Python では TypeError) になる。その項目だけ飛ばす。"""
    _fake_sys(tmp_path, tx=1, rx=1)
    zone = tmp_path / "class/thermal/thermal_zone2"
    zone.mkdir(parents=True)
    (zone / "type").write_text("cv0-thermal")
    (zone / "temp").write_text("0")
    original = Path.read_text

    def flaky(self, *args, **kwargs):
        if self == zone / "temp":
            raise TypeError("can't concat NoneType to bytes")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky)
    assert hfl_sampler.read_temps(tmp_path) == {"tj": 47.1}
