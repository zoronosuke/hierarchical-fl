#!/usr/bin/env python
"""
Jetson 側の記録 (電力・温度・GPU・メモリ・通信量)。

学習とは別プロセスとして scripts/hfl.sh start が起動し、interval 秒ごとに
logs/<run-id>/metrics-<NUM>.jsonl に1行追記する (行ごとに fsync。電源断でも直前まで残る)。
学習が fail-fast で落ちても記録を続ける (停止中の消費電力も残す)。監視役 (hfl_monitor.py) は
このファイルを events と同じように差分コピーする。

Usage (Jetson のリポジトリルートで):
    .venv/bin/python scripts/hfl_sampler.py --profile config/jetson-5tier-paced \
        --run-id RUN --node-num 202 [--interval 5]

1行の主な項目:
    power_mw      INA3221 の電力 (VDD_IN = ボード入力全体、VDD_CPU_GPU_CV、VDD_SOC)
    temp_c        温度 (cpu, gpu, soc0..2, tj = 最高温部)
    gpu_load      GPU 使用率 (%)、fan_rpm、mem_used_mb、load1
    fl_alive      学習プロセスが動いているか
    net           iface (自ノードの IP を持つ口)、tx_bytes / rx_bytes (記録開始からの累計)、
                  links (学習の接続ごとの累計: parent / child:<IP>)、raw (各インターフェースの
                  起動時からのカウンター)
SSH などその他の通信は数えない (= tx_bytes − links の合計)。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hfl_nodes import load_inventory  # noqa: E402

SYS = Path("/sys")
PROC = Path("/proc")
_BYTES = re.compile(r"bytes_(sent|received):(\d+)")


# ------------------------------------------------------------------ センサー
def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (OSError, TypeError, UnicodeDecodeError):
        # Jetson の cv0〜cv2-thermal などは EAGAIN を返し、Python では TypeError になる
        return None


def read_power(sys_root: Path = SYS) -> dict[str, float]:
    """INA3221 の各レール (mV × mA → mW)。"""
    power: dict[str, float] = {}
    for hwmon in sorted((sys_root / "class/hwmon").glob("hwmon*")):
        if _read(hwmon / "name") != "ina3221":
            continue
        for i in range(1, 5):
            label = _read(hwmon / f"in{i}_label")
            volts = _read(hwmon / f"in{i}_input")
            amps = _read(hwmon / f"curr{i}_input")
            if label and volts and amps and volts.isdigit() and amps.isdigit():
                power[label.replace(" ", "_")] = round(int(volts) * int(amps) / 1000.0, 1)
    return power


def read_temps(sys_root: Path = SYS) -> dict[str, float]:
    temps: dict[str, float] = {}
    for zone in sorted((sys_root / "class/thermal").glob("thermal_zone*")):
        kind, value = _read(zone / "type"), _read(zone / "temp")
        if kind and value and value.lstrip("-").isdigit():
            temps[kind.removesuffix("-thermal")] = round(int(value) / 1000.0, 1)
    return temps


def read_fan_rpm(sys_root: Path = SYS) -> int | None:
    for hwmon in (sys_root / "class/hwmon").glob("hwmon*"):
        if _read(hwmon / "name") == "pwm_tach":
            value = _read(hwmon / "rpm")
            return int(value) if value and value.isdigit() else None
    return None


def read_gpu_load(sys_root: Path = SYS) -> float | None:
    value = _read(sys_root / "devices/platform/gpu.0/load")  # ‰ (1000 = 100%)
    return round(int(value) / 10.0, 1) if value and value.isdigit() else None


def read_memory(proc_root: Path = PROC) -> tuple[int | None, int | None]:
    text = _read(proc_root / "meminfo") or ""
    values = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable") and rest.split():
            values[key] = int(rest.split()[0])
    if len(values) < 2:
        return None, None
    return (values["MemTotal"] - values["MemAvailable"]) // 1024, values["MemTotal"] // 1024


def read_load1(proc_root: Path = PROC) -> float | None:
    text = _read(proc_root / "loadavg")
    try:
        return float(text.split()[0]) if text else None
    except ValueError:
        return None


def read_boot_id(proc_root: Path = PROC) -> str | None:
    return _read(proc_root / "sys/kernel/random/boot_id")


def read_iface_counters(sys_root: Path = SYS) -> dict[str, list[int]]:
    """ループバック以外の各インターフェースの [rx_bytes, tx_bytes] (起動時からの累計)。"""
    counters: dict[str, list[int]] = {}
    for iface in sorted((sys_root / "class/net").glob("*")):
        if iface.name == "lo":
            continue
        rx, tx = _read(iface / "statistics/rx_bytes"), _read(iface / "statistics/tx_bytes")
        if rx and tx and rx.isdigit() and tx.isdigit():
            counters[iface.name] = [int(rx), int(tx)]
    return counters


def iface_for_ip(ip: str) -> str | None:
    """その IPv4 アドレスを持つインターフェース名 (ip コマンドから)。"""
    try:
        out = subprocess.run(
            ["ip", "-o", "-4", "addr", "show"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[3].split("/")[0] == ip:
            return parts[1]
    return None


# -------------------------------------------------------------- 接続ごとの通信量
def normalize_address(text: str) -> tuple[str, int] | None:
    """'[::ffff:192.168.10.202]:9001' → ('192.168.10.202', 9001)。"""
    host, _, port = text.rpartition(":")
    if not port.isdigit():
        return None
    host = host.strip("[]")
    host = host.split("%", 1)[0]
    if host.startswith("::ffff:"):
        host = host[len("::ffff:"):]
    return host, int(port)


def parse_ss(text: str) -> list[tuple[tuple[str, int], tuple[str, int], int, int]]:
    """`ss -tinH state established` の出力 → [(自分, 相手, bytes_sent, bytes_received)]。

    1接続 = 接続行 + 字下げされた詳細行。値が 0 の項目は ss が省略する。
    """
    connections = []
    current: list[Any] | None = None
    for line in text.splitlines():
        if not line.strip():
            continue
        if not line[0].isspace():
            parts = line.split()
            current = None
            if len(parts) >= 4:
                local, peer = normalize_address(parts[2]), normalize_address(parts[3])
                if local and peer:
                    current = [local, peer, 0, 0]
                    connections.append(current)
            continue
        if current is not None:
            for kind, value in _BYTES.findall(line):
                current[2 if kind == "sent" else 3] = int(value)
    return [tuple(c) for c in connections]


def is_loopback(host: str) -> bool:
    return host.startswith("127.") or host == "::1" or host == "localhost"


def read_connections() -> list[tuple[tuple[str, int], tuple[str, int], int, int]]:
    try:
        out = subprocess.run(
            ["ss", "-tinH", "state", "established"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_ss(out)


class LinkCounter:
    """学習の接続 (親・子) ごとの送受信量を、接続の入れ替わりをまたいで累計する。"""

    def __init__(
        self,
        parent: tuple[str, int] | None,
        own_port: int | None,
        totals: dict[str, dict[str, int]] | None = None,
    ) -> None:
        self.parent = parent
        self.own_port = own_port
        self.totals: dict[str, dict[str, int]] = totals or {}
        self.previous: dict[tuple, tuple[int, int]] | None = None  # None = まだ基準が無い

    def classify(self, local: tuple[str, int], peer: tuple[str, int]) -> str | None:
        if is_loopback(peer[0]) or is_loopback(local[0]):
            return None  # 同じ Jetson 内 (Edge と Internal Client) は数えない
        if self.parent is not None and peer == self.parent:
            return "parent"
        if self.own_port is not None and local[1] == self.own_port:
            return f"child:{peer[0]}"
        return None

    def update(self, connections) -> None:
        first = self.previous is None
        current: dict[tuple, tuple[int, int]] = {}
        for local, peer, sent, received in connections:
            link = self.classify(local, peer)
            if link is None:
                continue
            key = (local, peer)
            current[key] = (sent, received)
            if first:
                continue  # 記録開始時に開いていた接続は、ここを基準にする
            before = self.previous.get(key)
            if before is None:
                delta = (sent, received)  # 新しい接続は 0 から数える
            else:
                delta = (max(sent - before[0], 0), max(received - before[1], 0))
            total = self.totals.setdefault(link, {"sent": 0, "recv": 0})
            total["sent"] += delta[0]
            total["recv"] += delta[1]
        self.previous = current


# ------------------------------------------------------------------ 記録本体
def last_record(path: Path) -> dict[str, Any] | None:
    """ファイル末尾から、最後の完全な1行を読む (書き込み途中の行は飛ばす)。"""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(size - 65536, 0))
        tail = f.read()
    for line in reversed(tail.splitlines()):
        try:
            return json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return None


class Sampler:
    def __init__(
        self,
        *,
        run_id: str,
        node_id: str,
        node_num: int,
        node_ip: str,
        parent: tuple[str, int] | None,
        own_port: int | None,
        out_path: Path,
        pid_path: Path,
    ) -> None:
        self.run_id = run_id
        self.node_id = node_id
        self.node_num = node_num
        self.node_ip = node_ip
        self.out_path = out_path
        self.pid_path = pid_path
        self.iface = iface_for_ip(node_ip)
        previous = last_record(out_path)
        net = (previous or {}).get("net", {})
        self.tx_total = int(net.get("tx_bytes", 0))
        self.rx_total = int(net.get("rx_bytes", 0))
        self.previous_raw = net.get("iface_raw") if previous else None
        self.previous_boot = (previous or {}).get("boot_id")
        self.links = LinkCounter(parent, own_port, totals=net.get("links") or {})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(out_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        if out_path.stat().st_size and not _ends_with_newline(out_path):
            os.write(self._fd, b"\n")  # 電源断で途中まで書かれた行を閉じる

    def _update_iface(self, boot_id: str | None, raw: dict[str, list[int]]) -> list[int] | None:
        if self.iface is None:
            self.iface = iface_for_ip(self.node_ip)  # 起動直後でまだアドレスが無い場合
        current = raw.get(self.iface) if self.iface else None
        if current is None:
            return None
        if self.previous_raw is not None:
            same_boot = boot_id is not None and boot_id == self.previous_boot
            rx_before, tx_before = self.previous_raw
            if same_boot and current[0] >= rx_before and current[1] >= tx_before:
                self.rx_total += current[0] - rx_before
                self.tx_total += current[1] - tx_before
            else:
                # 再起動でカウンターが0から数え直した: 起動後の分をそのまま足す
                self.rx_total += current[0]
                self.tx_total += current[1]
        self.previous_raw = current
        self.previous_boot = boot_id
        return current

    def fl_alive(self) -> bool:
        pid = _read(self.pid_path)
        if not pid or not pid.isdigit():
            return False
        cmdline = _read(PROC / pid / "cmdline")
        # 再起動後に同じ PID が別のプロセスに使われていても誤判定しない
        return bool(cmdline) and "src.core" in cmdline

    def sample(self) -> dict[str, Any]:
        now = time.time()
        boot_id = read_boot_id()
        raw = read_iface_counters()
        iface_raw = self._update_iface(boot_id, raw)
        self.links.update(read_connections())
        mem_used, mem_total = read_memory()
        return {
            "ts": datetime.fromtimestamp(now).astimezone().isoformat(timespec="milliseconds"),
            "t": round(now, 3),
            "run_id": self.run_id,
            "node": self.node_id,
            "node_num": self.node_num,
            "source": "sampler",
            "boot_id": boot_id,
            "fl_alive": self.fl_alive(),
            "power_mw": read_power(),
            "temp_c": read_temps(),
            "gpu_load": read_gpu_load(),
            "fan_rpm": read_fan_rpm(),
            "mem_used_mb": mem_used,
            "mem_total_mb": mem_total,
            "load1": read_load1(),
            "net": {
                "iface": self.iface,
                "tx_bytes": self.tx_total,
                "rx_bytes": self.rx_total,
                "iface_raw": iface_raw,
                "links": self.links.totals,
                "raw": raw,
            },
        }

    def write(self, record: dict[str, Any]) -> None:
        os.write(self._fd, (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
        os.fsync(self._fd)


def _ends_with_newline(path: Path) -> bool:
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1) == b"\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HFL Jetson-side metrics sampler")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--node-num", type=int, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args(argv)

    inventory = load_inventory(args.profile)
    node = inventory.by_num(args.node_num)
    parent = None
    if node.parent_addr:
        host, port = node.parent_addr.rsplit(":", 1)
        parent = (host, int(port))
    run_dir = Path("logs") / args.run_id
    sampler = Sampler(
        run_id=args.run_id,
        node_id=node.id,
        node_num=node.num,
        node_ip=node.ip,
        parent=parent,
        own_port=node.port,
        out_path=run_dir / f"metrics-{node.num}.jsonl",
        pid_path=run_dir / f"hfl-{node.num}.pid",
    )
    stopping = False

    def handle_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    print(
        f"[sampler] {socket.gethostname()} node={node.num} iface={sampler.iface} "
        f"interval={args.interval}s -> {sampler.out_path}",
        flush=True,
    )
    while not stopping:
        try:
            sampler.write(sampler.sample())
        except Exception as exc:  # noqa: BLE001 - 記録の失敗で止まらない
            print(f"[sampler] sample failed: {exc}", file=sys.stderr, flush=True)
        # 時計の区切り (5 秒なら 0, 5, 10 … 秒) にそろえて待つ
        delay = args.interval - (time.time() % args.interval)
        end = time.monotonic() + delay
        while not stopping and time.monotonic() < end:
            time.sleep(min(0.5, end - time.monotonic()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
