#!/usr/bin/env python
"""
Jetson 7台の生存監視と記録のミラー。FL のコアには触れない。

各ノードへ一定間隔で SSH し、
  - 状態 (プロセス生存・待受ポート・メモリ・終了コード・時計ずれ) を調べ、
  - events-<NUM>.jsonl と hfl-<NUM>.log の「前回から増えた分だけ」を手元にコピーし (ミラー)、
  - 電力 (INA3221: VDD_IN ほか)・温度・ファン・メモリを metrics.jsonl に記録する。
グラフ用の時系列はミラーした全記録から作るので、いつ起動しても実験の最初から表示できる。
監視役を止めていた間の記録も、次に起動したときに取り戻す。

出力 (既定: logs/<run-id>/monitor/、HTTP でも配信する):
    mirror/<NUM>/events-<NUM>.jsonl, mirror/<NUM>/hfl-<NUM>.log   各ノードの記録のコピー
    mirror/<NUM>/metrics-<NUM>.jsonl  Jetson 側の記録 (電力・温度・通信量、5秒ごと)。
                                      台帳に sampler_interval_s がある profile で使う
    metrics.jsonl        Jetson 側の記録が無い run だけ、監視役が電力・温度を記録する
    clock_offsets.json   各ノードの時計ずれ (plot_run.py が補正に使う)
    status.json          監視ページ (index.html) が読む状態とグラフ用データ

Usage (リポジトリルートで実行):
    python scripts/hfl_monitor.py --profile config/jetson-5tier-longrun --run-id RUN \
        [--interval 15] [--port 8000] [--identity KEY]
    python scripts/hfl_monitor.py --profile ... --run-id RUN --once --table   # 端末に1回表示

--once は末尾だけ読む軽い確認用 (ミラー・metrics への記録はしない)。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hfl_activity import RoundTimeline, describe  # noqa: E402
from hfl_nodes import Inventory, Node, load_inventory, self_num  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
MIRROR_CHUNK_BYTES = 16 * 1024 * 1024  # 1回のポーリングで1ファイルから取り込む上限
ONCE_EVENTS_TAIL_BYTES = 64 * 1024     # --once で読む events の末尾
LOG_TAIL_BYTES = 4096
METRICS_TAIL_BYTES = 8192             # --once で読む Jetson 側の記録の末尾
NET_RATE_WINDOW_S = 60.0              # 送信速度は直近 1 分の平均
METRIC_CHART_INTERVAL_S = 60.0         # 監視ページの電力・温度グラフは1分に1点
STALE_FACTOR = 3.0        # 最新イベントからの経過が「想定ラウンド時間 × この倍率」を超えたら stale
DEFAULT_ROUND_S = 60.0    # ラウンド時間が未観測のときの想定
PROBE_TIMEOUT_S = 90.0
START_MARKER = b"===== [hfl] start"  # hfl.sh start がログに書く区切り
ANSI = re.compile(r"\x1b\[[0-9;]*m")

# 各ノードで実行する調査スクリプト。@KEY@ を置換して bash -s に流す。
# __FILE__ <名前> <offset> <n> <size> の行の直後に、ちょうど n バイトの中身が続く。
PROBE_SCRIPT = r"""
cd @DIR@ 2>/dev/null || { echo "__NOREPO__ 1"; echo "__END__"; exit 0; }
R=@RUN@; N=@NUM@
echo "__TIME__ $(date +%s.%N)"
pid=$(cat logs/$R/hfl-$N.pid 2>/dev/null)
echo "__PID__ ${pid:-}"
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
  echo "__ALIVE__ 1"
  echo "__UP__ $(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')"
  echo "__RSS__ $(ps -eo pgid=,rss= 2>/dev/null | awk -v g="$pid" '$1==g{s+=$2} END{print s+0}')"
else
  echo "__ALIVE__ 0"
fi
echo "__EXIT__ $(cat logs/$R/hfl-$N.exit 2>/dev/null)"
if [ -n "@PORT@" ]; then
  echo "__PORT__ $(ss -ltn 2>/dev/null | grep -c ':@PORT@ ')"
else
  echo "__PORT__ -"
fi
echo "__PROCS__ $(pgrep -fc '[s]rc\.core' 2>/dev/null || true)"
echo "__LOAD__ $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null)"
echo "__DISK__ $(df -Pm . 2>/dev/null | awk 'NR==2{print $4}')"
echo "__STOP__ $( [ -f logs/$R/STOP ] && echo 1 || echo 0 )"
spid=$(cat logs/$R/sampler-$N.pid 2>/dev/null)
if [ -n "$spid" ] && tr '\0' ' ' < /proc/$spid/cmdline 2>/dev/null | grep -q hfl_sampler.py; then
  echo "__SAMPLER__ 1"
else
  echo "__SAMPLER__ 0"
fi
echo "__MEM__ $(awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{print t+0, a+0}' /proc/meminfo)"
for h in /sys/class/hwmon/hwmon*; do
  case "$(cat "$h/name" 2>/dev/null)" in
    ina3221)
      for i in 1 2 3 4; do
        l=$(cat "$h/in${i}_label" 2>/dev/null) || continue
        v=$(cat "$h/in${i}_input" 2>/dev/null); c=$(cat "$h/curr${i}_input" 2>/dev/null)
        echo "__PWR__ $l $v $c"
      done ;;
    pwm_tach) echo "__FAN__ $(cat "$h/rpm" 2>/dev/null)" ;;
  esac
done
for z in /sys/class/thermal/thermal_zone*; do
  t=$(cat "$z/temp" 2>/dev/null) && echo "__TEMP__ $(cat "$z/type" 2>/dev/null) $t"
done
for spec in @FILES@; do
  name=${spec%:*}; off=${spec##*:}; f="logs/$R/$name"
  if [ ! -f "$f" ]; then echo "__FILE__ $name 0 0 -1"; continue; fi
  size=$(stat -c %s "$f")
  case "$off" in
    -*) k=${off#-}; if [ "$size" -gt "$k" ]; then off=$((size - k)); else off=0; fi ;;
  esac
  [ "$size" -lt "$off" ] && off=0
  n=$((size - off)); [ "$n" -gt @MAX@ ] && n=@MAX@
  echo "__FILE__ $name $off $n $size"
  if [ "$n" -gt 0 ]; then tail -c +$((off + 1)) "$f" | head -c "$n"; fi
done
echo "__END__"
"""


def _now_iso(t: float | None = None) -> str:
    moment = datetime.fromtimestamp(t) if t is not None else datetime.now()
    return moment.astimezone().isoformat(timespec="seconds")


def _int(text: str | None) -> int | None:
    if text is None:
        return None
    text = text.strip()
    return int(text) if text.lstrip("-").isdigit() else None


# --------------------------------------------------------------------------- 解析
@dataclass
class FileChunk:
    offset: int
    data: bytes
    size: int  # -1 = ファイルが無い


@dataclass
class ProbeResult:
    fields: dict[str, str] = field(default_factory=dict)
    power: list[str] = field(default_factory=list)
    temps: list[str] = field(default_factory=list)
    files: dict[str, FileChunk] = field(default_factory=dict)
    complete: bool = False


def parse_probe_output(raw: bytes) -> ProbeResult:
    """調査スクリプトの出力 (行とバイト列の混在) を解析する。"""
    result = ProbeResult()
    pos = 0
    while pos < len(raw):
        newline = raw.find(b"\n", pos)
        end = len(raw) if newline < 0 else newline
        text = raw[pos:end].decode("utf-8", errors="replace").rstrip("\r")
        pos = end + 1
        if not text.startswith("__"):
            continue
        key, _, value = text[2:].partition("__")
        value = value.strip()
        if key == "FILE":
            name, offset, length, size = value.split()
            length = int(length)
            result.files[name] = FileChunk(int(offset), raw[pos:pos + length], int(size))
            pos += length
        elif key == "PWR":
            result.power.append(value)
        elif key == "TEMP":
            result.temps.append(value)
        elif key == "END":
            result.complete = True
        else:
            result.fields[key] = value
    return result


def parse_metrics(probe: ProbeResult) -> dict[str, Any]:
    """電力 (mW)・温度 (℃)・ファン・メモリ・負荷を取り出す。"""
    power_mw: dict[str, float] = {}
    for line in probe.power:
        parts = line.split()
        if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
            label = "_".join(parts[:-2])
            power_mw[label] = round(int(parts[-2]) * int(parts[-1]) / 1000.0, 1)  # mV × mA
    temp_c: dict[str, float] = {}
    for line in probe.temps:
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            temp_c[parts[0].removesuffix("-thermal")] = round(int(parts[1]) / 1000.0, 1)
    mem = (probe.fields.get("MEM") or "").split()
    mem_total_mb = mem_used_mb = None
    if len(mem) == 2 and all(p.isdigit() for p in mem):
        mem_total_mb = round(int(mem[0]) / 1024)
        mem_used_mb = round((int(mem[0]) - int(mem[1])) / 1024)
    load = (probe.fields.get("LOAD") or "").split()
    return {
        "power_mw": power_mw,
        "temp_c": temp_c,
        "fan_rpm": _int(probe.fields.get("FAN")),
        "mem_used_mb": mem_used_mb,
        "mem_total_mb": mem_total_mb,
        "load1": float(load[0]) if load and _is_float(load[0]) else None,
    }


def _is_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def parse_event_lines(data: bytes) -> list[dict[str, Any]]:
    events = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def drop_partial_first_line(chunk: FileChunk) -> bytes:
    """ファイルの途中から読んだ場合、先頭の欠けた行を捨てる。"""
    if chunk.offset == 0:
        return chunk.data
    cut = chunk.data.find(b"\n")
    return b"" if cut < 0 else chunk.data[cut + 1:]


# ------------------------------------------------------------------------- ミラー
class MirrorStore:
    """各ノードの記録ファイルを、増えた分だけ追記してコピーする。

    コピー済みのバイト数 = 手元のファイルサイズ なので、監視役を再起動しても続きから取り込める。
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, num: int, name: str) -> Path:
        return self.root / str(num) / name

    def offset(self, num: int, name: str) -> int:
        path = self.path(num, name)
        return path.stat().st_size if path.exists() else 0

    def apply(self, num: int, name: str, chunk: FileChunk) -> bytes:
        """チャンクを追記し、追記した (改行で終わる) バイト列を返す。"""
        if chunk.size < 0:
            return b""
        path = self.path(num, name)
        local = self.offset(num, name)
        if chunk.offset == 0 < local:
            # 元のファイルが作り直された (調査スクリプトはサイズが減ったときだけ 0 から送る)。
            # 前のコピーは残して新しく始める
            try:
                path.rename(path.with_name(f"{path.name}.prev-{int(time.time())}"))
            except OSError:
                return b""  # Windows で配信中などで改名できない。次回やり直す
            local = 0
        if chunk.offset != local:
            return b""  # 想定外のずれ。次回のポーリングでやり直す
        cut = chunk.data.rfind(b"\n")
        if cut < 0:
            return b""  # 書き込み途中の行しか無い
        data = chunk.data[:cut + 1]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "ab") as f:
            f.write(data)
        return data


# --------------------------------------------------------------------- 状態の集計
@dataclass
class NodeState:
    last_round: int | None = None
    last_event: dict[str, Any] | None = None
    last_loss: float | None = None
    last_accuracy: float | None = None
    traceback_count: int = 0  # 最後に起動してからの Traceback 数
    log_tail: list[str] = field(default_factory=list)
    last_metric_chart_t: float = 0.0
    metrics: dict[str, Any] | None = None
    has_sampler: bool = False  # Jetson 側の記録 (metrics-<NUM>.jsonl) がある
    net_history: list[tuple[float, int, int]] = field(default_factory=list)  # (t, tx, rx)
    proc_events: dict[str, dict[str, Any]] = field(default_factory=dict)  # proc → 最新イベント


class RunState:
    """全ノードの記録からグラフ用の時系列を作る。"""

    def __init__(self) -> None:
        self.nodes: dict[int, NodeState] = {}
        self.global_series: list[dict[str, Any]] = []
        self.round_series: dict[str, list[list[float]]] = {}
        self.metric_series: dict[str, list[list[float | None]]] = {}
        self.resumes: list[dict[str, Any]] = []
        self.round_interval_s: float | None = None
        self.next_round_t: float | None = None
        self.timeline = RoundTimeline()

    def node(self, num: int) -> NodeState:
        return self.nodes.setdefault(num, NodeState())

    def consume_events(self, num: int, role: str, events: list[dict[str, Any]]) -> None:
        state = self.node(num)
        series = self.round_series.setdefault(str(num), [])
        for event in events:
            t = event.get("t")
            if not isinstance(t, (int, float)):
                continue
            rnd = event.get("global_round")
            resumed = bool(event.get("resumed"))
            # ラウンド進行は「ラウンド番号が変わったとき」だけ1点 (1ラウンド1点)
            if isinstance(rnd, int) and rnd > 0 and rnd != state.last_round and not resumed:
                series.append([round(float(t), 3), rnd])
                state.last_round = rnd
            if role == "global" and event.get("event") == "eval" and rnd and not resumed:
                self.global_series.append({
                    "t": round(float(t), 3), "round": rnd,
                    "loss": event.get("loss"), "accuracy": event.get("accuracy"),
                    "round_duration_s": event.get("round_duration_s"),
                })
            if role == "global" and event.get("event") == "resume":
                self.resumes.append({
                    "t": round(float(t), 3), "round": rnd,
                    "downtime_s": event.get("downtime_s"),
                })
            if "accuracy" in event and "loss" in event and not resumed:
                state.last_loss = event["loss"]
                state.last_accuracy = event["accuracy"]
            if role == "global":
                kind = event.get("event")
                if kind == "start":
                    self.round_interval_s = event.get("round_interval_s") or None
                if kind == "wait":
                    self.next_round_t = event.get("next_round_t")
                elif kind in ("round_start", "stop", "error", "start"):
                    self.next_round_t = None
            if event.get("proc"):
                state.proc_events[event["proc"]] = event
            self.timeline.consume(num, event)
            state.last_event = event

    def consume_log(self, num: int, data: bytes) -> None:
        state = self.node(num)
        marker = data.rfind(START_MARKER)
        if marker >= 0:
            state.traceback_count = 0  # 起動し直したので数え直す
            data_since_start = data[marker:]
        else:
            data_since_start = data
        state.traceback_count += data_since_start.count(b"Traceback (most recent call last)")
        lines = [
            ANSI.sub("", line).rstrip()
            for line in data.decode("utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        if lines:
            state.log_tail = (state.log_tail + lines)[-3:]

    def consume_metrics(self, num: int, t: float, metrics: dict[str, Any]) -> None:
        state = self.node(num)
        state.metrics = metrics
        net = metrics.get("net") or {}
        if isinstance(net.get("tx_bytes"), int):
            state.net_history.append((t, net["tx_bytes"], net.get("rx_bytes") or 0))
            cutoff = t - 3 * NET_RATE_WINDOW_S
            while state.net_history and state.net_history[0][0] < cutoff:
                state.net_history.pop(0)
        if t - state.last_metric_chart_t >= METRIC_CHART_INTERVAL_S:
            power = (metrics.get("power_mw") or {}).get("VDD_IN")
            temp = (metrics.get("temp_c") or {}).get("tj")
            self.metric_series.setdefault(str(num), []).append([
                round(t, 1),
                round(power / 1000.0, 2) if power is not None else None,
                temp,
                net.get("tx_bytes"),
            ])
            state.last_metric_chart_t = t

    def consume_sampler_records(self, num: int, records: list[dict[str, Any]]) -> None:
        """Jetson 側の記録 (5秒ごと)。"""
        if records:
            self.node(num).has_sampler = True
        for record in records:
            if isinstance(record.get("t"), (int, float)):
                self.consume_metrics(num, float(record["t"]), record)

    def net_rates(self, num: int) -> tuple[float | None, float | None]:
        """直近 1 分の送信・受信速度 (bytes/s)。"""
        history = self.node(num).net_history
        if len(history) < 2:
            return None, None
        t1, tx1, rx1 = history[-1]
        older = [h for h in history if t1 - h[0] >= NET_RATE_WINDOW_S]
        t0, tx0, rx0 = older[-1] if older else history[0]
        if t1 - t0 < 10:
            return None, None
        return max(tx1 - tx0, 0) / (t1 - t0), max(rx1 - rx0, 0) / (t1 - t0)

    def expected_round_s(self) -> float:
        if self.round_interval_s:
            return max(float(self.round_interval_s), self._median_round_s() or 0.0)
        return self._median_round_s() or DEFAULT_ROUND_S

    def _median_round_s(self) -> float | None:
        if len(self.global_series) >= 3:
            recent = self.global_series[-10:]
            spans = [b["t"] - a["t"] for a, b in zip(recent, recent[1:]) if b["t"] > a["t"]]
            if spans:
                median = sorted(spans)[len(spans) // 2]
                return max(min(median, 3600.0), 5.0)
        return None


# ------------------------------------------------------------------------ 監視本体
Runner = Callable[[Node, str], tuple[bytes, str, bool]]


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # アクセスログは出さない


class Monitor:
    def __init__(
        self,
        inventory: Inventory,
        run_id: str,
        out_dir: Path,
        *,
        interval: float,
        identity: str | None,
        mirror: bool = True,
        runner: Runner | None = None,
    ) -> None:
        self.inventory = inventory
        self.run_id = run_id
        self.out_dir = out_dir
        self.interval = interval
        self.identity = identity
        self.mirror_enabled = mirror
        self.runner = runner or self._run_ssh
        self.self_num = self_num(inventory)
        self.started_at = time.time()
        self.state = RunState()
        self.store = MirrorStore(out_dir / "mirror")
        self.metrics_path = out_dir / "metrics.jsonl"
        self.pending = False  # まだ取り込み切れていないデータがある
        if mirror:
            self._rebuild()

    # ---------------------------------------------------------------- 起動時
    def _rebuild(self) -> None:
        """手元のミラーと metrics.jsonl から時系列を作り直す (監視役の再起動に備える)。"""
        for node in self.inventory.nodes:
            events_path = self.store.path(node.num, f"events-{node.num}.jsonl")
            if events_path.exists():
                self.state.consume_events(
                    node.num, node.role, parse_event_lines(events_path.read_bytes())
                )
            log_path = self.store.path(node.num, f"hfl-{node.num}.log")
            if log_path.exists():
                self.state.consume_log(node.num, log_path.read_bytes())
            metrics_path = self.store.path(node.num, f"metrics-{node.num}.jsonl")
            if metrics_path.exists():
                self.state.consume_sampler_records(
                    node.num, parse_event_lines(metrics_path.read_bytes())
                )
        if self.metrics_path.exists():
            for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not self.state.node(record["node_num"]).has_sampler:
                    self.state.consume_metrics(record["node_num"], record["t"], record)

    # ------------------------------------------------------------------ 調査
    def _ssh_base(self) -> list[str]:
        cmd = [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            "-o", "StrictHostKeyChecking=accept-new",
        ]
        if self.identity:
            cmd += ["-i", self.identity]
        return cmd

    def _run_ssh(self, node: Node, script: str) -> tuple[bytes, str, bool]:
        is_self = self.self_num == node.num
        if is_self:
            cmd = ["bash", "-s"]
        else:
            cmd = self._ssh_base() + [f"{node.user}@{node.ip}", "bash -s"]
        try:
            # bytes で渡す: Windows の text モードは \n を \r\n に変えて remote の bash を壊し、
            # 既定の cp932 では UTF-8 の stderr を落とす
            proc = subprocess.run(
                cmd, input=script.encode("utf-8"), capture_output=True,
                timeout=PROBE_TIMEOUT_S, cwd=str(REPO_ROOT) if is_self else None,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return b"", str(exc), False
        error = proc.stderr.decode("utf-8", errors="replace").strip()
        return proc.stdout, error, proc.returncode == 0

    def _file_specs(self, node: Node) -> str:
        names = [f"events-{node.num}.jsonl", f"hfl-{node.num}.log", f"metrics-{node.num}.jsonl"]
        if not self.mirror_enabled:
            tails = (ONCE_EVENTS_TAIL_BYTES, LOG_TAIL_BYTES, METRICS_TAIL_BYTES)
            return " ".join(f"{name}:-{tail}" for name, tail in zip(names, tails))
        return " ".join(f"{name}:{self.store.offset(node.num, name)}" for name in names)

    def _script(self, node: Node) -> str:
        is_self = self.self_num == node.num
        return (
            PROBE_SCRIPT.replace("@DIR@", "." if is_self else self.inventory.remote_dir)
            .replace("@RUN@", self.run_id)
            .replace("@NUM@", str(node.num))
            .replace("@PORT@", str(node.port) if node.port else "")
            .replace("@FILES@", self._file_specs(node))
            .replace("@MAX@", str(MIRROR_CHUNK_BYTES))
        )

    def probe(self, node: Node) -> tuple[Node, ProbeResult, str, bool, float, float]:
        t0 = time.time()
        raw, error, ok = self.runner(node, self._script(node))
        t1 = time.time()
        probe = parse_probe_output(raw)
        ok = ok and "TIME" in probe.fields and probe.complete
        return node, probe, error, ok, (t0 + t1) / 2, t1 - t0

    # ------------------------------------------------------------------ 反映
    def _absorb(self, node: Node, probe: ProbeResult) -> None:
        events_name, log_name = f"events-{node.num}.jsonl", f"hfl-{node.num}.log"
        for name, chunk in probe.files.items():
            if self.mirror_enabled:
                data = self.store.apply(node.num, name, chunk)
                if chunk.size >= 0 and chunk.offset + len(chunk.data) < chunk.size:
                    self.pending = True
            else:
                data = drop_partial_first_line(chunk)
            if name == events_name:
                self.state.consume_events(node.num, node.role, parse_event_lines(data))
            elif name == log_name:
                self.state.consume_log(node.num, data)
            elif name == f"metrics-{node.num}.jsonl":
                self.state.consume_sampler_records(node.num, parse_event_lines(data))

    def _record_metrics(self, node: Node, t: float, metrics: dict[str, Any], alive: bool,
                        rss_mb: int | None) -> None:
        record = {
            "ts": _now_iso(t), "t": round(t, 3), "run_id": self.run_id,
            "node": node.id, "node_num": node.num, "alive": alive, "rss_mb": rss_mb,
            **metrics,
        }
        if self.state.node(node.num).has_sampler:
            return  # Jetson 側の記録 (5秒ごと) を使う
        self.state.consume_metrics(node.num, t, record)
        if self.mirror_enabled:
            with open(self.metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _node_status(
        self, node: Node, probe: ProbeResult, error: str, ok: bool, local_mid: float, rtt: float
    ) -> dict[str, Any]:
        state = self.state.node(node.num)
        status: dict[str, Any] = {
            "num": node.num, "ip": node.ip, "id": node.id, "role": node.role,
            "port": node.port, "parent": node.parent_num,
            "reachable": ok, "error": None if ok else (error or "no response")[:200],
            "probe_rtt_s": round(rtt, 3),
            "alive": False, "pid": None, "uptime_s": None, "rss_mb": None,
            "exit_code": None, "port_listening": None, "procs": None, "loadavg": None,
            "disk_free_mb": None, "stop_requested": False, "clock_offset_s": None,
            "traceback_count": state.traceback_count,
            "last_event": None, "last_event_age_s": None,
            "current_round": state.last_round,
            "last_loss": state.last_loss, "last_accuracy": state.last_accuracy,
            "metrics": state.metrics, "log_tail": state.log_tail,
            "activity": describe(node.role, state.proc_events),
            "sampler_alive": None,
            "net": self._net_status(node.num),
        }
        if state.last_event is not None:
            status["last_event"] = {
                k: state.last_event.get(k)
                for k in ("ts", "t", "proc", "event", "global_round", "reason")
            }
        if not ok:
            status["state"] = "unreachable"
            return status
        fields = probe.fields
        if fields.get("NOREPO"):
            status.update(state="no_repo", error="remote dir not found")
            return status

        remote_t = float(fields["TIME"])
        status["clock_offset_s"] = round(remote_t - local_mid, 3)
        status["alive"] = fields.get("ALIVE") == "1"
        status["pid"] = _int(fields.get("PID"))
        status["uptime_s"] = _int(fields.get("UP"))
        rss_kb = _int(fields.get("RSS"))
        status["rss_mb"] = round(rss_kb / 1024) if rss_kb is not None else None
        status["exit_code"] = _int(fields.get("EXIT"))
        port = fields.get("PORT", "-")
        status["port_listening"] = None if port == "-" else port not in ("", "0")
        status["procs"] = _int(fields.get("PROCS")) or 0
        status["loadavg"] = fields.get("LOAD") or None
        status["disk_free_mb"] = _int(fields.get("DISK"))
        status["stop_requested"] = fields.get("STOP") == "1"
        if fields.get("SAMPLER") in ("0", "1"):
            status["sampler_alive"] = fields["SAMPLER"] == "1"
        last_t = (state.last_event or {}).get("t")
        if isinstance(last_t, (int, float)):
            status["last_event_age_s"] = round(remote_t - float(last_t), 1)
        metrics = parse_metrics(probe)
        self._record_metrics(node, local_mid, metrics, status["alive"], status["rss_mb"])
        status["metrics"] = state.metrics
        status["net"] = self._net_status(node.num)
        status["state"] = self._state(status)
        return status

    def _net_status(self, num: int) -> dict[str, Any] | None:
        net = ((self.state.node(num).metrics or {}).get("net")) or None
        if not net or not isinstance(net.get("tx_bytes"), int):
            return None
        tx_rate, rx_rate = self.state.net_rates(num)
        return {
            "tx_bytes": net["tx_bytes"], "rx_bytes": net.get("rx_bytes"),
            "tx_rate_bps": round(tx_rate, 1) if tx_rate is not None else None,
            "rx_rate_bps": round(rx_rate, 1) if rx_rate is not None else None,
            "links": net.get("links") or {},
            "iface": net.get("iface"),
        }

    def _state(self, s: dict[str, Any]) -> str:
        last = s.get("last_event") or {}
        if s["alive"]:
            if s["traceback_count"]:
                return "error"
            age = s.get("last_event_age_s")
            stale_after = STALE_FACTOR * self.state.expected_round_s()
            waiting = (
                self.state.next_round_t is not None
                and time.time() < self.state.next_round_t + DEFAULT_ROUND_S
            )
            if (age is not None and age > stale_after and last.get("event") != "start"
                    and not waiting):
                return "stale"
            return "running"
        if last.get("event") == "stop" and s.get("exit_code") in (0, None):
            return "finished"
        failed_exit = s.get("exit_code") not in (None, 0)
        if failed_exit or last.get("event") == "error" or s["traceback_count"]:
            return "failed"
        if s.get("pid") is None and not last:
            return "not_started"
        return "stopped"

    def _alerts(self, nodes: list[dict[str, Any]]) -> list[str]:
        alerts = []
        for n in nodes:
            label = f"{n['num']} {n['id']}"
            if n["state"] == "unreachable":
                alerts.append(f"{label}: 到達不能 ({n['error']})")
            elif n["state"] in ("failed", "error"):
                alerts.append(
                    f"{label}: 異常終了または Traceback "
                    f"(exit={n['exit_code']}, tb={n['traceback_count']})"
                )
            elif n["state"] == "stale":
                age = n["last_event_age_s"]
                alerts.append(f"{label}: イベントが {age:.0f}s 更新されていません")
            elif n["state"] == "stopped":
                alerts.append(f"{label}: プロセスが終了しています (exit なし = 強制終了か電源断)")
            if n["alive"] and n["port_listening"] is False:
                alerts.append(f"{label}: ポート {n['port']} が待受していません")
            if n["clock_offset_s"] is not None and abs(n["clock_offset_s"]) > 2:
                alerts.append(f"{label}: 時計ずれ {n['clock_offset_s']:+.1f}s")
            temp = ((n.get("metrics") or {}).get("temp_c") or {}).get("tj")
            if n["reachable"] and temp is not None and temp >= 85:
                alerts.append(f"{label}: 高温 {temp:.0f}℃")
        return alerts

    def poll_once(self) -> dict[str, Any]:
        self.pending = False
        with ThreadPoolExecutor(max_workers=len(self.inventory.nodes)) as pool:
            results = list(pool.map(self.probe, self.inventory.nodes))
        nodes = []
        for node, probe, error, ok, local_mid, rtt in results:
            if ok:
                self._absorb(node, probe)
            nodes.append(self._node_status(node, probe, error, ok, local_mid, rtt))
        state = self.state
        global_node = next((n for n in nodes if n["role"] == "global"), None)
        latest = state.global_series[-1] if state.global_series else None
        first = state.global_series[0] if state.global_series else None
        downtime = sum(r.get("downtime_s") or 0 for r in state.resumes)
        status = {
            "generated_at": _now_iso(),
            "t": round(time.time(), 3),
            "run_id": self.run_id,
            "profile": self.inventory.profile,
            "interval_s": self.interval,
            "monitor_host": socket.gethostname(),
            "monitor_started_t": self.started_at,
            "mirror": self.mirror_enabled,
            "catching_up": self.pending,
            "summary": {
                "alive": sum(1 for n in nodes if n["alive"]),
                "total": len(nodes),
                "global_state": global_node["state"] if global_node else None,
                "global_round": latest["round"] if latest else None,
                "last_accuracy": latest["accuracy"] if latest else None,
                "last_loss": latest["loss"] if latest else None,
                "expected_round_s": round(state.expected_round_s(), 1),
                "global_uptime_s": global_node["uptime_s"] if global_node else None,
                "first_eval_t": first["t"] if first else None,
                "resumes": len(state.resumes),
                "downtime_s": round(downtime, 1),
                "total_power_w": _total_power_w(nodes),
                "total_tx_bytes": _total(nodes, "tx_bytes"),
                "total_rx_bytes": _total(nodes, "rx_bytes"),
                "total_tx_rate_bps": _total(nodes, "tx_rate_bps"),
                "round_interval_s": state.round_interval_s,
                "next_round_t": state.next_round_t,
            },
            "timeline": state.timeline.latest(
                {n["num"]: n for n in nodes},
                {n["num"]: n["clock_offset_s"] for n in nodes if n["clock_offset_s"] is not None},
            ),
            "alerts": self._alerts(nodes),
            "nodes": nodes,
            "global_series": state.global_series,
            "round_series": state.round_series,
            "metric_series": state.metric_series,
            "resumes": state.resumes,
        }
        if self.mirror_enabled:
            self._write(status, nodes)
        return status

    def _write(self, status: dict[str, Any], nodes: list[dict[str, Any]]) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        offsets = {
            str(n["num"]): n["clock_offset_s"] for n in nodes if n["clock_offset_s"] is not None
        }
        if offsets:
            _replace_text(self.out_dir / "clock_offsets.json", json.dumps(offsets, indent=1))
        _replace_text(self.out_dir / "status.json", json.dumps(status, ensure_ascii=False))


def _total(nodes: list[dict[str, Any]], key: str) -> float | None:
    """全ノードの通信量の合計 (Jetson 側の記録があるノードだけ)。"""
    values = [(n.get("net") or {}).get(key) for n in nodes]
    values = [v for v in values if v is not None]
    return round(sum(values), 1) if values else None


def _total_power_w(nodes: list[dict[str, Any]]) -> float | None:
    values = [
        ((n.get("metrics") or {}).get("power_mw") or {}).get("VDD_IN")
        for n in nodes if n["reachable"]
    ]
    values = [v for v in values if v is not None]
    return round(sum(values) / 1000.0, 2) if values else None


def _replace_text(path: Path, text: str) -> None:
    """一時ファイル経由で置き換える。Windows では配信中のファイルを置き換えられないことが
    あるので、少し待って再試行する。"""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.1 * (attempt + 1))
    os.replace(tmp, path)


# -------------------------------------------------------------------------- 表示
def _cell(value: Any, fmt: str = "{}", dash: str = "-") -> str:
    return dash if value is None else fmt.format(value)


def _uptime(seconds: int | None) -> str:
    if seconds is None:
        return "-"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _port(n: dict[str, Any]) -> str:
    if n["port_listening"] is None:
        return "-"
    return "open" if n["port_listening"] else "CLOSED"


def _metric(n: dict[str, Any], group: str, key: str) -> Any:
    return ((n.get("metrics") or {}).get(group) or {}).get(key)


def _tx(n: dict[str, Any]) -> str:
    value = (n.get("net") or {}).get("tx_bytes")
    return "-" if value is None else _bytes(value)


def _bytes(value: float) -> str:
    for unit, size in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if value >= size:
            return f"{value / size:.1f}{unit}"
    return f"{value:.0f}B"


def _power_w(n: dict[str, Any]) -> float | None:
    milliwatts = _metric(n, "power_mw", "VDD_IN")
    return None if milliwatts is None else milliwatts / 1000.0


TABLE_COLUMNS: list[tuple[str, int, Callable[[dict[str, Any]], str]]] = [
    ("NUM", 4, lambda n: str(n["num"])),
    ("ID", 9, lambda n: n["id"]),
    ("ROLE", 6, lambda n: n["role"]),
    ("STATE", 11, lambda n: n["state"]),
    ("PID", 7, lambda n: _cell(n["pid"])),
    ("UP", 8, lambda n: _uptime(n["uptime_s"])),
    ("RSS", 7, lambda n: _cell(n["rss_mb"], "{}M")),
    ("PORT", 6, _port),
    ("ROUND", 7, lambda n: _cell(n["current_round"])),
    ("LOSS", 7, lambda n: _cell(n["last_loss"], "{:.3f}")),
    ("ACC", 6, lambda n: _cell(n["last_accuracy"], "{:.3f}")),
    ("AGE", 6, lambda n: _cell(n["last_event_age_s"], "{:.0f}s")),
    ("POWER", 6, lambda n: _cell(_power_w(n), "{:.1f}W")),
    ("TEMP", 6, lambda n: _cell(_metric(n, "temp_c", "tj"), "{:.0f}C")),
    ("TX", 9, lambda n: _tx(n)),
    ("CLOCK", 7, lambda n: _cell(n["clock_offset_s"], "{:+.1f}s")),
    ("TB", 3, lambda n: str(n["traceback_count"])),
]


def format_table(status: dict[str, Any]) -> str:
    header = " ".join(f"{name:<{width}}" for name, width, _ in TABLE_COLUMNS)
    lines = [header, "-" * len(header)]
    for n in status["nodes"]:
        lines.append(" ".join(f"{fn(n):<{width}}" for _, width, fn in TABLE_COLUMNS))
    s = status["summary"]
    lines.append("")
    lines.append(
        f"run_id={status['run_id']} alive={s['alive']}/{s['total']} "
        f"global_round={s['global_round']} acc={s['last_accuracy']} loss={s['last_loss']} "
        f"~{s['expected_round_s']}s/round power={s['total_power_w']}W ({status['generated_at']})"
    )
    for alert in status["alerts"]:
        lines.append(f"  ! {alert}")
    return "\n".join(lines)


def serve(out_dir: Path, port: int) -> ThreadingHTTPServer:
    handler = partial(NoCacheHandler, directory=str(out_dir))
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    thread = threading.Thread(target=server.serve_forever, name="http", daemon=True)
    thread.start()
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HFL Jetson monitor")
    parser.add_argument("--profile", default="config/jetson-5tier-longrun")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--out-dir", default=None, help="既定: logs/<run-id>/monitor")
    parser.add_argument("--identity", default=None, help="SSH 秘密鍵")
    parser.add_argument("--once", action="store_true", help="末尾だけ読んで1回表示 (記録しない)")
    parser.add_argument("--table", action="store_true", help="端末に表で表示")
    parser.add_argument("--no-serve", action="store_true", help="HTTP 配信しない")
    args = parser.parse_args(argv)

    inventory = load_inventory(args.profile)
    run_id = args.run_id
    if not run_id:
        marker = REPO_ROOT / "logs" / "current_run"
        if not marker.exists():
            print("--run-id を指定してください (logs/current_run がありません)", file=sys.stderr)
            return 1
        run_id = marker.read_text(encoding="utf-8").strip()
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "logs" / run_id / "monitor"

    if args.once:
        monitor = Monitor(
            inventory, run_id, out_dir, interval=args.interval, identity=args.identity,
            mirror=False,
        )
        status = monitor.poll_once()
        if args.table:
            print(format_table(status))
        else:
            print(json.dumps(status, ensure_ascii=False, indent=1))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    # 監視画面 (index.html・viz.js など) を一式コピーして配信する
    shutil.copytree(Path(__file__).resolve().parent / "monitor", out_dir, dirs_exist_ok=True)
    print(f"[monitor] run_id={run_id} loading mirror from {out_dir} ...", flush=True)
    monitor = Monitor(inventory, run_id, out_dir, interval=args.interval, identity=args.identity)
    port = args.port or inventory.monitor_port
    server = None if args.no_serve else serve(out_dir, port)
    url = "" if server is None else f" http://localhost:{port}/"
    print(f"[monitor] interval={args.interval}s out={out_dir}{url}  (Ctrl+C で終了)", flush=True)
    try:
        while True:
            started = time.monotonic()
            status = monitor.poll_once()
            s = status["summary"]
            print(
                f"[monitor] {status['generated_at']} alive={s['alive']}/{s['total']} "
                f"round={s['global_round']} acc={s['last_accuracy']} power={s['total_power_w']}W "
                f"alerts={len(status['alerts'])}"
                + (" (catching up...)" if monitor.pending else ""),
                flush=True,
            )
            if args.table:
                print(format_table(status), flush=True)
            if monitor.pending:
                continue  # 取り込み切れていない記録があれば待たずに続ける
            time.sleep(max(0.0, args.interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
