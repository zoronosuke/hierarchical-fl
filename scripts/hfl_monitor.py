#!/usr/bin/env python
"""
Jetson 7台の生存監視。各ノードを SSH でポーリングし、status.json を書き、
監視ページ (scripts/monitor/index.html) を HTTP で配信する。FL のコアには触れない。

Usage (リポジトリルートで実行。通常は scripts/hfl.sh monitor start から起動する):
    python scripts/hfl_monitor.py --profile config/jetson-5tier-longrun --run-id RUN \
        [--interval 15] [--port 8000] [--identity KEY]
    python scripts/hfl_monitor.py --profile ... --run-id RUN --once --table   # 端末に1回表示

取得するもの (ノードごと): 到達可否 / プロセス生存 (pid ファイル) / 待受ポート /
RSS / 稼働時間 / 終了コード / 時計ずれ / events-<num>.jsonl の末尾 / ログ末尾。
Global の eval イベントは蓄積して精度・loss の時系列にする。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hfl_nodes import Inventory, Node, load_inventory, self_num  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_SERIES_POINTS = 20000
STALE_FACTOR = 3.0        # 最新イベントからの経過が「想定ラウンド時間 × この倍率」を超えたら stale
DEFAULT_ROUND_S = 60.0    # ラウンド時間が未観測のときの想定

# 各ノードで実行する調査スクリプト。@KEY@ を置換して bash -s に流す。
PROBE_SCRIPT = r"""
cd @DIR@ 2>/dev/null || { echo "__NOREPO__ 1"; exit 0; }
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
tb=$(grep -c Traceback logs/$R/hfl-$N.log 2>/dev/null); echo "__TB__ ${tb:-0}"
echo "__STOP__ $( [ -f logs/$R/STOP ] && echo 1 || echo 0 )"
echo "__EVENTS__"
tail -n @TAIL@ logs/$R/events-$N.jsonl 2>/dev/null
echo "__LOG__"
tail -n 3 logs/$R/hfl-$N.log 2>/dev/null | cut -c1-300
"""


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _int(text: str | None) -> int | None:
    if text is None:
        return None
    text = text.strip()
    return int(text) if text.lstrip("-").isdigit() else None


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
        tail: int,
    ) -> None:
        self.inventory = inventory
        self.run_id = run_id
        self.out_dir = out_dir
        self.interval = interval
        self.identity = identity
        self.tail = tail
        self.self_num = self_num(inventory)
        self.started_at = time.time()
        # 蓄積する時系列 (再起動時は前回の status.json から復元)
        self.global_series: list[dict[str, Any]] = []
        self.round_series: dict[str, list[list[float]]] = {}
        self.last_seen_t: dict[str, float] = {}
        self._restore()

    # ------------------------------------------------------------------ probe
    def _ssh_base(self) -> list[str]:
        cmd = [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            "-o", "StrictHostKeyChecking=accept-new",
        ]
        if self.identity:
            cmd += ["-i", self.identity]
        return cmd

    def probe(self, node: Node) -> dict[str, Any]:
        is_self = self.self_num == node.num
        script = (
            PROBE_SCRIPT.replace("@DIR@", "." if is_self else self.inventory.remote_dir)
            .replace("@RUN@", self.run_id)
            .replace("@NUM@", str(node.num))
            .replace("@PORT@", str(node.port) if node.port else "")
            .replace("@TAIL@", str(self.tail))
        )
        if is_self:
            cmd = ["bash", "-s"]
        else:
            cmd = self._ssh_base() + [f"{node.user}@{node.ip}", "bash -s"]
        t0 = time.time()
        try:
            # bytes で渡す: Windows の text モードは \n を \r\n に変えて remote の bash を壊し、
            # 既定の cp932 では UTF-8 の stderr を落とす
            proc = subprocess.run(
                cmd, input=script.encode("utf-8"), capture_output=True, timeout=25,
                cwd=str(REPO_ROOT) if is_self else None,
            )
            output = proc.stdout.decode("utf-8", errors="replace")
            error = proc.stderr.decode("utf-8", errors="replace").strip()
            ok = proc.returncode == 0 and "__TIME__" in output
        except (OSError, subprocess.SubprocessError) as exc:
            output, error, ok = "", str(exc), False
        t1 = time.time()
        return self._parse(node, output, ok, error, (t0 + t1) / 2, t1 - t0)

    @staticmethod
    def _split_output(output: str) -> tuple[dict[str, str], list[str], list[str]]:
        fields: dict[str, str] = {}
        events: list[str] = []
        log_tail: list[str] = []
        section = None
        for line in output.splitlines():
            if line.startswith("__EVENTS__"):
                section = "events"
            elif line.startswith("__LOG__"):
                section = "log"
            elif section == "events":
                events.append(line)
            elif section == "log":
                log_tail.append(line)
            elif line.startswith("__") and "__ " in line:
                key, _, value = line[2:].partition("__ ")
                fields[key] = value.strip()
        return fields, events, log_tail

    def _parse(
        self, node: Node, output: str, ok: bool, error: str, local_mid: float, rtt: float
    ) -> dict[str, Any]:
        status: dict[str, Any] = {
            "num": node.num, "ip": node.ip, "id": node.id, "role": node.role,
            "port": node.port, "parent": node.parent_num,
            "reachable": ok, "error": None if ok else (error or "no response")[:200],
            "probe_rtt_s": round(rtt, 3),
            "alive": False, "pid": None, "uptime_s": None, "rss_mb": None,
            "exit_code": None, "port_listening": None, "procs": None, "loadavg": None,
            "disk_free_mb": None, "traceback_count": 0, "stop_requested": False,
            "clock_offset_s": None, "last_event": None, "last_event_age_s": None,
            "current_round": None, "last_loss": None, "last_accuracy": None,
            "events": [], "log_tail": [],
        }
        if not ok:
            status["state"] = "unreachable"
            return status

        fields, events, log_tail = self._split_output(output)
        if fields.get("NOREPO"):
            status.update(state="no_repo", error="remote dir not found")
            return status

        remote_t = float(fields["TIME"]) if fields.get("TIME") else None
        status["clock_offset_s"] = round(remote_t - local_mid, 3) if remote_t else None
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
        status["traceback_count"] = _int(fields.get("TB")) or 0
        status["stop_requested"] = fields.get("STOP") == "1"
        status["log_tail"] = log_tail

        parsed: list[dict[str, Any]] = []
        for line in events:
            line = line.strip()
            if not line:
                continue
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        self._absorb_events(node, parsed)
        status["events"] = parsed[-12:]
        if parsed:
            last = parsed[-1]
            status["last_event"] = {
                k: last.get(k) for k in ("ts", "t", "proc", "event", "global_round", "reason")
            }
            if remote_t is not None and isinstance(last.get("t"), (int, float)):
                status["last_event_age_s"] = round(remote_t - float(last["t"]), 1)
            rounds = [
                e.get("global_round") for e in parsed if isinstance(e.get("global_round"), int)
            ]
            status["current_round"] = max(rounds) if rounds else None
            for event in reversed(parsed):
                if "accuracy" in event and "loss" in event:
                    status["last_loss"] = event["loss"]
                    status["last_accuracy"] = event["accuracy"]
                    break
        status["state"] = self._state(status)
        return status

    def _state(self, s: dict[str, Any]) -> str:
        last = s.get("last_event") or {}
        if s["alive"]:
            if s["traceback_count"]:
                return "error"
            age = s.get("last_event_age_s")
            stale_after = STALE_FACTOR * self._expected_round_s()
            if age is not None and age > stale_after and last.get("event") != "start":
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

    def _expected_round_s(self) -> float:
        if len(self.global_series) >= 3:
            recent = self.global_series[-10:]
            spans = [b["t"] - a["t"] for a, b in zip(recent, recent[1:]) if b["t"] > a["t"]]
            if spans:
                median = sorted(spans)[len(spans) // 2]
                return max(min(median, 3600.0), 5.0)
        return DEFAULT_ROUND_S

    def _absorb_events(self, node: Node, events: list[dict[str, Any]]) -> None:
        key = str(node.num)
        series = self.round_series.setdefault(key, [])
        last_t = self.last_seen_t.get(key, 0.0)
        for event in events:
            t = event.get("t")
            rnd = event.get("global_round")
            if not isinstance(t, (int, float)) or t <= last_t:
                continue
            if isinstance(rnd, int) and rnd > 0:
                series.append([round(float(t), 3), rnd])
            if node.role == "global" and event.get("event") == "eval" and rnd:
                self.global_series.append({
                    "t": round(float(t), 3), "round": rnd,
                    "loss": event.get("loss"), "accuracy": event.get("accuracy"),
                    "round_duration_s": event.get("round_duration_s"),
                })
            last_t = max(last_t, float(t))
        self.last_seen_t[key] = last_t
        del series[:-MAX_SERIES_POINTS]
        del self.global_series[:-MAX_SERIES_POINTS]

    # ----------------------------------------------------------------- output
    def _restore(self) -> None:
        path = self.out_dir / "status.json"
        if not path.exists():
            return
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if previous.get("run_id") != self.run_id:
            return
        self.global_series = previous.get("global_series", [])
        self.round_series = previous.get("round_series", {})
        for key, points in self.round_series.items():
            if points:
                self.last_seen_t[key] = points[-1][0]
        global_key = next((str(n.num) for n in self.inventory.nodes if n.role == "global"), None)
        if global_key and self.global_series:
            last_global = self.global_series[-1]["t"]
            self.last_seen_t[global_key] = max(self.last_seen_t.get(global_key, 0), last_global)
        self.started_at = previous.get("monitor_started_t", self.started_at)

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
        return alerts

    def poll_once(self) -> dict[str, Any]:
        with ThreadPoolExecutor(max_workers=len(self.inventory.nodes)) as pool:
            nodes = list(pool.map(self.probe, self.inventory.nodes))
        global_node = next((n for n in nodes if n["role"] == "global"), None)
        latest = self.global_series[-1] if self.global_series else None
        status = {
            "generated_at": _now_iso(),
            "t": round(time.time(), 3),
            "run_id": self.run_id,
            "profile": self.inventory.profile,
            "interval_s": self.interval,
            "monitor_host": os.uname().nodename if hasattr(os, "uname") else "",
            "monitor_started_t": self.started_at,
            "summary": {
                "alive": sum(1 for n in nodes if n["alive"]),
                "total": len(nodes),
                "global_state": global_node["state"] if global_node else None,
                "global_round": latest["round"] if latest else None,
                "last_accuracy": latest["accuracy"] if latest else None,
                "last_loss": latest["loss"] if latest else None,
                "expected_round_s": round(self._expected_round_s(), 1),
                "global_uptime_s": global_node["uptime_s"] if global_node else None,
            },
            "alerts": self._alerts(nodes),
            "nodes": nodes,
            "global_series": self.global_series,
            "round_series": self.round_series,
        }
        self._write(status)
        return status

    def _write(self, status: dict[str, Any]) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.out_dir / "status.json.tmp"
        tmp.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.out_dir / "status.json")


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
        f"~{s['expected_round_s']}s/round  ({status['generated_at']})"
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
    parser.add_argument("--tail", type=int, default=80, help="各ノードから読む events 行数")
    parser.add_argument("--once", action="store_true", help="1回だけ取得して終了")
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
    out_dir.mkdir(parents=True, exist_ok=True)
    page = Path(__file__).resolve().parent / "monitor" / "index.html"
    shutil.copyfile(page, out_dir / "index.html")

    monitor = Monitor(
        inventory, run_id, out_dir, interval=args.interval, identity=args.identity, tail=args.tail
    )
    if args.once:
        status = monitor.poll_once()
        if args.table:
            print(format_table(status))
        else:
            print(json.dumps(status, ensure_ascii=False, indent=1))
        return 0

    port = args.port or inventory.monitor_port
    server = None if args.no_serve else serve(out_dir, port)
    url = "" if server is None else f" http://0.0.0.0:{port}/"
    print(f"[monitor] run_id={run_id} interval={args.interval}s out={out_dir}{url}", flush=True)
    try:
        while True:
            started = time.monotonic()
            status = monitor.poll_once()
            s = status["summary"]
            print(
                f"[monitor] {status['generated_at']} alive={s['alive']}/{s['total']} "
                f"round={s['global_round']} acc={s['last_accuracy']} "
                f"alerts={len(status['alerts'])}",
                flush=True,
            )
            if args.table:
                print(format_table(status), flush=True)
            time.sleep(max(0.0, args.interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
