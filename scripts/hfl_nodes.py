#!/usr/bin/env python
"""
Jetson 台帳 (config/<profile>/nodes.yaml) を読み、scripts/hfl.sh が使う情報を出力する。

Usage (リポジトリルートで実行):
    python scripts/hfl_nodes.py --profile config/jetson-5tier-longrun list
    python scripts/hfl_nodes.py --profile DIR node 204       # KEY=VALUE 行 (bash の eval 用)
    python scripts/hfl_nodes.py --profile DIR command 204 [--dry-run] [--max-hours H] [--resume]
    python scripts/hfl_nodes.py --profile DIR self           # 自ホストの IP から num を判定
    python scripts/hfl_nodes.py --profile DIR settings       # 共通設定 (KEY=VALUE 行)
    python scripts/hfl_nodes.py --profile DIR check          # 台帳と YAML の整合性検証
    python scripts/hfl_nodes.py --profile DIR json           # 台帳全体を JSON で出力

台帳に書くのは num / role / id / port / parent だけで、Edge の設定ファイルは
topology.yaml の edges[id].config_file、Leaf の partition は
data_partition.assignments[id]、親アドレスは親ノードの ip:port から導く。
"""

from __future__ import annotations

import argparse
import json
import shlex
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROLES = ("global", "edge", "leaf")


@dataclass
class Node:
    num: int
    role: str
    id: str
    ip: str
    user: str
    port: int | None = None
    parent_num: int | None = None
    parent_addr: str | None = None
    partition: int | None = None
    config: str | None = None


@dataclass
class Inventory:
    profile: str
    ssh_user_prefix: str
    subnet_prefix: str
    remote_dir: str
    monitor_port: int
    defaults_config: str
    global_config: str
    topology_config: str
    nodes: list[Node] = field(default_factory=list)

    def by_num(self, num: int) -> Node:
        for node in self.nodes:
            if node.num == num:
                return node
        raise KeyError(f"node {num} is not in {self.profile}/nodes.yaml")

    def no_proxy_list(self) -> str:
        return ",".join(["localhost", "127.0.0.1"] + [node.ip for node in self.nodes])


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_inventory(profile: str) -> Inventory:
    profile_dir = Path(profile)
    raw = _load_yaml(profile_dir / "nodes.yaml")
    topology = _load_yaml(profile_dir / "topology.yaml")
    edges = topology.get("edges", {})
    assignments = topology.get("data_partition", {}).get("assignments", {})

    inventory = Inventory(
        profile=str(profile_dir).replace("\\", "/"),
        ssh_user_prefix=str(raw.get("ssh_user_prefix", "tachilab-orin")),
        subnet_prefix=str(raw.get("subnet_prefix", "192.168.10.")),
        remote_dir=str(raw.get("remote_dir", "~/hierarchical-fl-persistent")),
        monitor_port=int(raw.get("monitor_port", 8000)),
        defaults_config=str(raw.get("defaults_config", "config/defaults.yaml")),
        global_config=f"{profile_dir.as_posix()}/global.yaml",
        topology_config=f"{profile_dir.as_posix()}/topology.yaml",
    )
    for entry in raw.get("nodes", []):
        num = int(entry["num"])
        role = str(entry["role"])
        node_id = str(entry.get("id", "global" if role == "global" else ""))
        node = Node(
            num=num,
            role=role,
            id=node_id,
            ip=f"{inventory.subnet_prefix}{num}",
            user=f"{inventory.ssh_user_prefix}{num - 200:02d}",
            port=int(entry["port"]) if entry.get("port") is not None else None,
            parent_num=int(entry["parent"]) if entry.get("parent") is not None else None,
        )
        if role == "edge":
            node.config = edges.get(node_id, {}).get("config_file")
        elif role == "leaf":
            partition = assignments.get(node_id)
            node.partition = int(partition) if partition is not None else None
        inventory.nodes.append(node)

    for node in inventory.nodes:
        if node.parent_num is not None:
            parent = inventory.by_num(node.parent_num)
            node.parent_addr = f"{parent.ip}:{parent.port}" if parent.port else None
    return inventory


def build_command(
    inventory: Inventory,
    node: Node,
    *,
    dry_run: bool = False,
    max_hours: float | None = None,
    resume: bool = False,
) -> list[str]:
    """python に渡す引数列 (-m 以降)。--max-hours と --resume は Global だけに付ける。"""
    common = [
        "--global-config", inventory.global_config,
        "--topology-config", inventory.topology_config,
        "--defaults-config", inventory.defaults_config,
    ]
    if node.role == "global":
        args = ["-m", "src.core.global_server", "--config", inventory.global_config]
        if max_hours is not None:
            args += ["--max-hours", str(max_hours)]
        if resume:
            args.append("--resume")
    elif node.role == "edge":
        if not node.config:
            raise ValueError(f"edge {node.id}: config_file not found in topology.yaml")
        args = ["-m", "src.core.run_edge", "--edge-config", node.config] + common
    elif node.role == "leaf":
        if node.partition is None or node.parent_addr is None:
            raise ValueError(f"leaf {node.id}: partition or parent address missing")
        args = [
            "-m", "src.core.run_leaf",
            "--client-id", node.id,
            "--edge-address", node.parent_addr,
            "--partition-id", str(node.partition),
        ] + common
    else:
        raise ValueError(f"unknown role: {node.role}")
    if dry_run:
        args.append("--dry-run")
    return args


def check(inventory: Inventory) -> list[str]:
    """台帳・topology・各 Edge 設定の整合性を検証し、問題点を返す。"""
    errors: list[str] = []
    topology = _load_yaml(Path(inventory.topology_config))
    edges = topology.get("edges", {})
    seen: set[int] = set()
    globals_ = [node for node in inventory.nodes if node.role == "global"]
    if len(globals_) != 1:
        errors.append(f"exactly one global node is required, found {len(globals_)}")

    for node in inventory.nodes:
        prefix = f"node {node.num} ({node.id})"
        if node.role not in ROLES:
            errors.append(f"{prefix}: unknown role {node.role}")
            continue
        if node.role != "global" and node.parent_num is None:
            errors.append(f"{prefix}: parent is required")
        if node.parent_num is not None:
            if node.parent_num not in seen:
                errors.append(f"{prefix}: parent {node.parent_num} must be listed before it")
            if node.parent_addr is None:
                errors.append(f"{prefix}: parent {node.parent_num} has no port")
        if node.role in ("global", "edge") and node.port is None:
            errors.append(f"{prefix}: port is required")
        if node.role == "edge":
            if node.id not in edges:
                errors.append(f"{prefix}: not in topology edges")
            elif node.config:
                try:
                    ec = _load_yaml(Path(node.config))["edge"]
                except (OSError, KeyError) as exc:
                    errors.append(f"{prefix}: cannot read {node.config}: {exc}")
                else:
                    sub_port = ec["sub_server_address"].rsplit(":", 1)[1]
                    if str(node.port) != sub_port:
                        errors.append(
                            f"{prefix}: port {node.port} != sub_server_address {sub_port}"
                        )
                    if ec.get("global_server_address") != node.parent_addr:
                        errors.append(
                            f"{prefix}: global_server_address "
                            f"{ec.get('global_server_address')} != parent {node.parent_addr}"
                        )
        if node.role == "leaf":
            parent = inventory.by_num(node.parent_num) if node.parent_num else None
            if parent is None or node.id not in edges.get(parent.id, {}).get("leaf_clients", []):
                errors.append(f"{prefix}: not listed in parent's leaf_clients")
            if node.partition is None:
                errors.append(f"{prefix}: no data_partition.assignments entry")
        seen.add(node.num)
    return errors


def local_ips() -> set[str]:
    ips: set[str] = set()
    try:
        out = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
        ips.update(out.split())
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except socket.gaierror:
        pass
    return ips


def self_num(inventory: Inventory) -> int | None:
    ips = local_ips()
    for node in inventory.nodes:
        if node.ip in ips:
            return node.num
    return None


def _shell_assignments(prefix: str, values: dict[str, Any]) -> str:
    lines = []
    for key, value in values.items():
        text = "" if value is None else str(value)
        lines.append(f"{prefix}{key.upper()}={shlex.quote(text)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--profile", default="config/jetson-5tier-longrun")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("settings")
    sub.add_parser("self")
    sub.add_parser("check")
    sub.add_parser("json")
    node_parser = sub.add_parser("node")
    node_parser.add_argument("num", type=int)
    cmd_parser = sub.add_parser("command")
    cmd_parser.add_argument("num", type=int)
    cmd_parser.add_argument("--dry-run", action="store_true")
    cmd_parser.add_argument("--max-hours", type=float, default=None)
    cmd_parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    inventory = load_inventory(args.profile)
    if args.cmd == "list":
        for node in inventory.nodes:
            fields = [
                node.num, node.ip, node.user, node.role, node.id, node.port,
                node.parent_num, node.parent_addr, node.partition, node.config,
            ]
            print("\t".join("-" if value is None else str(value) for value in fields))
    elif args.cmd == "settings":
        print(_shell_assignments("HFL_", {
            "ssh_user_prefix": inventory.ssh_user_prefix,
            "remote_dir": inventory.remote_dir,
            "monitor_port": inventory.monitor_port,
            "no_proxy_list": inventory.no_proxy_list(),
            "global_config": inventory.global_config,
            "topology_config": inventory.topology_config,
            "defaults_config": inventory.defaults_config,
            "node_nums": " ".join(str(node.num) for node in inventory.nodes),
        }))
    elif args.cmd == "self":
        num = self_num(inventory)
        if num is None:
            print(
                f"this host ({', '.join(sorted(local_ips()))}) is not in {args.profile}/nodes.yaml",
                file=sys.stderr,
            )
            return 1
        print(num)
    elif args.cmd == "check":
        errors = check(inventory)
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        if errors:
            return 1
        print(f"OK: {len(inventory.nodes)} nodes in {args.profile}")
    elif args.cmd == "json":
        print(json.dumps(asdict(inventory), ensure_ascii=False, indent=2))
    elif args.cmd == "node":
        node = inventory.by_num(args.num)
        print(_shell_assignments("NODE_", asdict(node)))
    elif args.cmd == "command":
        node = inventory.by_num(args.num)
        command = build_command(
            inventory, node, dry_run=args.dry_run, max_hours=args.max_hours, resume=args.resume
        )
        for arg in command:
            print(arg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
