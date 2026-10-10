"""Jetson 台帳ヘルパー (scripts/hfl_nodes.py) のテスト。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hfl_nodes", REPO / "scripts" / "hfl_nodes.py")
hfl_nodes = importlib.util.module_from_spec(spec)
sys.modules["hfl_nodes"] = hfl_nodes
spec.loader.exec_module(hfl_nodes)

PROFILES = ["config/jetson-5tier", "config/jetson-5tier-longrun"]


@pytest.fixture(autouse=True)
def _chdir_repo(monkeypatch):
    monkeypatch.chdir(REPO)


@pytest.mark.parametrize("profile", PROFILES)
def test_inventory_is_consistent(profile):
    inventory = hfl_nodes.load_inventory(profile)
    assert hfl_nodes.check(inventory) == []
    assert [n.num for n in inventory.nodes] == [201, 202, 203, 204, 205, 206, 207]
    assert inventory.by_num(201).role == "global"
    assert inventory.by_num(204).parent_addr == "192.168.10.202:9001"
    assert inventory.by_num(205).partition == 2
    assert inventory.by_num(207).partition == 5
    assert inventory.by_num(206).config == f"{profile}/leaf_03.yaml"
    assert inventory.by_num(203).user == "tachilab-orin03"
    assert inventory.no_proxy_list().startswith("localhost,127.0.0.1,192.168.10.201")


def test_commands_match_the_powershell_launcher():
    inventory = hfl_nodes.load_inventory("config/jetson-5tier-longrun")
    leaf = hfl_nodes.build_command(inventory, inventory.by_num(205), dry_run=True)
    assert leaf[:2] == ["-m", "src.core.run_leaf"]
    assert leaf[leaf.index("--edge-address") + 1] == "192.168.10.202:9001"
    assert leaf[leaf.index("--partition-id") + 1] == "2"
    assert leaf[-1] == "--dry-run"

    edge = hfl_nodes.build_command(inventory, inventory.by_num(206))
    assert edge[:4] == [
        "-m", "src.core.run_edge", "--edge-config", "config/jetson-5tier-longrun/leaf_03.yaml",
    ]
    assert "--topology-config" in edge and "--dry-run" not in edge

    global_cmd = hfl_nodes.build_command(inventory, inventory.by_num(201), max_hours=1.5)
    assert global_cmd == [
        "-m", "src.core.global_server", "--config", "config/jetson-5tier-longrun/global.yaml",
        "--max-hours", "1.5",
    ]


def test_self_detection(monkeypatch):
    inventory = hfl_nodes.load_inventory("config/jetson-5tier-longrun")
    monkeypatch.setattr(hfl_nodes, "local_ips", lambda: {"127.0.0.1", "192.168.10.204"})
    assert hfl_nodes.self_num(inventory) == 204
    monkeypatch.setattr(hfl_nodes, "local_ips", lambda: {"192.168.10.148"})
    assert hfl_nodes.self_num(inventory) is None


def test_check_reports_bad_parent_order_and_missing_port(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "topology.yaml").write_text(
        "edges:\n  edge_01:\n    config_file: x.yaml\n    leaf_clients: [leaf_02]\n"
        "data_partition:\n  assignments: {leaf_02: 2}\n",
        encoding="utf-8",
    )
    (profile / "nodes.yaml").write_text(
        "nodes:\n"
        "  - {num: 205, role: leaf, id: leaf_02, parent: 202}\n"
        "  - {num: 201, role: global}\n"
        "  - {num: 202, role: edge, id: edge_01, port: 9001, parent: 201}\n",
        encoding="utf-8",
    )
    inventory = hfl_nodes.load_inventory(str(profile))
    errors = hfl_nodes.check(inventory)
    assert any("must be listed before" in e for e in errors)
    assert any("port is required" in e for e in errors)


def test_cli_list_and_node(capsys):
    assert hfl_nodes.main(["--profile", "config/jetson-5tier-longrun", "list"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 7 and out[0].startswith("201\t192.168.10.201\ttachilab-orin01\tglobal")
    assert hfl_nodes.main(["--profile", "config/jetson-5tier-longrun", "node", "207"]) == 0
    out = capsys.readouterr().out
    assert "NODE_ROLE=leaf" in out and "NODE_PARENT_ADDR=192.168.10.206:9004" in out
    assert hfl_nodes.main(["--profile", "config/jetson-5tier-longrun", "check"]) == 0
