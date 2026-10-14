"""基本的なユニットテスト。"""

import tempfile
from pathlib import Path

import yaml


def test_config_deep_merge():
    """設定のマージが正しく動作するか。"""
    from src.utils.config import _deep_merge

    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99}, "e": 5}
    result = _deep_merge(base, override)

    assert result["a"] == 1
    assert result["b"]["c"] == 99
    assert result["b"]["d"] == 3
    assert result["e"] == 5


def test_config_load_yaml():
    """YAMLファイルの読み込み。"""
    from src.utils.config import load_yaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"key": "value", "nested": {"a": 1}}, f)
        f.flush()
        result = load_yaml(f.name)

    assert result["key"] == "value"
    assert result["nested"]["a"] == 1


def test_jetson_7node_config():
    """7台実機設定が1PC用設定と分離され、Partitionが重複しない。"""
    from src.utils.config import load_yaml

    repository = Path(__file__).resolve().parents[1]
    common_edge = load_yaml(repository / "config/edge/edge_01.yaml")
    global_config = load_yaml(repository / "config/jetson-7node/global.yaml")
    edge_01 = load_yaml(repository / "config/jetson-7node/edge_01.yaml")
    edge_02 = load_yaml(repository / "config/jetson-7node/edge_02.yaml")
    topology = load_yaml(repository / "config/topology.yaml")

    assert common_edge["edge"]["global_server_address"] == "127.0.0.1:8080"
    assert global_config["server"]["address"] == "0.0.0.0:8080"
    assert global_config["server"]["round_timeout"] == 4200.0
    assert edge_01["edge"]["global_server_address"] == "192.168.10.201:8080"
    assert edge_02["edge"]["global_server_address"] == "192.168.10.201:8080"

    assignments = topology["data_partition"]["assignments"]
    assert assignments == {
        "edge_01_internal": 0,
        "leaf_01": 1,
        "leaf_02": 2,
        "edge_02_internal": 3,
        "leaf_03": 4,
        "leaf_04": 5,
    }
    assert sorted(assignments.values()) == list(range(6))



def test_5tier_configs():
    """5段構成の親子アドレス・タイムアウト連鎖・Partitionが整合する。"""
    from src.utils.config import load_yaml

    repository = Path(__file__).resolve().parents[1]
    for config_dir, global_file in (
        ("config/jetson-5tier", "config/jetson-5tier/global.yaml"),
        ("config/jetson-5tier-longrun", "config/jetson-5tier-longrun/global.yaml"),
        ("config/jetson-5tier-paced", "config/jetson-5tier-paced/global.yaml"),
        ("config/local-5tier", "config/global.yaml"),
        ("config/local-5tier", "config/local-5tier/global_longrun.yaml"),
        ("config/local-5tier", "config/local-5tier/global_paced.yaml"),
    ):
        global_config = load_yaml(repository / global_file)
        topology = load_yaml(repository / config_dir / "topology.yaml")
        edges = topology["edges"]
        assert list(edges) == ["edge_01", "edge_02", "leaf_01", "leaf_03"]

        configs = {
            edge_id: load_yaml(repository / info["config_file"])["edge"]
            for edge_id, info in edges.items()
        }
        parents = {
            child: parent
            for parent, info in edges.items()
            for child in info["leaf_clients"]
            if child in edges
        }
        assert parents == {"leaf_01": "edge_01", "leaf_03": "leaf_01"}

        for edge_id, ec in configs.items():
            participants = len(edges[edge_id]["leaf_clients"]) + 1
            assert ec["min_available_clients"] == participants
            assert ec["parent_result_timeout"] > ec["sub_round_timeout"]
            parent = parents.get(edge_id)
            if parent is None:
                assert ec["global_server_address"].endswith(":8080")
                parent_timeout = global_config["server"]["round_timeout"]
            else:
                parent_port = configs[parent]["sub_server_address"].rsplit(":", 1)[1]
                assert ec["global_server_address"].endswith(f":{parent_port}")
                parent_timeout = configs[parent]["sub_round_timeout"]
                assert ec["parent_round_timeout"] == parent_timeout
            assert parent_timeout > ec["parent_result_timeout"]
            # 次の親モデルは Global が他の子を round_timeout まで待った後に届きうる
            assert ec["parent_model_timeout"] > parent_timeout
            assert ec["parent_model_timeout"] > global_config["server"]["round_timeout"]

        assignments = topology["data_partition"]["assignments"]
        assert sorted(assignments.values()) == list(range(6))
        assert topology["data_partition"]["total_partitions"] == 6


def test_longrun_profile():
    """長時間実験プロファイルは実質無限ラウンド + 時間上限で止まる設定になっている。"""
    from src.utils.config import load_yaml

    repository = Path(__file__).resolve().parents[1]
    global_config = load_yaml(repository / "config/jetson-5tier-longrun/global.yaml")
    assert global_config["server"]["num_rounds"] >= 1_000_000
    assert global_config["server"]["max_duration_hours"] > 0
    assert global_config["server"]["round_timeout"] >= 600
    for name in ("edge_01", "edge_02", "leaf_01", "leaf_03"):
        ec = load_yaml(repository / f"config/jetson-5tier-longrun/{name}.yaml")["edge"]
        assert ec["shutdown_timeout"] >= 120
        assert ec["parent_model_timeout"] >= global_config["server"]["round_timeout"] + 600


def test_paced_profile():
    """10分間隔プロファイル: 間隔・毎ラウンド保存・待ち時間の連鎖・自分のファイルを参照。"""
    from src.utils.config import load_yaml
    from src.utils.schedule import round_interval_from_config

    repository = Path(__file__).resolve().parents[1]
    global_config = load_yaml(repository / "config/jetson-5tier-paced/global.yaml")
    server = global_config["server"]
    assert server["round_interval_s"] == 600 and server["round_align"] == "clock"
    assert server["checkpoint_every"] == 1
    assert 0 < global_config["dry_run"]["round_interval_s"] <= 30  # 疎通確認で長く待たない
    assert round_interval_from_config(global_config) == 600
    dry = {**global_config, "mode": "dry_run"}
    assert round_interval_from_config(dry) == global_config["dry_run"]["round_interval_s"]
    for name in ("edge_01", "edge_02", "leaf_01", "leaf_03"):
        ec = load_yaml(repository / f"config/jetson-5tier-paced/{name}.yaml")["edge"]
        needed = server["round_timeout"] + server["round_interval_s"] + 60
        assert ec["parent_model_timeout"] > needed
    nodes = load_yaml(repository / "config/jetson-5tier-paced/nodes.yaml")
    assert nodes["sampler_interval_s"] == 5


def test_profiles_reference_their_own_files():
    """topology.yaml は自分のディレクトリの Edge 設定を参照する (コピー時の取り違え防止)。"""
    from src.utils.config import load_yaml

    repository = Path(__file__).resolve().parents[1]
    for profile in ("jetson-5tier", "jetson-5tier-longrun", "jetson-5tier-paced", "local-5tier"):
        topology = load_yaml(repository / "config" / profile / "topology.yaml")
        for edge_id, info in topology["edges"].items():
            assert info["config_file"].startswith(f"config/{profile}/"), (profile, edge_id)

def test_resolve_device():
    """デバイス解決。"""
    from src.utils.config import resolve_device

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    # "auto" は環境依存なので cpu か cuda のいずれか
    result = resolve_device("auto")
    assert result in ("cpu", "cuda")


def test_create_model_simplecnn():
    """SimpleCNN モデルの生成。"""
    from src.models.nets import create_model

    model = create_model("simplecnn", num_classes=10, in_channels=3)
    assert model is not None

    import torch
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)


def test_create_model_dry_run():
    """dry_run 時は TinyCNN になる。"""
    from src.models.nets import create_model

    model = create_model("simplecnn", num_classes=10, in_channels=3, dry_run=True)
    assert model.__class__.__name__ == "TinyCNN"

    import torch
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)


def test_get_set_parameters():
    """パラメータの取得・設定の往復。"""
    from src.models.nets import create_model, get_parameters, set_parameters

    model = create_model("simplecnn", num_classes=10, in_channels=3, dry_run=True)
    params = get_parameters(model)
    assert isinstance(params, list)
    assert len(params) > 0

    # 別のモデルに設定
    model2 = create_model("simplecnn", num_classes=10, in_channels=3, dry_run=True)
    set_parameters(model2, params)

    params2 = get_parameters(model2)
    for p1, p2 in zip(params, params2):
        assert (p1 == p2).all()


def test_logger():
    """ロガーが取得できる。"""
    from src.utils.logger import get_logger

    logger = get_logger("test")
    assert logger is not None
    logger.info("Test log message")


def test_dummy_dataloader():
    """ダミーデータローダーの生成。"""
    from src.data.loader import create_dummy_dataloader

    loader = create_dummy_dataloader(batch_size=8, num_samples=32)
    batch = next(iter(loader))
    images, labels = batch
    assert images.shape[0] <= 8
    assert images.shape[1] == 3
    assert labels.shape[0] == images.shape[0]


def test_strategy_registry():
    """戦略レジストリに fedavg が登録されている。"""
    from src.strategies.aggregation import _STRATEGY_REGISTRY

    assert "fedavg" in _STRATEGY_REGISTRY
