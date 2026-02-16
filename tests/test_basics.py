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
