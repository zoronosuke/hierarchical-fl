"""
設定ファイルの読み込み・マージユーティリティ。

defaults.yaml をベースに、各ノード固有の設定でオーバーライドする。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """dictを再帰的にマージする。override側が優先。"""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    """YAMLファイルを読み込んで辞書を返す。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_config(
    config_path: str | Path,
    defaults_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    設定ファイルを読み込み、defaults があればマージして返す。

    Parameters
    ----------
    config_path : str | Path
        ノード固有の設定ファイルパス
    defaults_path : str | Path | None
        共通デフォルト設定ファイルパス。Noneなら config_path のみ使用。
    """
    config = load_yaml(config_path)
    if defaults_path is not None:
        defaults = load_yaml(defaults_path)
        config = _deep_merge(defaults, config)
    return config


def resolve_device(device_cfg: str = "auto") -> str:
    """
    デバイス文字列を解決する。

    "auto" の場合は CUDA が利用可能なら "cuda"、そうでなければ "cpu"。
    """
    if device_cfg == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg
