"""
Global モデルのチェックポイント (保存と再開)。

1ファイルの .npz に、モデルの重み (arr_0, arr_1, ...) とメタ情報 (JSON 文字列) を入れる。
書き込みは一時ファイル → fsync → os.replace で行い、保存中に電源が落ちても
直前のチェックポイントが壊れないようにする。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from flwr.common import NDArrays

_META_KEY = "__meta__"


def save_checkpoint(path: str | Path, parameters: NDArrays, meta: dict[str, Any]) -> None:
    """重みとメタ情報を原子的に保存する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    arrays = {f"arr_{i}": np.asarray(array) for i, array in enumerate(parameters)}
    arrays[_META_KEY] = np.array(json.dumps(meta, ensure_ascii=False))
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_directory(path.parent)


def load_checkpoint(path: str | Path) -> tuple[NDArrays, dict[str, Any]]:
    """save_checkpoint で保存した重みとメタ情報を読む。"""
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data[_META_KEY]))
        count = sum(1 for key in data.files if key.startswith("arr_"))
        parameters = [data[f"arr_{i}"].copy() for i in range(count)]
    return parameters, meta


def _fsync_directory(directory: Path) -> None:
    """rename を電源断後も残すためにディレクトリも fsync する (Linux のみ)。"""
    if os.name != "posix":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
