"""学習間隔 (round_interval_s) の設定を読む。Global と Edge で同じ値を使うために共通化する。"""

from __future__ import annotations

from typing import Any


def round_interval_from_config(global_config: dict[str, Any]) -> float | None:
    """有効なラウンド間隔 (秒)。未設定・0 なら None (間隔なしで連続して学習する)。

    dry-run のときは dry_run.round_interval_s を使う (疎通確認で 10 分待たないため)。
    """
    if global_config.get("mode") == "dry_run":
        value = global_config.get("dry_run", {}).get("round_interval_s")
    else:
        value = global_config.get("server", {}).get("round_interval_s")
    return float(value) if value else None
