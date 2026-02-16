"""
モデル定義モジュール。

設定ファイルの model.name に応じてモデルを生成する。
dry_run モードでは軽量モデルに自動切り替え。
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """CIFAR-10 / MNIST 用の軽量CNN。"""

    def __init__(self, num_classes: int = 10, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        # CIFAR-10: 32x32 -> 8x8, MNIST: 28x28 -> 7x7 (概算、adaptive poolで吸収)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


class TinyCNN(nn.Module):
    """Dry-run用の超軽量モデル。"""

    def __init__(self, num_classes: int = 10, in_channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(8 * 2 * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_model(
    name: str,
    num_classes: int = 10,
    in_channels: int = 3,
    dry_run: bool = False,
) -> nn.Module:
    """設定に応じたモデルインスタンスを生成する。"""
    if dry_run:
        return TinyCNN(num_classes=num_classes, in_channels=in_channels)

    models = {
        "simplecnn": SimpleCNN,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    return models[name](num_classes=num_classes, in_channels=in_channels)


# =========================================================================
# パラメータ ⇔ NumPy配列 の変換ヘルパー (Flower との橋渡し)
# =========================================================================

def get_parameters(model: nn.Module) -> list[np.ndarray]:
    """モデルパラメータを NumPy 配列のリストとして返す。"""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """NumPy 配列のリストからモデルパラメータを設定する。"""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)
