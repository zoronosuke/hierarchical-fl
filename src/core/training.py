"""
学習・評価の共通ロジック。

Flower Client の fit() / evaluate() から呼ばれる。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.logger import get_logger

logger = get_logger("training")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    lr: float = 0.01,
    optimizer_name: str = "sgd",
    momentum: float = 0.9,
    weight_decay: float = 0.0001,
    max_steps: int | None = None,
) -> tuple[float, int]:
    """
    1エポック分の学習を行う。

    Parameters
    ----------
    model : nn.Module
    dataloader : DataLoader
    device : str
    lr, optimizer_name, momentum, weight_decay : ハイパーパラメータ
    max_steps : int | None
        dry_run 用。指定されたステップ数で打ち切り。

    Returns
    -------
    avg_loss : float
        平均損失
    num_examples : int
        学習に使用したサンプル数
    """
    model.to(device)
    model.train()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    num_examples = 0

    for step, (images, labels) in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        num_examples += images.size(0)

    avg_loss = total_loss / max(num_examples, 1)
    return avg_loss, num_examples


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> tuple[float, float, int]:
    """
    モデルの評価を行う。

    Returns
    -------
    loss : float
    accuracy : float
    num_examples : int
    """
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    num_examples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            num_examples += images.size(0)

    avg_loss = total_loss / max(num_examples, 1)
    accuracy = correct / max(num_examples, 1)
    return avg_loss, accuracy, num_examples
