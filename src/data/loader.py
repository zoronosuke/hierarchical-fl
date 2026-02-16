"""
データセット読み込み・分割モジュール。

flwr-datasets の DirichletPartitioner / IidPartitioner を利用し、
設定ファイルに基づいてパーティションを生成する。
dry_run モードではダミーデータを返す。
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset


def _get_transforms(dataset_name: str):
    """データセットに応じた前処理transformを返す。"""
    from torchvision import transforms

    if dataset_name == "mnist":
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:  # cifar10
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])


def _apply_transforms(partition, transform):
    """HuggingFace Dataset のサンプルに torchvision transform を適用する。"""
    images = []
    labels = []

    # flwr-datasets は HuggingFace Dataset を返す
    # 画像カラム名はデータセットによって異なる
    img_key = "image" if "image" in partition.column_names else "img"
    label_key = "label"

    for sample in partition:
        img = sample[img_key]
        label = sample[label_key]
        images.append(transform(img))
        labels.append(label)

    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(images_tensor, labels_tensor)


def load_partition(
    dataset_name: str,
    partition_id: int,
    total_partitions: int,
    partition_method: str = "dirichlet",
    partition_params: dict[str, Any] | None = None,
    batch_size: int = 32,
) -> DataLoader:
    """
    指定パーティションのトレーニング用 DataLoader を返す。

    Parameters
    ----------
    dataset_name : str
        "cifar10" or "mnist"
    partition_id : int
        このクライアントに割り当てるパーティション番号
    total_partitions : int
        パーティション総数
    partition_method : str
        "iid" or "dirichlet"
    partition_params : dict
        DirichletPartitioner に渡すパラメータ (alpha, seed 等)
    batch_size : int
        バッチサイズ
    """
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner

    params = partition_params or {}

    if partition_method == "dirichlet":
        partitioner = DirichletPartitioner(
            num_partitions=total_partitions,
            partition_by="label",
            alpha=params.get("alpha", 0.5),
            min_partition_size=params.get("min_partition_size", 10),
            seed=params.get("seed", 42),
        )
    else:
        partitioner = IidPartitioner(num_partitions=total_partitions)

    # データセット名のマッピング
    ds_mapping = {
        "cifar10": "uoft-cs/cifar10",
        "mnist": "ylecun/mnist",
    }
    ds_name = ds_mapping.get(dataset_name, dataset_name)

    fds = FederatedDataset(
        dataset=ds_name,
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id)
    transform = _get_transforms(dataset_name)
    dataset = _apply_transforms(partition, transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)


def load_test_data(dataset_name: str, batch_size: int = 128) -> DataLoader:
    """テスト用データセット全体の DataLoader を返す (Global Server 評価用)。"""
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import IidPartitioner

    ds_mapping = {
        "cifar10": "uoft-cs/cifar10",
        "mnist": "ylecun/mnist",
    }
    ds_name = ds_mapping.get(dataset_name, dataset_name)

    # テスト分割はパーティションせず全データを使う
    fds = FederatedDataset(
        dataset=ds_name,
        partitioners={"train": IidPartitioner(num_partitions=1)},
    )
    test_split = fds.load_split("test")
    transform = _get_transforms(dataset_name)
    dataset = _apply_transforms(test_split, transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def create_dummy_dataloader(
    batch_size: int = 32,
    num_samples: int = 64,
    in_channels: int = 3,
    img_size: int = 32,
    num_classes: int = 10,
) -> DataLoader:
    """Dry-run 用のランダムダミーデータを返す。"""
    images = torch.randn(num_samples, in_channels, img_size, img_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=True)
