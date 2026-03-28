"""Dataset adapters and sonar-specific preprocessing.

支持的数据集：
1. DummySonarDataset - 虚拟数据集，用于快速验证
2. NKSIDDataset - NK-Sonar-Image-Dataset (南开大学声呐图像数据集)
3. SonarImageDataset - 通用声呐图像数据集
"""

from .dataset import (
    # 数据集类
    DummySonarDataset,
    NKSIDDataset,
    SonarImageDataset,
    # 数据加载器工厂函数
    create_dataloader,
    create_dummy_dataloaders,
    create_nksid_dataloaders,
    # 工具函数
    get_sonar_transforms,
    download_nksid_dataset,
    # 常量
    NKSID_CLASSES,
)

__all__ = [
    # 数据集类
    "DummySonarDataset",
    "NKSIDDataset",
    "SonarImageDataset",
    # 数据加载器工厂函数
    "create_dataloader",
    "create_dummy_dataloaders",
    "create_nksid_dataloaders",
    # 工具函数
    "get_sonar_transforms",
    "download_nksid_dataset",
    # 常量
    "NKSID_CLASSES",
]
