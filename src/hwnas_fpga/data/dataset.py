"""Dataset adapters and sonar-specific preprocessing.

支持的数据集：
1. DummySonarDataset - 虚拟数据集，用于快速验证
2. NKSIDDataset - NK-Sonar-Image-Dataset (南开大学声呐图像数据集)
   GitHub: https://github.com/Jorwnpay/NK-Sonar-Image-Dataset
"""

from torch.utils.data import Dataset, DataLoader, Subset
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Callable, Union
import os
import re

# 尝试导入PIL，用于图像加载
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

# 尝试导入torchvision transforms
try:
    import torchvision.transforms as T
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    T = None


class DummySonarDataset(Dataset):
    """虚拟声呐数据集，用于快速验证流程"""

    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 224,
        num_classes: int = 10,
        input_channels: int = 1,
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.input_channels = input_channels

        # 生成随机数据
        self.data = torch.randn(num_samples, input_channels, image_size, image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


# NKSID 官方类别名称
NKSID_CLASSES = [
    "big_propeller",
    "cylinder",
    "fishing_net",
    "floats",
    "iron_pipeline",
    "small_propeller",
    "soft_pipeline",
    "tire",
]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _resolve_nksid_root(data_dir: Union[str, Path]) -> Path:
    """解析 NKSID 有效根目录。

    支持以下传入方式：
    - 已解压后的 `.../NKSID`
    - 仓库根目录，里面包含 `NKSID/`
    """
    root = Path(data_dir).expanduser().resolve()
    nested_root = root / "NKSID"

    if nested_root.exists() and nested_root.is_dir():
        return nested_root
    return root


def _resolve_sample_path(root: Path, relative_path: str) -> Path:
    """解析 train_abs.txt 中的相对路径。"""
    rel = Path(relative_path.replace("\\", "/"))
    candidates = [root / rel, root.parent / rel]

    if rel.parts and rel.parts[0].lower() == root.name.lower():
        candidates.append(root / Path(*rel.parts[1:]))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


class AddSpeckleNoise(object):
    """声呐特有的散斑噪声 (Speckle Noise) 仿真
    散斑噪声是一种乘性噪声: J = I + I * n
    """
    def __init__(self, prob: float = 0.5, variance: float = 0.04):
        self.prob = prob
        self.variance = variance

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.prob:
            # tensor 形状通常为 [C, H, W]
            noise = torch.randn_like(tensor) * (self.variance ** 0.5)
            noisy_tensor = tensor + tensor * noise
            return torch.clamp(noisy_tensor, 0.0, 1.0)
        return tensor

def get_sonar_transforms(
    image_size: int = 224,
    is_training: bool = True,
    normalize: bool = True,
    output_channels: int = 1,
) -> Callable:
    """
    获取声呐图像专用的数据变换。
    
    声呐图像特点：
    - 单通道灰度图像 (支持转为3通道以适配预训练模型)
    - 存在斑点噪声 (Speckle Noise)
    - 目标边界模糊
    - 主要依赖纹理和几何特征
    
    Args:
        image_size: 目标图像尺寸
        is_training: 是否为训练模式（启用数据增强）
        normalize: 是否进行归一化
        output_channels: 输出通道数 (1或3)
        
    Returns:
        组合的变换函数
    """
    if not HAS_TORCHVISION:
        def _fallback_transform(image: "Image.Image") -> torch.Tensor:
            if output_channels == 3:
                image = image.convert('RGB')
            else:
                image = image.convert('L')
            resized = image.resize((image_size, image_size))
            array = np.asarray(resized, dtype=np.float32) / 255.0
            
            if output_channels == 1:
                tensor = torch.from_numpy(array).unsqueeze(0)
            else:
                tensor = torch.from_numpy(array).permute(2, 0, 1)
                
            if normalize:
                if output_channels == 1:
                    tensor = (tensor - 0.5) / 0.5
                else:
                    tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
            return tensor

        return _fallback_transform
    
    transforms_list = []
    
    # 调整大小
    transforms_list.append(T.Resize((image_size, image_size)))
    
    if is_training:
        # 训练时的数据增强
        transforms_list.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=15),
            # 声呐图像专用：模拟不同距离和角度
            T.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            # 亮度和对比度调整（模拟不同水下条件）
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ])
    
    # 转换为Tensor
    transforms_list.append(T.ToTensor())
    
    if is_training:
        # 添加物理仿真的散斑噪声
        transforms_list.append(AddSpeckleNoise(prob=0.3, variance=0.04))
    
    # 归一化
    if normalize:
        if output_channels == 1:
            # 使用声呐图像的典型均值和标准差
            transforms_list.append(T.Normalize(mean=[0.5], std=[0.5]))
        else:
            # ImageNet的均值和标准差
            transforms_list.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return T.Compose(transforms_list)


class NKSIDDataset(Dataset):
    """
    NK-Sonar-Image-Dataset (NKSID) 南开大学声呐图像数据集
    数据集信息：
    - 来源: https://github.com/Jorwnpay/NK-Sonar-Image-Dataset
    - 规模: 8个类别，2617张图片
    - 特点: 标签呈自然长尾分布（最多951张floats，最少20张fishing_net）
    - 采集: Oculus M750d多波束前视声纳，渤海湾 (39°N 118°E)
    - 参数: 距离2-15米，频率750kHz/1.2MHz
    
    类别分布（长尾）：
    0: big_propeller (大螺旋桨) - 203张
    1: cylinder (圆柱体) - 288张  
    2: fishing_net (渔网) - 20张 [最少]
    3: floats (浮标) - 951张 [最多]
    4: iron_pipeline (铁管道) - 112张
    5: small_propeller (小螺旋桨) - 94张
    6: soft_pipeline (软管道) - 115张
    7: tire (轮胎) - 834张
    
    目录结构：
    data_dir/
    ├── big_propeller/
    ├── cylinder/
    ├── fishing_net/
    ├── floats/
    ├── iron_pipeline/
    ├── small_propeller/
    ├── soft_pipeline/
    ├── tire/
    ├── train_abs.txt        # 图片路径和标签
    ├── kfold_train.txt      # 10次5折交叉验证训练集索引
    └── kfold_val.txt        # 10次5折交叉验证验证集索引
    """
    
    NUM_CLASSES = 8
    CLASSES = NKSID_CLASSES
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        transform: Optional[Callable] = None,
        is_training: bool = True,
        fold: int = 0,
        use_kfold: bool = True,
        split: str = "train",  # "train" or "val"
        output_channels: int = 1, # 1 为灰度图, 3 为 RGB (适配预训练模型)
        image_error_policy: str = "raise",
        split_file_policy: str = "raise",
    ):
        """
        初始化NKSID数据集。
        
        Args:
            data_dir: 数据集根目录路径
            image_size: 目标图像尺寸
            transform: 自定义变换函数，如果为None则使用默认变换
            is_training: 是否为训练模式
            fold: 使用第几折交叉验证 (0-9，共10次5折)
            use_kfold: 是否使用k折交叉验证划分
            split: 数据集划分 ("train" 或 "val")
            output_channels: 输出通道数 (1或3)
        """
        if not HAS_PIL:
            raise ImportError("PIL is required for image loading. Install with: pip install Pillow")
        
        self.original_data_dir = Path(data_dir)
        self.data_dir = _resolve_nksid_root(data_dir)
        self.image_size = image_size
        self.is_training = is_training
        self.fold = fold
        self.use_kfold = use_kfold
        self.split = split
        self.output_channels = output_channels
        self.image_error_policy = str(image_error_policy).strip().lower()
        if self.image_error_policy not in {"raise", "blank"}:
            raise ValueError(
                "image_error_policy must be 'raise' or 'blank', "
                f"got: {image_error_policy}"
            )
        self.split_file_policy = str(split_file_policy).strip().lower()
        if self.split_file_policy not in {"raise", "fallback"}:
            raise ValueError(
                "split_file_policy must be 'raise' or 'fallback', "
                f"got: {split_file_policy}"
            )
        self.classes = list(self.CLASSES)
        self.label_to_class = {idx: name for idx, name in enumerate(self.classes)}
        
        # 设置变换
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_sonar_transforms(
                image_size=image_size,
                is_training=is_training,
                output_channels=output_channels,
            )
        
        # 加载数据列表
        self.samples: List[Tuple[str, int]] = []
        self._load_samples()
        
        print(f"[NKSID] 加载完成: {len(self.samples)} 张图片, {len(self.classes)} 个类别")
        if self.use_kfold:
            print(f"  折数: {fold}, 划分: {split}")
    
    def _load_samples(self) -> None:
        """加载样本列表"""
        train_abs_file = self.data_dir / "train_abs.txt"
        
        if not train_abs_file.exists():
            # 如果没有train_abs.txt，尝试从目录结构加载
            self._load_from_directory()
            return
        
        # 从train_abs.txt加载所有样本
        all_samples = []
        with open(train_abs_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 格式: 相对路径 标签
                parts = line.split()
                if len(parts) >= 2:
                    img_path = parts[0]
                    label = int(parts[1])
                    full_path = str(_resolve_sample_path(self.data_dir, img_path))
                    all_samples.append((full_path, label))
                    relative_parts = Path(img_path.replace("\\", "/")).parts
                    if relative_parts:
                        class_name = relative_parts[0]
                        if class_name.lower() == self.data_dir.name.lower() and len(relative_parts) > 1:
                            class_name = relative_parts[1]
                        self.label_to_class.setdefault(label, class_name)

        if not all_samples:
            raise ValueError(f"No samples found in {train_abs_file}")

        self._normalize_labels(all_samples)
        
        # 使用k折交叉验证划分
        if self.use_kfold:
            self.samples = self._apply_kfold_split(all_samples)
        else:
            self.samples = all_samples
    
    def _load_from_directory(self) -> None:
        """从目录结构直接加载样本"""
        print(f"[NKSID] 从目录结构加载...")

        available_dirs = {path.name: path for path in self.data_dir.iterdir() if path.is_dir()}
        class_names = [name for name in self.CLASSES if name in available_dirs]
        if not class_names:
            class_names = sorted(available_dirs.keys())

        for class_idx, class_name in enumerate(class_names):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"  [警告] 类别目录不存在: {class_dir}")
                continue

            for img_path in class_dir.rglob("*"):
                if _is_image_file(img_path):
                    self.samples.append((str(img_path), class_idx))

        if not self.samples:
            zip_files = list(self.data_dir.rglob("*.zip"))
            if zip_files:
                raise ValueError(
                    f"No extracted images found in {self.data_dir}. "
                    "The NKSID repository stores images in zip archives; extract all class zips first."
                )
            raise ValueError(f"No images found in {self.data_dir}")

        self.classes = class_names
        self.label_to_class = {idx: name for idx, name in enumerate(self.classes)}
        
        # 简单的训练/验证划分（80/20）
        if self.use_kfold:
            np.random.seed(42 + self.fold)
            indices = np.random.permutation(len(self.samples))
            split_idx = int(len(self.samples) * 0.8)
            
            if self.split == "train":
                selected_indices = indices[:split_idx]
            else:
                selected_indices = indices[split_idx:]
            
            self.samples = [self.samples[i] for i in selected_indices]

    def _normalize_labels(self, samples: List[Tuple[str, int]]) -> None:
        """将标签规范到从 0 开始的连续整数。"""
        labels = sorted({label for _, label in samples})
        if labels == list(range(len(labels))):
            self.classes = [
                self.label_to_class.get(
                    idx,
                    self.CLASSES[idx] if idx < len(self.CLASSES) else str(idx),
                )
                for idx in labels
            ]
            return

        label_map = {old: new for new, old in enumerate(labels)}
        for idx, (path, label) in enumerate(samples):
            samples[idx] = (path, label_map[label])

        remapped = {}
        for old_label, class_name in self.label_to_class.items():
            remapped[label_map[old_label]] = class_name
        self.label_to_class = remapped
        self.classes = [self.label_to_class.get(idx, str(idx)) for idx in range(len(labels))]
    
    def _apply_kfold_split(self, all_samples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """应用k折交叉验证划分"""
        if self.split == "full" or self.fold is None:
            return all_samples

        if self.split == "train":
            kfold_file = self.data_dir / "kfold_train.txt"
        else:
            kfold_file = self.data_dir / "kfold_val.txt"
        
        if not kfold_file.exists():
            if self.split_file_policy != "fallback":
                raise FileNotFoundError(
                    f"NKSID k-fold split file is missing: {kfold_file}. "
                    "The frozen evaluation protocol requires the official split files; "
                    "pass split_file_policy='fallback' only for informal experiments."
                )
            print(f"  [警告] k折划分文件不存在: {kfold_file}，使用随机划分 (fallback 模式)")
            np.random.seed(42 + self.fold)
            indices = np.random.permutation(len(all_samples))
            split_idx = int(len(all_samples) * 0.8)

            if self.split == "train":
                selected_indices = indices[:split_idx]
            else:
                selected_indices = indices[split_idx:]

            return [all_samples[i] for i in selected_indices]
        
        # 读取k折划分文件
        with open(kfold_file, 'r', encoding='utf-8') as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        split_records: List[List[int]] = []
        for line in raw_lines:
            if line.startswith("#"):
                continue

            indices = [int(idx) for idx in re.findall(r"\d+", line)]
            if not indices:
                continue

            split_records.append(indices)

        if not split_records:
            raise ValueError(f"No valid fold indices found in {kfold_file}")

        record_index = self.fold
        if record_index >= len(split_records):
            print(f"  [警告] 折数 {self.fold} 超出范围，使用第0折")
            record_index = 0

        indices = split_records[record_index]
        return [all_samples[i] for i in indices if i < len(all_samples)]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # 加载图像
        try:
            target_mode = 'RGB' if self.output_channels == 3 else 'L'
            with Image.open(img_path) as source:
                image = source.convert(target_mode)
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            else:
                # 默认转换为tensor
                image_arr = np.array(image)
                if self.output_channels == 1:
                    image_tensor = torch.from_numpy(image_arr).float().unsqueeze(0) / 255.0
                else:
                    image_tensor = torch.from_numpy(image_arr).float().permute(2, 0, 1) / 255.0
                image = image_tensor
            
            return image, label
            
        except Exception as e:
            if self.image_error_policy == "raise":
                raise RuntimeError(
                    f"NKSID image load failed for index={idx}, path={img_path}"
                ) from e
            print(f"[NKSID] 加载图像失败: {img_path}, 错误: {e}")
            # 返回一个空白图像
            target_mode = 'RGB' if self.output_channels == 3 else 'L'
            if self.transform:
                dummy = Image.new(target_mode, (self.image_size, self.image_size), 128 if target_mode == 'L' else (128,128,128))
                return self.transform(dummy), label
            else:
                return torch.zeros(self.output_channels, self.image_size, self.image_size), label
    
    def get_class_distribution(self) -> dict:
        """获取类别分布统计"""
        distribution = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            if 0 <= label < len(self.classes):
                distribution[self.classes[label]] += 1
        return distribution
    
    def get_class_weights(self) -> torch.Tensor:
        """
        计算类别权重（用于处理长尾分布）
        
        使用逆频率加权：weight[c] = N / (num_classes * count[c])
        """
        num_classes = len(self.classes)
        counts = torch.zeros(num_classes)
        for _, label in self.samples:
            counts[label] += 1
        
        # 避免除零
        counts = torch.clamp(counts, min=1)
        
        # 逆频率权重
        weights = len(self.samples) / (num_classes * counts)
        
        # 归一化
        weights = weights / weights.sum() * num_classes
        
        return weights


class SonarImageDataset(Dataset):
    """通用声呐图像数据集
    
    支持从目录加载任意声呐图像数据集。
    目录结构应为：
    data_dir/
    ├── class1/
    │   ├── img1.png
    │   └── img2.png
    ├── class2/
    │   └── ...
    └── ...
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        num_classes: Optional[int] = None,
        input_channels: int = 1,
        transform: Optional[Callable] = None,
        is_training: bool = True,
    ):
        if not HAS_PIL:
            raise ImportError("PIL is required. Install with: pip install Pillow")
        
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.input_channels = input_channels
        self.is_training = is_training
        
        # 设置变换
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_sonar_transforms(
                image_size=image_size,
                is_training=is_training,
            )
        
        # 加载样本
        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = []
        self._load_from_directory()
        
        self.num_classes = num_classes or len(self.classes)
        
        print(f"[SonarDataset] 加载完成: {len(self.samples)} 张图片, {self.num_classes} 个类别")

    def _load_from_directory(self) -> None:
        """从目录结构加载样本"""
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        # 获取所有子目录作为类别
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        self.classes = [d.name for d in class_dirs]
        
        for class_idx, class_dir in enumerate(class_dirs):
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']:
                for img_path in class_dir.glob(ext):
                    self.samples.append((str(img_path), class_idx))
                for img_path in class_dir.glob(ext.upper()):
                    self.samples.append((str(img_path), class_idx))
        
        if not self.samples:
            raise ValueError(f"No images found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path)
            
            # 转换为灰度图
            if image.mode != 'L':
                image = image.convert('L')
            
            if self.transform:
                image = self.transform(image)
            else:
                image = torch.from_numpy(np.array(image)).float().unsqueeze(0) / 255.0
            
            return image, label
            
        except Exception as e:
            print(f"[SonarDataset] 加载失败: {img_path}, 错误: {e}")
            return torch.zeros(1, self.image_size, self.image_size), label


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """创建数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_dummy_dataloaders(
    batch_size: int = 32,
    image_size: int = 224,
    num_classes: int = 10,
    input_channels: int = 1,
    num_train: int = 1000,
    num_val: int = 200,
) -> Tuple[DataLoader, DataLoader]:
    """创建虚拟数据加载器，用于快速验证"""
    train_dataset = DummySonarDataset(
        num_samples=num_train,
        image_size=image_size,
        num_classes=num_classes,
        input_channels=input_channels,
    )
    val_dataset = DummySonarDataset(
        num_samples=num_val,
        image_size=image_size,
        num_classes=num_classes,
        input_channels=input_channels,
    )

    train_loader = create_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def create_nksid_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    fold: int = 0,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_kfold: bool = True,
    valid_size: Optional[Union[int, float]] = None,
    split_seed: int = 42,
    image_error_policy: str = "raise",
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    创建NKSID数据集的数据加载器。
    
    Args:
        data_dir: NKSID数据集根目录
        batch_size: 批次大小
        image_size: 图像尺寸
        fold: k折交叉验证折数 (0-9)
        num_workers: 数据加载进程数
        pin_memory: 是否固定内存
        
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        class_weights: 类别权重（用于处理长尾分布）
    """
    if valid_size is None and not use_kfold:
        valid_size = 0.2

    if valid_size is not None:
        train_full = NKSIDDataset(
            data_dir=data_dir,
            image_size=image_size,
            is_training=True,
            fold=fold,
            use_kfold=False,
            split="full",
            image_error_policy=image_error_policy,
        )
        val_full = NKSIDDataset(
            data_dir=data_dir,
            image_size=image_size,
            is_training=False,
            fold=fold,
            use_kfold=False,
            split="full",
            image_error_policy=image_error_policy,
        )
        total = len(train_full)
        if total <= 1:
            raise ValueError("NKSID dataset is too small to split into train/validation sets")

        if isinstance(valid_size, float):
            if not (0.0 < valid_size < 1.0):
                raise ValueError(f"valid_size as float must be in (0, 1), got: {valid_size}")
            valid_count = int(round(total * valid_size))
        else:
            valid_count = int(valid_size)

        if valid_count <= 0 or valid_count >= total:
            raise ValueError(f"valid_size must be in [1, {total - 1}], got: {valid_size}")

        rng = np.random.default_rng(int(split_seed) + int(fold))
        indices = np.arange(total)
        rng.shuffle(indices)
        valid_indices = indices[:valid_count].tolist()
        train_indices = indices[valid_count:].tolist()

        train_dataset = Subset(train_full, train_indices)
        val_dataset = Subset(val_full, valid_indices)

        num_classes = len(train_full.classes)
        counts = torch.zeros(num_classes, dtype=torch.float32)
        for sample_idx in train_indices:
            _, label = train_full.samples[sample_idx]
            if 0 <= label < num_classes:
                counts[label] += 1
        counts = torch.clamp(counts, min=1.0)
        class_weights = len(train_indices) / (num_classes * counts)
        class_weights = class_weights / class_weights.sum() * num_classes
    else:
        # 创建训练集
        train_dataset = NKSIDDataset(
            data_dir=data_dir,
            image_size=image_size,
            is_training=True,
            fold=fold,
            use_kfold=use_kfold,
            split="train",
            image_error_policy=image_error_policy,
        )

        # 创建验证集
        val_dataset = NKSIDDataset(
            data_dir=data_dir,
            image_size=image_size,
            is_training=False,
            fold=fold,
            use_kfold=use_kfold,
            split="val",
            image_error_policy=image_error_policy,
        )

        # 获取类别权重
        class_weights = train_dataset.get_class_weights()
    
    # 创建数据加载器
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader, class_weights

def create_protocol_dataloaders(
    data_dir: str,
    *,
    fold: int,
    seed: int,
    batch_size: int = 32,
    image_size: int = 224,
    inner_val_fraction: float = 0.15,
    num_workers: int = 4,
    pin_memory: bool = True,
    output_channels: int = 1,
):
    """Create dataloaders under the frozen NKSID evaluation protocol.

    Returns train / inner-val / outer-val loaders plus the resolved
    :class:`~hwnas_fpga.data.protocol.ProtocolSplit` and per-class train
    counts. The outer validation loader must only be consumed once, for the
    final report of a run — never for epoch or architecture selection.
    """
    from hwnas_fpga.data.protocol import build_protocol_split

    train_view = NKSIDDataset(
        data_dir=data_dir,
        image_size=image_size,
        is_training=True,
        fold=fold,
        use_kfold=False,
        split="full",
        output_channels=output_channels,
        image_error_policy="raise",
    )
    eval_view = NKSIDDataset(
        data_dir=data_dir,
        image_size=image_size,
        is_training=False,
        fold=fold,
        use_kfold=False,
        split="full",
        output_channels=output_channels,
        image_error_policy="raise",
    )
    if len(train_view) != len(eval_view):
        raise RuntimeError("train/eval dataset views disagree on sample count")

    num_classes = len(train_view.classes)
    split = build_protocol_split(
        train_view.samples,
        data_dir,
        fold_index=fold,
        seed=seed,
        inner_val_fraction=inner_val_fraction,
        num_classes=num_classes,
    )

    train_loader = create_dataloader(
        Subset(train_view, list(split.train_indices)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    inner_val_loader = create_dataloader(
        Subset(eval_view, list(split.inner_val_indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    outer_val_loader = create_dataloader(
        Subset(eval_view, list(split.outer_val_indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_class_counts = torch.tensor(
        split.metadata["train_class_counts"], dtype=torch.float32
    )
    return {
        "train_loader": train_loader,
        "inner_val_loader": inner_val_loader,
        "outer_val_loader": outer_val_loader,
        "split": split,
        "num_classes": num_classes,
        "classes": list(train_view.classes),
        "train_class_counts": train_class_counts,
    }


def download_nksid_dataset(target_dir: str = "data/NKSID") -> str:
    """
    下载NKSID数据集的说明。
    
    由于数据集托管在GitHub，需要手动下载。
    
    Args:
        target_dir: 目标目录
        
    Returns:
        下载说明字符串
    """
    instructions = f"""
    ============================================================
    NK-Sonar-Image-Dataset (NKSID) 下载说明
    ============================================================
    
    数据集地址: https://github.com/Jorwnpay/NK-Sonar-Image-Dataset
    
    下载步骤:
    1. git clone https://github.com/Jorwnpay/NK-Sonar-Image-Dataset.git temp_nksid
    2. 将 temp_nksid/NKSID/ 目录移动到 {target_dir}
    3. 解压所有 .zip 文件 (包括 tire/ 目录下的两个分卷)
    4. 删除 .zip 文件
    
    解压后的目录结构:
    {target_dir}/
    ├── big_propeller/    # 203张 - 大螺旋桨
    ├── cylinder/         # 288张 - 圆柱体
    ├── fishing_net/      # 20张  - 渔网 (最少)
    ├── floats/           # 951张 - 浮标 (最多)
    ├── iron_pipeline/    # 112张 - 铁管道
    ├── small_propeller/  # 94张  - 小螺旋桨
    ├── soft_pipeline/    # 115张 - 软管道
    ├── tire/             # 834张 - 轮胎
    ├── train_abs.txt
    ├── kfold_train.txt
    └── kfold_val.txt
    
    数据集信息:
    - 8个类别，2617张图片
    - 长尾分布 (不平衡比例约 47.5:1)
    - Oculus M750d 前视声呐采集
    - 渤海湾水下环境 (39°N 118°E)
    ============================================================
    """
    print(instructions)
    return instructions
