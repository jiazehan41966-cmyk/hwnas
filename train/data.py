"""YOLO-format dataset helpers for the detection experiment branch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import random
from PIL import Image
HAS_PIL = True
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def _seed_dataloader_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("valid", "val"),
    "valid": ("valid", "val"),
    "test": ("test",),
}


def _resolve_yolo_root(data_dir: str | Path) -> Path:
    root = Path(data_dir).expanduser().resolve()
    nested = root / "source_reconstruction" / "v6"
    if nested.is_dir():
        return nested
    return root


def _load_yolo_names(root: Path) -> list[str]:
    data_yaml = root / "data.yaml"
    if not data_yaml.is_file():
        return []
    try:
        import yaml

        payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    names = payload.get("names", [])
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def _resolve_images_dir(root: Path, split: str) -> Path:
    aliases = SPLIT_ALIASES.get(str(split).strip().lower(), (split,))
    for alias in aliases:
        candidate = root / alias / "images"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"YOLO image split not found under {root}: {split}")


def network_boxes_to_original(
    boxes: torch.Tensor,
    *,
    image_size: int,
    orig_width: int,
    orig_height: int,
) -> torch.Tensor:
    """Map xyxy boxes from the square network input back to original pixels.

    The loader applies a plain per-axis resize with no letterbox padding and no
    crop, so the inverse is an independent scale on x and y.
    """

    tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    if int(image_size) <= 0:
        raise ValueError("image_size must be positive")
    scale_x = float(orig_width) / float(image_size)
    scale_y = float(orig_height) / float(image_size)
    scaled = tensor.clone()
    scaled[:, 0::2] *= scale_x
    scaled[:, 1::2] *= scale_y
    return scaled


def yolo_normalized_to_xyxy(
    rows: Sequence[Sequence[float]],
    *,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert YOLO normalized labels to resized xyxy boxes.

    The loader stretches every image to a square network input, so both axes
    use the same ``image_size``. For original-pixel boxes use
    ``yolo_normalized_to_original_xyxy``.
    """

    return yolo_normalized_to_original_xyxy(
        rows, orig_width=int(image_size), orig_height=int(image_size)
    )


def yolo_normalized_to_original_xyxy(
    rows: Sequence[Sequence[float]],
    *,
    orig_width: int,
    orig_height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert YOLO normalized labels to original-pixel xyxy boxes."""

    boxes: list[list[float]] = []
    labels: list[int] = []
    width = float(orig_width)
    height = float(orig_height)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("original image size must be positive")
    for row in rows:
        if len(row) < 5:
            continue
        label, cx, cy, box_w, box_h = row[:5]
        x1 = (float(cx) - float(box_w) / 2.0) * width
        y1 = (float(cy) - float(box_h) / 2.0) * height
        x2 = (float(cx) + float(box_w) / 2.0) * width
        y2 = (float(cy) + float(box_h) / 2.0) * height
        boxes.append(
            [
                max(0.0, min(width, x1)),
                max(0.0, min(height, y1)),
                max(0.0, min(width, x2)),
                max(0.0, min(height, y2)),
            ]
        )
        labels.append(int(label))
    return (
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(labels, dtype=torch.long).reshape(-1),
    )


class YoloDetectionDataset(Dataset):
    """Minimal YOLO txt detection dataset for branch-level experiment smoke tests."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        split: str = "train",
        image_size: int = 640,
        input_channels: int = 1,
        max_samples: int | None = None,
        augment_config: Mapping[str, Any] | None = None,
    ) -> None:
        if not HAS_PIL:
            raise ImportError("PIL is required for YOLO detection image loading")
        if split not in {"train", "valid", "val"}:
            raise ValueError("Only train/valid are permitted; test is sealed")
        if image_size != 640 or input_channels != 1:
            raise ValueError("Fixed ELASTIC input is grayscale 640x640")
        self.root = _resolve_yolo_root(data_dir)
        self.split = str(split)
        self.image_size = int(image_size)
        self.input_channels = int(input_channels)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.input_channels not in {1, 3}:
            raise ValueError("input_channels must be 1 or 3")
        self.augment_config = dict(augment_config or {})

        images_dir = _resolve_images_dir(self.root, self.split)
        self.images_dir = images_dir
        self.labels_dir = images_dir.parent.parent / images_dir.parent.name / "labels"
        self.class_names = _load_yolo_names(self.root)
        samples = sorted(
            path for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if max_samples is not None:
            samples = samples[: int(max_samples)]
        if not samples:
            raise ValueError(f"No YOLO images found in {images_dir}")
        self.samples = samples
        self.sample_labels = [
            _read_yolo_label_ids(self.labels_dir / f"{path.stem}.txt")
            for path in self.samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image_path = self.samples[index]
        mode = "L" if self.input_channels == 1 else "RGB"
        with Image.open(image_path) as source:
            source_width, source_height = source.size
            image = source.convert(mode).resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BILINEAR,
            )
        array = np.asarray(image, dtype=np.float32) / 255.0

        label_path = self.labels_dir / f"{image_path.stem}.txt"
        rows: list[list[float]] = []
        if label_path.is_file():
            for raw_line in label_path.read_text(encoding="utf-8").splitlines():
                parts = raw_line.strip().split()
                if len(parts) >= 5:
                    rows.append([float(value) for value in parts[:5]])
        boxes, labels = yolo_normalized_to_xyxy(rows, image_size=self.image_size)
        array, boxes = _apply_train_augmentations(
            array,
            boxes,
            image_size=self.image_size,
            config=self.augment_config,
        )
        if self.input_channels == 1:
            tensor = torch.from_numpy(array).unsqueeze(0)
        else:
            tensor = torch.from_numpy(array).permute(2, 0, 1)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.long),
            "path": str(image_path),
            "orig_size": torch.tensor([source_height, source_width], dtype=torch.long),
            "input_size": torch.tensor([self.image_size, self.image_size], dtype=torch.long),
        }
        return tensor, target


def detection_collate_fn(
    batch: Sequence[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets)


def _apply_train_augmentations(
    array: np.ndarray,
    boxes: torch.Tensor,
    *,
    image_size: int,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, torch.Tensor]:
    if not config:
        return array, boxes
    size = float(image_size)
    augmented = array
    out_boxes = boxes.clone()

    flip_prob = float(config.get("horizontal_flip_prob", 0.0))
    if flip_prob > 0.0 and np.random.random() < flip_prob:
        augmented = np.ascontiguousarray(augmented[:, ::-1])
        if out_boxes.numel() > 0:
            x1 = out_boxes[:, 0].clone()
            x2 = out_boxes[:, 2].clone()
            out_boxes[:, 0] = size - x2
            out_boxes[:, 2] = size - x1

    brightness = float(config.get("brightness", 0.0))
    if brightness > 0.0:
        delta = np.random.uniform(-brightness, brightness)
        augmented = augmented + delta

    contrast = float(config.get("contrast", 0.0))
    if contrast > 0.0:
        scale = np.random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
        mean = float(np.mean(augmented))
        augmented = (augmented - mean) * scale + mean

    noise_std = float(config.get("noise_std", 0.0))
    if noise_std > 0.0:
        augmented = augmented + np.random.normal(0.0, noise_std, size=augmented.shape)

    return np.clip(augmented, 0.0, 1.0).astype(np.float32, copy=False), out_boxes.clamp(min=0.0, max=size)


def _read_yolo_label_ids(label_path: Path) -> list[int]:
    labels: list[int] = []
    if not label_path.is_file():
        return labels
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if parts:
            labels.append(int(float(parts[0])))
    return labels


def _build_class_balanced_sampler(dataset: YoloDetectionDataset) -> WeightedRandomSampler:
    counts: dict[int, int] = {}
    for labels in dataset.sample_labels:
        for label in set(labels):
            counts[label] = counts.get(label, 0) + 1
    weights = []
    for labels in dataset.sample_labels:
        if not labels or not counts:
            weights.append(1.0)
            continue
        weights.append(max(1.0 / max(1, counts[label]) for label in set(labels)))
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def create_yolo_detection_dataloader(
    data_dir: str | Path,
    *,
    split: str,
    batch_size: int = 1,
    image_size: int = 640,
    input_channels: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle: bool = False,
    seed: int | None = None,
    max_samples: int | None = None,
    augment_config: Mapping[str, Any] | None = None,
    class_balanced_sampler: bool = False,
) -> DataLoader:
    dataset = YoloDetectionDataset(
        data_dir,
        split=split,
        image_size=image_size,
        input_channels=input_channels,
        max_samples=max_samples,
        augment_config=augment_config,
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    sampler = _build_class_balanced_sampler(dataset) if class_balanced_sampler else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=detection_collate_fn,
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_dataloader_worker if num_workers > 0 else None,
        generator=generator,
    )



# make_loader preserves scripts/detection_controlled_study.py's protocol.
def make_loader(config, data_dir, split, seed, workers=0, train=False):
    if split not in {"train", "valid"}:
        raise ValueError("Only train/valid are permitted; test is sealed")
    dc,tc = config["dataset"],config["training"]
    loader = create_yolo_detection_dataloader(data_dir,split=split,
        image_size=dc["image_size"],input_channels=dc["input_channels"],
        batch_size=tc["batch_size"],num_workers=workers,shuffle=train,seed=seed,
        augment_config=tc.get("detection_augmentation",{}) if train else None,
        class_balanced_sampler=train and tc.get("detection_sampling",{}).get("class_balanced",False))
    if train and hasattr(loader.sampler,"generator"):
        loader.sampler.generator = torch.Generator().manual_seed(seed)
    return loader


def bind_dataset(loader):
    """Content-bound image/label manifest; paths relative to explicit data root."""
    from hashlib import sha256
    from space import canonical_hash
    dataset = loader.dataset
    entries = []
    for path in dataset.samples:
        label = dataset.labels_dir / (path.stem+".txt")
        read_labels(label,len(dataset.class_names))
        entries.append(dict(image=path.relative_to(dataset.root).as_posix(),
            image_sha256=sha256(path.read_bytes()).hexdigest(),
            annotation=label.relative_to(dataset.root).as_posix(),
            annotation_sha256=sha256(label.read_bytes()).hexdigest()))
    return dict(split=dataset.split,images=len(entries),entries=entries,
                class_names=dataset.class_names,sha256=canonical_hash(entries))


def prepare_calibration(config, data_root=None, *, images=64, seed=42):
    """Sorted first-64 train images with no augmentation or class sampler."""
    if type(images) is not int or images != 64:
        raise ValueError("Frozen PTQ requires exactly first 64 train images")
    root = data_root if data_root is not None else config["dataset"]["data_dir"]
    dc = config["dataset"]
    loader = create_yolo_detection_dataloader(root,split="train",batch_size=1,
        image_size=dc["image_size"],input_channels=dc["input_channels"],
        num_workers=0,shuffle=False,seed=seed,max_samples=images,
        augment_config=None,class_balanced_sampler=False)
    if len(loader.dataset) != images:
        raise ValueError("Not enough train images for the frozen first-64 calibration")
    return loader, bind_dataset(loader)


def read_labels(path, num_classes):
    """Strict compatible reader: empty files are negatives; missing files are errors."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing annotation (not an explicit negative): {path}')
    rows = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = [float(x) for x in line.split()]
            if len(row) != 5 or not np.isfinite(row).all():
                raise ValueError('expected five finite values: class cx cy w h')
            c, x, y, w, h = row
            if c != int(c) or not 0 <= c < num_classes:
                raise ValueError('invalid class id')
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError('invalid normalized geometry')
            if min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
                raise ValueError('bbox outside image; refusing silent clipping')
        except ValueError as exc:
            raise ValueError(f'{path}:{line_number}: {exc}') from exc
        rows.append(row)
    return rows
