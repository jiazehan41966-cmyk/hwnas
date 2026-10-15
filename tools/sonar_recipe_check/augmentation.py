"""Controlled additive YOLOv9 mosaic/affine augmentation for grayscale boxes.

The frozen loader still owns image decoding, 640-square resize, sampling,
flipping and photometric changes. Its four outputs are composed using the
pinned author's mosaic placement and random_perspective box filtering. Values
are passed as float32 in [0,255], avoiding an extra uint8 quantization step.
Copy-Paste is deliberately disabled: the source labels contain no segments.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PARAMETERS = dict(mosaic_probability=1.0, translate=0.1, scale=0.9,
                  degrees=0.0, shear=0.0, perspective=0.0, copy_paste=0.0,
                  extra_hsv=False, mixup=0.0, multiscale_input=False,
                  background=114)


def compose_mosaic(samples, size, center):
    """Pinned load_mosaic four-tile placement; targets are pixel xyxy boxes."""
    if len(samples) != 4:
        raise ValueError('A mosaic requires exactly four samples')
    yc, xc = center
    if not size // 2 <= yc <= 3 * size // 2 or not size // 2 <= xc <= 3 * size // 2:
        raise ValueError('Mosaic center outside registered range')
    canvas = np.full((size * 2, size * 2), 114.0, dtype=np.float32)
    labels = []
    for index, (tensor, target) in enumerate(samples):
        if tuple(tensor.shape) != (1, size, size):
            raise ValueError('Mosaic inputs must retain frozen grayscale square geometry')
        im = tensor[0].numpy() * np.float32(255.0)
        h, w = im.shape
        if index == 0:
            x1a, y1a, x2a, y2a = max(xc-w, 0), max(yc-h, 0), xc, yc
            x1b, y1b, x2b, y2b = w-(x2a-x1a), h-(y2a-y1a), w, h
        elif index == 1:
            x1a, y1a, x2a, y2a = xc, max(yc-h, 0), min(xc+w, size*2), yc
            x1b, y1b, x2b, y2b = 0, h-(y2a-y1a), min(w, x2a-x1a), h
        elif index == 2:
            x1a, y1a, x2a, y2a = max(xc-w, 0), yc, xc, min(size*2, yc+h)
            x1b, y1b, x2b, y2b = w-(x2a-x1a), 0, w, min(y2a-y1a, h)
        else:
            x1a, y1a, x2a, y2a = xc, yc, min(xc+w, size*2), min(size*2, yc+h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a-x1a), min(h, y2a-y1a)
        canvas[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]
        boxes = target['boxes'].numpy().copy()
        boxes[:, [0, 2]] += x1a-x1b
        boxes[:, [1, 3]] += y1a-y1b
        labels.append(np.column_stack((target['labels'].numpy(), boxes)).astype(np.float32))
    labels = np.concatenate(labels, axis=0)
    np.clip(labels[:, 1:], 0, 2*size, out=labels[:, 1:])
    return canvas, labels


def author_random_perspective(source_root):
    """Model construction must have loaded the pinned author module namespace."""
    module = importlib.import_module('utils.augmentations')
    if Path(module.__file__).resolve() != (Path(source_root) / 'utils/augmentations.py').resolve():
        raise ValueError('Wrong author augmentation module imported')
    return module.random_perspective


class MosaicDataset(Dataset):
    def __init__(self, base, source_root, parameters=None, *, perspective_fn=None):
        if parameters is not None and parameters != PARAMETERS:
            raise ValueError('Unregistered mosaic/affine parameters')
        if getattr(base, 'split', None) != 'train' or getattr(base, 'input_channels', None) != 1:
            raise ValueError('Additive augmentation is train-only and grayscale-only')
        self.base = base
        self.image_size = base.image_size
        self.samples = base.samples
        self.random_perspective = perspective_fn or author_random_perspective(source_root)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        size = self.image_size
        # Same random selection and placement order as pinned load_mosaic.
        center = tuple(int(random.uniform(size/2, 3*size/2)) for _ in range(2))
        indices = [index] + random.choices(range(len(self.base)), k=3)
        random.shuffle(indices)
        samples = [self.base[item] for item in indices]
        image, labels = compose_mosaic(samples, size, center)
        image, labels = self.random_perspective(image, labels, segments=(),
            degrees=0., translate=.1, scale=.9, shear=0., perspective=0.,
            border=(-size//2, -size//2))
        if image.shape != (size, size) or not np.isfinite(image).all():
            raise ValueError('Invalid mosaic/affine image')
        if (not np.isfinite(labels).all() or (labels[:, 1:] < 0).any()
                or (labels[:, 1:] > size).any()
                or (labels[:, 3:5] <= labels[:, 1:3]).any()):
            raise ValueError('Invalid mosaic/affine bbox')
        anchor = next(target for item, (_, target) in zip(indices, samples) if item == index)
        target = dict(anchor, boxes=torch.from_numpy(labels[:, 1:].copy()).float(),
                      labels=torch.from_numpy(labels[:, 0].copy()).long(),
                      mosaic_sources=[str(self.base.samples[item]) for item in indices])
        return torch.from_numpy(np.ascontiguousarray(image/np.float32(255.0))).unsqueeze(0), target


def wrap_loader(loader, source_root, parameters=None):
    if loader.num_workers != 0:
        raise ValueError('Frozen workers=0 is required for RNG replay')
    dataset = MosaicDataset(loader.dataset, source_root, parameters)
    return DataLoader(dataset, batch_size=loader.batch_size, sampler=loader.sampler,
                      drop_last=loader.drop_last, num_workers=0,
                      collate_fn=loader.collate_fn, pin_memory=loader.pin_memory,
                      generator=loader.generator)
