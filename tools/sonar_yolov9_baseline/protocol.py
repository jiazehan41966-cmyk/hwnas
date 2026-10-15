"""Shared, train/valid-only contracts for the L1 and author YOLOv9-T control.

The frozen project loader supplies a single grayscale channel. Three-channel
author models receive identical copies after the frozen augmentations. Native
detector loss and assignment remain the responsibility of the author adapter.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts'),
                str(ROOT / 'tools/sonar_analysis')]
from tools.sonar_analysis import baseline as frozen

BASELINE_PATH = ROOT / 'configs/baseline_sonar_frozen.yaml'
BASELINE_SHA256 = '23989d09cae29139be7bd74a92bf9c53f75605047b7d8b2d2e2601b98d2eb56e'
FREEZE_PATH = ROOT / 'reports/baseline_sonar_frozen/freeze_manifest.json'
FREEZE_SHA256 = '450c4db58045d241753f7ed8c8bd0e8ef34b3941d52fe6153917d6d33db4f074'
METADATA_SHA256 = '9290b9b13a247e106d39f49461b95590fea752760ef3ab8e85f084392dbea85a'
SPLIT_SHA256 = {
    'train': 'f74ae7623ba47a260b25646c698d7cb9eb09ee45a2e87a4e85a27c5d60f4341e',
    'valid': 'c9b3bf58a0734cabb2be6845eb0a84602808861739a47d960222ff7ca5592ef0',
}
SPLIT_COUNTS = {'train': 330, 'valid': 99}
CLASS_NAMES = ['cylider', 'manta']
SEEDS = (42, 43, 44)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def dump(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def load_baseline() -> dict:
    if sha(BASELINE_PATH) != BASELINE_SHA256 or sha(FREEZE_PATH) != FREEZE_SHA256:
        raise ValueError('Frozen baseline config or manifest hash drift')
    baseline = yaml.safe_load(BASELINE_PATH.read_text(encoding='utf-8'))
    dataset, training, evaluation = (baseline[key] for key in
                                     ('dataset', 'training', 'evaluation_protocol'))
    if (dataset['image_size'], dataset['input_channels'], dataset['num_classes'],
            dataset['class_names']) != (640, 1, 2, CLASS_NAMES):
        raise ValueError('Frozen grayscale geometry or class contract drift')
    expected = dict(epochs=80, optimizer='adamw', lr=.001, weight_decay=.0001,
                    batch_size=4, amp=False, scheduler='none')
    if any(training[key] != value for key, value in expected.items()):
        raise ValueError('Frozen training budget or optimizer drift')
    if (evaluation['development_splits'] != ['train', 'valid'] or
            evaluation['checkpoint_selection_metric'] != 'valid mAP50_95' or
            evaluation['checkpoint_tie_break'] != 'earliest epoch'):
        raise ValueError('Frozen development split or checkpoint selection drift')
    return baseline


def make_loaders(baseline: dict, seed: int, *, workers: int = 0):
    if seed not in SEEDS or workers != 0:
        raise ValueError('Use registered seeds and frozen workers=0')
    data = ROOT / baseline['dataset']['data_dir']
    return (
        frozen.make_loader(baseline, data, 'train', seed, workers, train=True),
        frozen.make_loader(baseline, data, 'valid', seed, workers),
    )


def _normalize_entries(entries: list[dict]) -> list[dict]:
    return [{key: (value.replace('\\', '/') if key in ('image', 'annotation') else value)
             for key, value in entry.items()} for entry in entries]


def verify_inputs(baseline: dict | None = None) -> dict:
    """Hash exact frozen train/valid bytes without loading or inspecting test."""
    baseline = load_baseline() if baseline is None else baseline
    original = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
    original_sources = {key.replace('\\', '/'): value
                        for key, value in original['source_sha256'].items()}
    for relative, expected in original_sources.items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Historical frozen source drift: {relative}')
    metadata = ROOT / baseline['dataset']['data_dir'] / 'data.yaml'
    if sha(metadata) != METADATA_SHA256:
        raise ValueError('Dataset class-name metadata hash drift')
    datasets = {}
    for split, expected_count in SPLIT_COUNTS.items():
        loader = frozen.make_loader(baseline, ROOT / baseline['dataset']['data_dir'],
                                    split, 42, 0)
        actual = frozen.bind_dataset(loader)
        actual['entries'] = _normalize_entries(actual['entries'])
        actual['sha256'] = canonical_sha(actual['entries'])
        if (actual['images'] != expected_count or actual['class_names'] != CLASS_NAMES or
                actual['entries'] != _normalize_entries(original['datasets'][split]['entries']) or
                actual['sha256'] != SPLIT_SHA256[split]):
            raise ValueError(f'Frozen dataset content, order, or labels drift: {split}')
        datasets[split] = actual
    # Later independent modules do not invalidate bytes that were frozen earlier.
    # This is an explicit content check, not a claim that the old glob gate passed.
    return dict(historical_frozen_files_verified=len(original_sources),
                historical_source_gate='FROZEN_BYTES_VERIFIED_NO_GLOB_EQUALITY_CLAIM',
                datasets=datasets, dataset_metadata_sha256=sha(metadata),
                test_performance_accessed=False)


def gray_to_three(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 640, 640):
        raise ValueError('Expected frozen grayscale batch [B,1,640,640]')
    if not torch.isfinite(images).all() or (images < 0).any() or (images > 1).any():
        raise ValueError('Grayscale pixels must be finite in [0,1]')
    return images.repeat(1, 3, 1, 1)


def targets_to_yolo(targets: list[dict], *, image_size: int = 640,
                    device: str | torch.device = 'cpu') -> torch.Tensor:
    """Author loss targets: image index, class index, normalized cx/cy/w/h."""
    if image_size != 640:
        raise ValueError('The comparison requires 640 input coordinates')
    rows = []
    for index, target in enumerate(targets):
        boxes = torch.as_tensor(target['boxes'], dtype=torch.float32, device=device).reshape(-1, 4)
        labels = torch.as_tensor(target['labels'], dtype=torch.long, device=device).reshape(-1)
        if len(boxes) != len(labels) or not torch.isfinite(boxes).all():
            raise ValueError('Invalid detection target rows')
        if len(boxes) == 0:
            continue
        if (labels < 0).any() or (labels >= 2).any() or (boxes < 0).any() or (boxes > 640).any():
            raise ValueError('Target class or coordinate range violation')
        size = boxes[:, 2:] - boxes[:, :2]
        if (size <= 0).any():
            raise ValueError('Target boxes must have positive width and height')
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        rows.append(torch.cat((torch.full((len(boxes), 1), index, device=device),
                               labels[:, None], center / image_size, size / image_size), dim=1))
    return torch.cat(rows, dim=0) if rows else torch.empty((0, 6), dtype=torch.float32, device=device)


def records_from_batch(predictions: list[dict], targets: list[dict],
                       data_root: str | Path) -> list[dict]:
    if len(predictions) != len(targets):
        raise ValueError('Prediction/target image count mismatch')
    records = []
    for prediction, target in zip(predictions, targets):
        relative = Path(target['path']).relative_to(Path(data_root)).as_posix()
        if relative.split('/')[0] not in ('train', 'valid'):
            raise ValueError('Records may contain train/valid images only')
        converted = {key: torch.as_tensor(prediction[key]).detach().cpu()
                     for key in ('boxes', 'scores', 'labels')}
        converted['boxes'] = converted['boxes'].reshape(-1, 4)
        converted['scores'] = converted['scores'].reshape(-1)
        converted['labels'] = converted['labels'].reshape(-1)
        if len({len(value) for value in converted.values()}) != 1:
            raise ValueError('Prediction boxes/scores/labels count mismatch')
        if not all(torch.isfinite(value).all() for value in converted.values()):
            raise ValueError('Nonfinite prediction record')
        if (converted['labels'] < 0).any() or (converted['labels'] >= 2).any():
            raise ValueError('Prediction labels must use frozen 0/1 indices')
        if (converted['scores'] < 0).any() or (converted['scores'] > 1).any():
            raise ValueError('Prediction scores must lie in [0,1]')
        if len(converted['scores']) > 100:
            raise ValueError('Frozen post-NMS limit is global 100 per image')
        records.append(dict(image=relative,
                            prediction={key: value.tolist() for key, value in converted.items()},
                            target={key: torch.as_tensor(target[key]).detach().cpu().tolist()
                                    for key in ('boxes', 'labels')}))
    return records


def metrics_from_records(records: list[dict], baseline: dict) -> dict:
    if not records:
        raise ValueError('Empty validation records')
    return frozen.metrics_from_records(records, baseline)


def training_rng_identity(seed: int) -> dict:
    return dict(seed=seed,
                torch_cpu_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                numpy_rng_sha256=canonical_sha(np.random.get_state()[1].tolist()),
                python_rng_sha256=canonical_sha(random.getstate()),
                policy='reset_after_construction_before_loaders')


def rng_state(loader) -> dict:
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                sampler_generator=loader.sampler.generator.get_state()
                    if getattr(loader.sampler, 'generator', None) is not None else None,
                loader_generator=loader.generator.get_state() if loader.generator is not None else None)


def restore_rng(state: dict, loader) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    if state['sampler_generator'] is not None:
        loader.sampler.generator.set_state(state['sampler_generator'])
    if state['loader_generator'] is not None:
        loader.generator.set_state(state['loader_generator'])


def atomic_save(path: str | Path, value: Any) -> None:
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


seed_all = frozen.seed_all
state_hash = frozen.state_hash
