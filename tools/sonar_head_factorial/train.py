"""Frozen train/valid-only, scratch 2x2 backbone-neck x head-package experiment.

No promotion or NAS execution is performed by this runner. The decision is a
paired validation comparison, not an independent test-generalization claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import importlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OVERLAY = ROOT / 'results/sonar_yolov9_baseline_20261005/env_overlay'
if OVERLAY.is_dir():
    sys.path.insert(0, str(OVERLAY))
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts')]
os.environ.update(YOLOv5_AUTOINSTALL='false', YOLOM='10', YOLOA='0.5', YOLOB='6.0')
import numpy as np
import torch
import yaml

from tools.sonar_yolov9_baseline import protocol as p

CONFIG = ROOT / 'configs/sonar_head_factorial_20261012.yaml'
BACKBONES = ('E0', 'YOLOv9_T')
HEADS = ('obj_softmax', 'quality_aware')
SEEDS = (42, 43, 44)  # Common seeds used for backbone effects and interaction.
ALL_SEEDS = (42, 43, 44, 45, 46)
SEEDS_BY_BACKBONE = {'E0': ALL_SEEDS, 'YOLOv9_T': SEEDS}
STABLE_EPOCHS = (76, 77, 78, 79, 80)
VERSION = 'sonar_head_factorial_20261012_02_stable'
SUBSET_CONTRACT = 'results/experiment0_similarity_supplement_20261012_01/subset_contract.json'
METRICS = ('mAP50_95', 'AP50', 'AP75', 'Precision', 'Recall', 'map')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_config(config=CONFIG):
    cfg = yaml.safe_load(Path(config).read_text(encoding='utf-8'))
    baseline = p.load_baseline()
    expected = dict(epochs=80, batch_size=4, input=640, optimizer='AdamW',
                    lr=.001, weight_decay=.0001, scheduler='none', amp=False,
                    workers=0, ema=False, accumulation=1,
                    checkpoint_selection='earliest_best_valid_mAP50_95_secondary_only')
    if (cfg['version'] != VERSION or cfg['backbones'] != list(BACKBONES)
            or cfg['heads'] != list(HEADS)
            or cfg['seeds_by_backbone'] != {key: list(value) for key, value in SEEDS_BY_BACKBONE.items()}
            or cfg['total_formal_runs'] != 16
            or cfg['result_root'] != f'results/{VERSION}/remote'
            or cfg['subset_contract'] != SUBSET_CONTRACT
            or any(cfg['training'].get(key) != value for key, value in expected.items())
            or cfg['baseline'] != p.BASELINE_PATH.relative_to(ROOT).as_posix()):
        raise ValueError('Unregistered factorial protocol or frozen optimizer/budget drift')
    expected_eval = dict(splits=['train', 'valid'], ap_score_threshold=0.,
                         precision_recall_score_threshold=.5, pre_nms_topk=500,
                         nms_iou_threshold=.5, max_detections=100,
                         class_agnostic_nms=False)
    if cfg['evaluation'] != expected_eval:
        raise ValueError('Frozen decode/metric/split contract drift')
    if (cfg['decision']['metric'] != 'mAP50_95'
            or cfg['decision']['metric_definition'] != 'COCO101_mean_AP50_to_AP95'
            or cfg['decision']['primary_checkpoint'] != 'mean_epochs_76_80'
            or cfg['decision']['primary_epochs'] != list(STABLE_EPOCHS)
            or cfg['decision']['primary_subset'] != 'full99'
            or cfg['decision']['report_subsets'] != ['full99', 'without_similar89']
            or 'extension' in cfg['decision']
            or cfg['decision']['positivity'] != 'hierarchical_paired_seed_and_image_CI95_lower_strictly_positive'
            or cfg['decision']['bootstrap']['confidence'] != .95
            or cfg['decision']['bootstrap']['method'] != 'hierarchical_paired_seed_and_image_percentile'
            or cfg['decision']['bootstrap']['pairing_units'] != ['training_seed', 'validation_image']
            or cfg['decision']['bootstrap']['replicates'] != 2000
            or cfg['decision']['bootstrap']['random_seed'] != 20261012
            or cfg['decision']['threshold_absolute'] != .03
            or cfg['decision']['eligible_split'] != 'valid'
            or cfg['decision']['require_all_16_complete'] is not True
            or cfg['decision']['automatic_promotion'] is not False
            or cfg['decision']['automatic_search'] is not False
            or cfg['training']['save_prediction_epochs'] != list(STABLE_EPOCHS)
            or cfg['training']['stable_prediction_image_count'] != 99
            or cfg['training']['missing_stable_epoch_policy'] != 'fail_closed'
            or cfg['queue']['adaptive_seed_extension'] is not False):
        raise ValueError('Decision metric or threshold is not registered')
    load_subset_contract(cfg)
    return cfg, baseline


def _inside_root(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Source reference must remain inside the project')
    return path


def load_subset_contract(cfg, input_identity=None):
    """Bind the frozen auxiliary view without changing validation or selection."""
    path = _inside_root(cfg['subset_contract'])
    if p.sha(path) != cfg['subset_contract_sha256']:
        raise ValueError('Frozen evaluation subset contract hash drift')
    contract = read_json(path)
    if (contract.get('schema_version') != 1 or contract.get('source_split') != 'valid'
            or contract.get('original_split_changed') is not False
            or contract.get('selection_uses_model_predictions') is not False
            or contract.get('selection_uses_performance') is not False
            or contract.get('original_checkpoint_selection_changed') is not False):
        raise ValueError('Subset contract changes the registered evaluation-only scope')
    groups = {}
    for key, count in [('full_images', 99), ('kept_images', 89), ('excluded_images', 10)]:
        values = contract.get(key, [])
        if (len(values) != count or len(set(values)) != count or values != sorted(values)
                or any(not isinstance(name, str) or Path(name).name != name or '\\' in name for name in values)):
            raise ValueError(f'Invalid frozen image membership: {key}')
        groups[key] = set(values)
    if (groups['kept_images'] & groups['excluded_images']
            or groups['full_images'] != groups['kept_images'] | groups['excluded_images']):
        raise ValueError('Subset images must partition the exact original validation images')
    for key in ('source_pairs', 'source_full_image_order', 'source_label_inventory'):
        if p.sha(_inside_root(contract[key])) != contract[key + '_sha256']:
            raise ValueError(f'Frozen subset source drift: {key}')
    if input_identity is not None:
        names = {Path(row['image']).name for row in input_identity['datasets']['valid']['entries']}
        if names != groups['full_images']:
            raise ValueError('Subset full99 differs from the frozen validation dataset')
    return contract


def source_paths(cfg, config=CONFIG):
    """Source closure usable by staging without importing the model adapter."""
    paths = set((ROOT / 'src/hwnas_fpga').rglob('*.py'))
    for directory in ('tools/sonar_head_factorial', 'tools/sonar_yolov9_baseline',
                      'tools/sonar_spec', 'tools/sonar_analysis',
                      'tools/detector_b1_light', 'tools/detector_elastic_pilot'):
        paths.update((ROOT / directory).glob('*.py'))
    original = read_json(p.FREEZE_PATH)
    paths.update(ROOT / name.replace('\\', '/') for name in original['source_sha256'])
    author = ROOT / cfg['author']['source_root']
    for pattern in ('*.py', '*.yaml', '*.yml'):
        paths.update(author.rglob(pattern))
    paths.update((Path(config), ROOT / cfg['environment_spec'], p.BASELINE_PATH,
                  p.FREEZE_PATH, ROOT / 'configs/detector_b1_light_model.yaml',
                  ROOT / 'configs/detector_elastic_pilot_model.yaml',
                  ROOT / 'configs/search_space_sonar_v0_1.yaml',
                  author / 'LICENSE.md'))
    subset = load_subset_contract(cfg)
    paths.add(ROOT / cfg['subset_contract'])
    paths.update(_inside_root(subset[key]) for key in
                 ('source_pairs', 'source_full_image_order', 'source_label_inventory'))
    paths.add(ROOT / 'tools/sonar_head_factorial/stable_analysis.py')
    paths.add(ROOT / 'tests/test_sonar_head_factorial_stable_analysis.py')
    for candidate in (author / 'source_receipt.json',
                      ROOT / 'tests/test_sonar_head_factorial_model.py',
                      ROOT / 'tests/test_sonar_head_factorial_protocol.py'):
        if candidate.is_file():
            paths.add(candidate)
    return sorted(paths)


def source_hashes(cfg, config=CONFIG):
    return {path.relative_to(ROOT).as_posix(): p.sha(path)
            for path in source_paths(cfg, config)}


def frozen_validation_targets(baseline):
    """Read GT through the existing frozen loader, without a model or new transform."""
    loader = p.frozen.make_loader(baseline, ROOT / baseline['dataset']['data_dir'],
                                  'valid', 42, 0)
    targets = []
    for _, batch_targets in loader:
        for target in batch_targets:
            targets.append(dict(image=Path(target['path']).name,
                target={key: torch.as_tensor(target[key]).detach().cpu().tolist()
                        for key in ('boxes', 'labels')}))
    targets.sort(key=lambda row: row['image'])
    if len(targets) != 99 or len({row['image'] for row in targets}) != 99:
        raise ValueError('Frozen loader did not provide all 99 distinct validation targets')
    return targets


def prepare(root, config=CONFIG):
    cfg, baseline = load_config(config)
    author = ROOT / cfg['author']['source_root']
    if (author / '.git').exists():
        commit = subprocess.check_output(['git', '-C', str(author), 'rev-parse', 'HEAD'],
                                         text=True).strip()
    else:
        commit = read_json(author / 'source_receipt.json')['commit']
    if commit != cfg['author']['commit']:
        raise ValueError('Pinned author commit drift')
    inputs = p.verify_inputs(baseline)
    subset = load_subset_contract(cfg, inputs)
    validation_targets = frozen_validation_targets(baseline)
    if [row['image'] for row in validation_targets] != subset['full_images']:
        raise ValueError('Frozen validation GT reference and subset image IDs differ')
    manifest = dict(version=cfg['version'], config=cfg, config_sha256=p.sha(config),
                    author_commit=commit, source_sha256=source_hashes(cfg, config),
                    inputs=inputs, subset_contract_sha256=cfg['subset_contract_sha256'],
                    validation_targets=validation_targets,
                    validation_target_sha256=p.canonical_sha(validation_targets),
                    subset_id=subset['subset_id'], test_evaluated=False)
    path = root / 'execution_manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise ValueError('Existing freeze differs; preserve campaign and use a new result root')
    if not path.exists():
        p.dump(path, manifest)
    return manifest


def verify_manifest(root, config=CONFIG):
    cfg, baseline = load_config(config)
    manifest = read_json(root / 'execution_manifest.json')
    if (manifest['config'] != cfg or manifest['config_sha256'] != p.sha(config)
            or manifest['source_sha256'] != source_hashes(cfg, config)
            or manifest['subset_contract_sha256'] != cfg['subset_contract_sha256']
            or manifest['validation_target_sha256'] != p.canonical_sha(manifest['validation_targets'])
            or manifest['inputs'] != p.verify_inputs(baseline)):
        raise ValueError('Frozen source/config/data identity drift')
    subset = load_subset_contract(cfg, manifest['inputs'])
    if [row['image'] for row in manifest['validation_targets']] != subset['full_images']:
        raise ValueError('Frozen validation target reference differs from subset full99')
    return cfg, baseline


def environment(device, cfg):
    spec = yaml.safe_load((ROOT / cfg['environment_spec']).read_text(encoding='utf-8'))
    packages, imported = {}, {}
    modules = {'torch': 'torch', 'torchvision': 'torchvision', 'numpy': 'numpy',
               'Pillow': 'PIL', 'PyYAML': 'yaml', 'opencv-python-headless': 'cv2',
               'ipython': 'IPython', 'seaborn': 'seaborn', 'requests': 'requests',
               'psutil': 'psutil', 'thop': 'thop', 'setuptools': 'setuptools',
               'matplotlib': 'matplotlib', 'pandas': 'pandas', 'scipy': 'scipy'}
    for package, module in modules.items():
        packages[package] = importlib.metadata.version(package)
        item = importlib.import_module(module)
        location = Path(item.__file__).resolve()
        imported[module] = dict(path=str(location), sha256=p.sha(location),
                                module_version=getattr(item, '__version__', None))
    overlay_path = Path(spec['overlay_source']) if os.name != 'nt' else OVERLAY
    if not overlay_path.is_dir():
        raise FileNotFoundError(f'Existing common overlay is absent: {overlay_path}')
    overlay = {path.relative_to(overlay_path).as_posix(): p.sha(path)
                   for path in sorted(overlay_path.rglob('*'))
                   if path.is_file() and '__pycache__' not in path.parts
                   and path.suffix not in ('.pyc', '.pyo')}
    value = dict(python=platform.python_version(), interpreter=str(Path(sys.executable).resolve()),
                 packages=packages, imported=imported, overlay_sha256=p.canonical_sha(overlay),
                 overlay_files=len(overlay), device=device, cuda_runtime=torch.version.cuda,
                 cudnn_version=torch.backends.cudnn.version(),
                 cudnn_deterministic=torch.backends.cudnn.deterministic,
                 cudnn_benchmark=torch.backends.cudnn.benchmark,
                 deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                 deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                 matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                 cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                 float32_matmul_precision=torch.get_float32_matmul_precision(),
                 torch_threads=torch.get_num_threads(), amp=False, workers=0)
    if device == 'cuda':
        value.update(gpu=torch.cuda.get_device_name(0),
                     gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
    expected = spec['expected']
    actual = dict(python=value['python'], torch=packages['torch'], numpy=packages['numpy'],
                  cuda_runtime=value['cuda_runtime'])
    if (actual != expected or any(packages[key] != val for key, val in spec['packages'].items())
            or value['torch_threads'] != spec['threads']
            or (device == 'cuda' and value['gpu'] != spec['gpu'])):
        raise ValueError(f'Actual execution environment differs from pinned spec: {actual}')
    return value


def bind_environment(root, device, cfg):
    actual = environment(device, cfg)
    path = root / 'environment_lock.json'
    lock = dict(fingerprint=p.canonical_sha(actual), environment=actual)
    if path.exists():
        if read_json(path) != lock:
            raise ValueError('Common environment fingerprint differs across factorial runs')
    else:
        # First smoke is sequential; formal queues may only read an existing lock.
        p.dump(path, lock)
    return lock


def arm(backbone, head):
    return f'{backbone}__{head}'


def result_dir(root, backbone, head, seed, mode):
    return root / mode / arm(backbone, head) / f'seed_{seed}'


def binding(root, backbone, head, seed, mode):
    return dict(execution_manifest_sha256=p.sha(root / 'execution_manifest.json'),
                environment_fingerprint=read_json(root / 'environment_lock.json')['fingerprint'],
                subset_contract_sha256=read_json(root / 'execution_manifest.json')['subset_contract_sha256'],
                validation_target_sha256=read_json(root / 'execution_manifest.json')['validation_target_sha256'],
                backbone=backbone, head=head, seed=seed, mode=mode, test_evaluated=False)


def make_run(root, backbone, head, seed, device, mode, config=CONFIG):
    from tools.sonar_head_factorial import model as adapter
    cfg, baseline = verify_manifest(root, config)
    if (backbone not in BACKBONES or head not in HEADS
            or seed not in SEEDS_BY_BACKBONE.get(backbone, ())):
        raise ValueError('Unregistered factorial run')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent fallback')
    p.seed_all(seed)
    lock = bind_environment(root, device, cfg)
    detector = adapter.build_model(backbone, head, seed,
                                   source_root=ROOT / cfg['author']['source_root']).to(device)
    criterion = adapter.build_criterion(detector, baseline)
    identity = detector.initialization_identity()
    metadata = detector.metadata()
    metadata['criterion'] = adapter.criterion_metadata(detector, criterion)
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    initial_sha = p.state_hash(detector)
    # Construction consumes no sampler or augmentation RNG in the comparison.
    p.seed_all(seed)
    rng_identity = p.training_rng_identity(seed)
    data = ROOT / baseline['dataset']['data_dir']
    train = p.frozen.make_loader(baseline, data, 'train', seed, 0, train=True)
    valid = p.frozen.make_loader(baseline, data, 'valid', seed, 0)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=.001, weight_decay=.0001)
    folder = result_dir(root, backbone, head, seed, mode)
    folder.mkdir(parents=True, exist_ok=True)
    for filename, content in (('initialization_identity.json', identity),
                              ('model_info.json', metadata),
                              ('training_rng_identity.json', rng_identity)):
        target = folder / filename
        if target.exists() and read_json(target) != content:
            raise ValueError(f'Existing run initialization identity drift: {filename}')
        if not target.exists():
            p.dump(target, content)
    return folder, cfg, baseline, detector, criterion, optimizer, train, valid, initial_sha, identity, lock


def batch_digest(images, targets):
    """Hash actual augmented pixels, target tensors and ordered relative paths."""
    digest = hashlib.sha256()
    for tensor in [images] + [torch.as_tensor(target[key]) for target in targets
                             for key in ('boxes', 'labels')]:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    digest.update(p.canonical_sha([Path(target['path']).as_posix().split('/v6_yolov8/')[-1]
                                   for target in targets]).encode())
    return digest.hexdigest()


def train_epoch(model, criterion, optimizer, loader, device, max_batches=None):
    from tools.sonar_head_factorial import model as adapter
    model.train()
    totals, batches, hashes, image_order = Counter(), 0, [], []
    tick = time.perf_counter()
    iterator = iter(loader)
    if max_batches is not None:
        iterator = itertools.islice(iterator, max_batches)
    for images, targets in iterator:
        hashes.append(batch_digest(images, targets))
        image_order.extend(Path(target['path']).name for target in targets)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = adapter.batch_loss(model, criterion, images.to(device), targets)
        if not torch.isfinite(loss) or not all(math.isfinite(float(x)) for x in parts.values()):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not grads or any(not torch.isfinite(grad).all() for grad in grads):
            raise ValueError('Absent or nonfinite gradients')
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ValueError('Optimizer produced nonfinite parameters')
        batches += 1
        totals.update({key: float(value) for key, value in parts.items()})
    if not batches:
        raise ValueError('Empty training epoch')
    if device == 'cuda':
        torch.cuda.synchronize()
    return dict(batches=batches, images=len(image_order),
                image_order_sha256=p.canonical_sha(image_order),
                augmented_batches_sha256=p.canonical_sha(hashes), batch_sha256=hashes,
                losses={key: value / batches for key, value in totals.items()},
                gradient_finite=True, lr=optimizer.param_groups[0]['lr'],
                train_seconds=time.perf_counter() - tick)


@torch.no_grad()
def evaluate(model, criterion, loader, baseline, device, max_batches=None):
    from tools.sonar_head_factorial import model as adapter
    model.eval()
    records, hashes = [], []
    ep = baseline['evaluation_protocol']
    iterator = iter(loader)
    if max_batches is not None:
        iterator = itertools.islice(iterator, max_batches)
    for images, targets in iterator:
        hashes.append(batch_digest(images, targets))
        output = model(images.to(device))
        predictions = adapter.decode(model, criterion, output,
            score_threshold=ep['ap_decode_score_threshold'], pre_nms_topk=ep['pre_nms_topk'],
            nms_iou_threshold=ep['nms_iou_threshold'],
            max_detections=ep['post_nms_max_detections_per_image'])
        records.extend(p.records_from_batch(predictions, targets,
                                             ROOT / baseline['dataset']['data_dir']))
    return enrich_metrics(p.metrics_from_records(records, baseline)), records, p.canonical_sha(hashes)


def enrich_metrics(metrics):
    result = dict(metrics)
    result['map'] = (float(metrics['AP50']) + float(metrics['AP75'])) / 2
    result['per_class_AP'] = {}
    for label, values in metrics['coco_details']['per_class_ap']['all'].items():
        result['per_class_AP'][label] = dict(AP50=values['0.50'], AP75=values['0.75'],
            map=(values['0.50'] + values['0.75']) / 2,
            mAP50_95=statistics.mean(values.values()))
    return result


def smoke_one(root, backbone, head, seed, device, config=CONFIG):
    folder = result_dir(root, backbone, head, seed, 'smoke')
    if folder.exists():
        raise FileExistsError('Preserve existing smoke invocation; use a new result root')
    run = make_run(root, backbone, head, seed, device, 'smoke', config)
    folder, cfg, baseline, model, criterion, opt, train, valid, initial, identity, lock = run
    tick = time.perf_counter()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    try:
        trained = train_epoch(model, criterion, opt, train, device, 2)
        metrics, records, valid_sha = evaluate(model, criterion, valid, baseline, device, 1)
        final = p.state_hash(model)
        if initial == final or trained['batches'] != 2 or trained['images'] != 8 or metrics['num_images'] != 4:
            raise ValueError('Smoke lacks two optimizer steps and one full validation batch')
        report = dict(status='PASS', binding=binding(root, backbone, head, seed, 'smoke'),
                      training=trained, valid=metrics, valid_batches_sha256=valid_sha,
                      initialization=identity, initial_state_sha256=initial,
                      final_state_sha256=final, seconds=time.perf_counter() - tick,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
                      formal_eligible=False, test_evaluated=False)
        p.dump(folder / 'valid_records.json', records)
        p.dump(folder / 'smoke_report.json', report)
        return report
    except Exception as error:
        p.dump(folder / 'smoke_failure.json', dict(status='FAILED', error=str(error),
               error_type=type(error).__name__, formal_eligible=False, test_evaluated=False))
        raise
    finally:
        del run, model, criterion, opt, train, valid
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()


def verify_pairing(rows, *, smoke=False):
    for seed in sorted({row['binding']['seed'] for row in rows}):
        group = [row for row in rows if row['binding']['seed'] == seed]
        backbones = BACKBONES if smoke else tuple(backbone for backbone in BACKBONES
                                                 if seed in SEEDS_BY_BACKBONE[backbone])
        expected = set(itertools.product(backbones, HEADS))
        if (not expected or len(group) != len(expected)
                or {(r['binding']['backbone'], r['binding']['head']) for r in group} != expected):
            raise ValueError('Pairing requires the exact registered cells for each seed')
        if len({row['binding']['environment_fingerprint'] for row in group}) != 1:
            raise ValueError('Environment differs between cells')
        for backbone in backbones:
            pair = [row for row in group if row['binding']['backbone'] == backbone]
            for key in ('backbone_sha256', 'neck_sha256'):
                if pair[0]['initialization'][key] != pair[1]['initialization'][key]:
                    raise ValueError(f'Initial {key} differs between heads')
        witnesses = []
        for row in group:
            if smoke:
                witness = [row['training']['augmented_batches_sha256'], row['valid_batches_sha256']]
            else:
                witness = [(epoch['augmented_batches_sha256'], epoch['image_order_sha256'],
                            epoch['valid_batches_sha256']) for epoch in row['history']]
            witnesses.append(p.canonical_sha(witness))
        if len(set(witnesses)) != 1:
            raise ValueError('Actual augmented training pixels/labels/order differ between cells')


def smoke(root, device, config=CONFIG):
    verify_manifest(root, config)
    rows = [smoke_one(root, backbone, head, 42, device, config)
            for backbone, head in itertools.product(BACKBONES, HEADS)]
    verify_pairing(rows, smoke=True)
    gate = dict(status='PASS', execution_manifest_sha256=p.sha(root / 'execution_manifest.json'),
                environment_fingerprint=read_json(root / 'environment_lock.json')['fingerprint'],
                arms=[row['binding'] for row in rows], actual_two_train_batches_each=True,
                actual_one_valid_batch_each=True, identical_augmented_batches=True,
                paired_initial_backbone_neck=True, formal_eligible=False, test_evaluated=False)
    p.dump(root / 'smoke_gate.json', gate)
    return gate


def require_smoke(root):
    gate = read_json(root / 'smoke_gate.json')
    if (gate['status'] != 'PASS'
            or gate['execution_manifest_sha256'] != p.sha(root / 'execution_manifest.json')
            or gate['environment_fingerprint'] != read_json(root / 'environment_lock.json')['fingerprint']):
        raise ValueError('Four-cell smoke gate is absent or belongs to different source/environment')
    rows = []
    for backbone, head in itertools.product(BACKBONES, HEADS):
        report = read_json(result_dir(root, backbone, head, 42, 'smoke') / 'smoke_report.json')
        if report['status'] != 'PASS' or report['binding'] != binding(root, backbone, head, 42, 'smoke'):
            raise ValueError('Four-cell smoke artifact mismatch')
        rows.append(report)
    verify_pairing(rows, smoke=True)


def validate_history(history, epochs=80):
    try:
        invalid = (type(epochs) is not int or not 1 <= epochs <= 80
            or len(history) != epochs or [row['epoch'] for row in history] != list(range(1, epochs + 1))
            or any(row['valid']['num_images'] != 99 for row in history)
            or any(row['batches'] != 83 or row['images'] != 330 for row in history)
            or any(not math.isfinite(float(row['valid'][metric]))
                   or not 0 <= float(row['valid'][metric]) <= 1
                   for row in history for metric in METRICS))
    except (KeyError, TypeError, ValueError):
        invalid = True
    if invalid:
        raise ValueError('Incomplete, nonfinite or wrong-size train/validation history')


def validate_prediction_records(records, expected_images, expected_target_sha256):
    """Reject missing/duplicated images and malformed cached detection inputs."""
    if not isinstance(records, list) or len(records) != 99:
        raise ValueError('Stable predictions require all 99 validation image records')
    images, targets = [], []
    for row in records:
        source = row.get('image', '').replace('\\', '/')
        image = source.split('/')[-1]
        if not image or ('/' in source and not source.startswith('valid/images/')):
            raise ValueError('Stable prediction image must belong to the frozen validation split')
        images.append(image)
        for kind in ('prediction', 'target'):
            value = row[kind]
            boxes = np.asarray(value['boxes'], dtype=np.float64).reshape(-1, 4)
            labels = np.asarray(value['labels'])
            if (labels.ndim != 1 or len(boxes) != len(labels)
                    or not np.isfinite(boxes).all() or not np.isin(labels, [0, 1]).all()
                    or (boxes[:, 2:] < boxes[:, :2]).any()):
                raise ValueError('Malformed stable prediction boxes or labels')
            if kind == 'prediction':
                scores = np.asarray(value['scores'], dtype=np.float64)
                if (scores.ndim != 1 or len(scores) != len(boxes) or len(scores) > 100
                        or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()):
                    raise ValueError('Malformed stable prediction scores or global detection cap')
        targets.append(dict(image=image, target=row['target']))
    if len(set(images)) != 99 or set(images) != set(expected_images):
        raise ValueError('Stable predictions differ from the exact frozen full99 image set')
    target_sha256 = p.canonical_sha(sorted(targets, key=lambda value: value['image']))
    if target_sha256 != expected_target_sha256:
        raise ValueError('Cached validation GT differs from the frozen source-label reference')
    return dict(image_count=99, image_ids_sha256=p.canonical_sha(sorted(images)),
                target_sha256=target_sha256)


def save_stable_predictions(folder, epoch, records, expected_binding, expected_images):
    """Write immutable per-epoch predictions before committing their checkpoint."""
    if epoch not in STABLE_EPOCHS:
        raise ValueError('Only registered stable epochs may enter the prediction archive')
    identity = validate_prediction_records(records, expected_images, expected_binding['validation_target_sha256'])
    relative = f'epoch_predictions/epoch_{epoch:03d}_valid_records.json'
    receipt_relative = f'epoch_predictions/epoch_{epoch:03d}_receipt.json'
    path, receipt_path = folder / relative, folder / receipt_relative
    if path.exists():
        if read_json(path) != records:
            raise ValueError('Existing stable predictions differ; preserve uncommitted epoch evidence')
    else:
        p.dump(path, records)
    receipt = dict(epoch=epoch, binding=expected_binding, records=relative,
                   records_sha256=p.sha(path), **identity)
    if receipt_path.exists():
        if read_json(receipt_path) != receipt:
            raise ValueError('Existing stable prediction receipt differs')
    else:
        p.dump(receipt_path, receipt)
    return dict(records=relative, records_sha256=receipt['records_sha256'],
                receipt=receipt_relative, receipt_sha256=p.sha(receipt_path), **identity)


def verify_stable_predictions(folder, history, expected_binding, baseline, expected_images):
    """Verify every committed late epoch; no inference or replacement is allowed."""
    paths, hashes, metric_rows, identities = {}, {}, {}, []
    for row in history:
        epoch = row['epoch']
        if epoch not in STABLE_EPOCHS:
            continue
        archive = row.get('prediction_archive')
        expected_path = f'epoch_predictions/epoch_{epoch:03d}_valid_records.json'
        expected_receipt = f'epoch_predictions/epoch_{epoch:03d}_receipt.json'
        if (not isinstance(archive, dict) or archive.get('records') != expected_path
                or archive.get('receipt') != expected_receipt):
            raise ValueError(f'Missing stable prediction archive for epoch {epoch}')
        path, receipt_path = folder / expected_path, folder / expected_receipt
        if (not path.is_file() or not receipt_path.is_file()
                or p.sha(path) != archive['records_sha256']
                or p.sha(receipt_path) != archive['receipt_sha256']):
            raise ValueError(f'Stable epoch {epoch} prediction/receipt hash drift or missing file')
        records = read_json(path)
        identity = validate_prediction_records(records, expected_images, expected_binding['validation_target_sha256'])
        receipt = read_json(receipt_path)
        expected = dict(epoch=epoch, binding=expected_binding, records=expected_path,
                        records_sha256=p.sha(path), **identity)
        if (receipt != expected or any(archive.get(key) != value for key, value in identity.items())):
            raise ValueError(f'Stable epoch {epoch} source/image/target binding drift')
        metrics = enrich_metrics(p.metrics_from_records(records, baseline))
        if metrics != row['valid']:
            raise ValueError(f'Stable epoch {epoch} predictions do not reproduce its logged metrics')
        paths[str(epoch)], hashes[str(epoch)], metric_rows[str(epoch)] = str(path.resolve()), p.sha(path), metrics
        identities.append(identity)
    required = {str(epoch) for epoch in STABLE_EPOCHS if epoch <= len(history)}
    if set(paths) != required or (identities and any(value != identities[0] for value in identities)):
        raise ValueError('Stable prediction epochs or paired image/target identities are incomplete')
    return dict(paths=paths, sha256=hashes, metrics=metric_rows)


def stable_metrics_from_history(history):
    validate_history(history)
    late = [row['valid'] for row in history if row['epoch'] in STABLE_EPOCHS]
    if len(late) != 5:
        raise ValueError('Stable metrics require all five epochs 76-80')
    metrics = {name: statistics.mean(float(row[name]) for row in late) for name in METRICS}
    metrics.update(num_images=99, num_epochs=5, epochs=list(STABLE_EPOCHS),
        per_class_AP={label: {name: statistics.mean(row['per_class_AP'][label][name] for row in late)
                             for name in ('AP50', 'AP75', 'mAP50_95', 'map')}
                      for label in p.CLASS_NAMES})
    return metrics


def validate_resume_state(folder, state, expected, initial, identity, baseline, expected_images):
    if (state['binding'] != expected or state['initial_state_sha256'] != initial
            or state['initialization'] != identity):
        raise ValueError('Resume source/data/model/initialization identity drift')
    start, history = state['epoch'], state['history']
    validate_history(history, start)
    log_path = folder / 'epoch_log.json'
    if log_path.exists():
        log = read_json(log_path)
        # Atomic latest.pt is authoritative; an interrupted following log write
        # may leave an exact shorter prefix, but contradictory history is fatal.
        if not isinstance(log, list) or log != history[:len(log)] or len(log) > start:
            raise ValueError('Resume epoch log contradicts the committed checkpoint history')
    verify_stable_predictions(folder, history, expected, baseline, expected_images)
    if not log_path.exists() or read_json(log_path) != history:
        p.dump(log_path, history)
    return start, history


def run_result(root, backbone, head, seed, *, write=False, config=CONFIG):
    cfg, baseline = verify_manifest(root, config)
    if (backbone not in SEEDS_BY_BACKBONE or head not in HEADS
            or seed not in SEEDS_BY_BACKBONE[backbone]):
        raise ValueError('Unregistered formal backbone/head/seed combination')
    folder = result_dir(root, backbone, head, seed, 'formal')
    expected = binding(root, backbone, head, seed, 'formal')
    latest = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
    history = latest['history']
    validate_history(history)
    best = max(history, key=lambda row: row['valid']['mAP50_95'])
    checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    if (latest['binding'] != expected or latest['epoch'] != 80
            or checkpoint['binding'] != expected or checkpoint['epoch'] != best['epoch']
            or checkpoint['history'] != history[:best['epoch']]
            or read_json(folder / 'epoch_log.json') != history
            or read_json(folder / 'best_valid_metrics.json') != best['valid']
            or latest['initialization'] != read_json(folder / 'initialization_identity.json')):
        raise ValueError('Checkpoint, binding, epoch archive or best-validation evidence differs')
    contract = load_subset_contract(cfg, read_json(root / 'execution_manifest.json')['inputs'])
    records = read_json(folder / 'best_valid_records.json')
    if enrich_metrics(p.metrics_from_records(records, baseline)) != best['valid']:
        raise ValueError('Best prediction records do not reproduce recorded metrics')
    terminal_records = read_json(folder / 'terminal_valid_records.json')
    terminal_metrics = enrich_metrics(p.metrics_from_records(terminal_records, baseline))
    if terminal_metrics != history[-1]['valid'] or terminal_metrics != read_json(folder / 'terminal_valid_metrics.json'):
        raise ValueError('Terminal epoch-80 records do not reproduce recorded metrics')
    stable = verify_stable_predictions(folder, history, expected, baseline, contract['full_images'])
    if set(stable['paths']) != {str(epoch) for epoch in STABLE_EPOCHS}:
        raise ValueError('Formal completion requires all five full99 prediction archives')
    stable_metrics = stable_metrics_from_history(history)
    report = dict(binding=expected, epochs=80, best_epoch=best['epoch'],
                  primary_checkpoint='mean_epochs_76_80', primary_metric='mAP50_95',
                  metrics=stable_metrics, stable_metrics=stable_metrics,
                  stable_prediction_paths=stable['paths'], stable_prediction_sha256=stable['sha256'],
                  stable_epoch_metrics=stable['metrics'], secondary_terminal_metrics=terminal_metrics,
                  secondary_best_metrics=best['valid'], initialization=latest['initialization'],
                  initial_state_sha256=latest['initial_state_sha256'],
                  model_info=read_json(folder / 'model_info.json'), history=history,
                  hashes={name: p.sha(folder / name) for name in
                          ('latest.pt', 'best.pt', 'epoch_log.json', 'best_valid_records.json',
                           'terminal_valid_records.json', 'terminal_valid_metrics.json')},
                  formal_eligible=True, test_evaluated=False)
    if write:
        p.dump(folder / 'result.json', report)
    return report


def run_training(root, backbone, head, seed, device, resume=False, config=CONFIG):
    require_smoke(root)
    folder = result_dir(root, backbone, head, seed, 'formal')
    if (folder / 'run_binding.json').exists() and not resume:
        raise FileExistsError('Existing formal invocation requires explicit epoch-boundary resume')
    run = make_run(root, backbone, head, seed, device, 'formal', config)
    folder, cfg, baseline, model, criterion, opt, train, valid, initial, identity, lock = run
    expected = binding(root, backbone, head, seed, 'formal')
    expected_images = load_subset_contract(cfg)['full_images']
    history, start = [], 0
    if resume:
        state = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
        start, history = validate_resume_state(folder, state, expected, initial, identity,
                                              baseline, expected_images)
        model.load_state_dict(state['model_state_dict'], strict=True)
        opt.load_state_dict(state['optimizer_state_dict'])
        p.restore_rng(state['rng'], train)
        if valid.generator is not None:
            valid.generator.set_state(state['valid_loader_generator'])
    else:
        p.dump(folder / 'run_binding.json', dict(binding=expected, initialization=identity,
               initial_state_sha256=initial, started_at=p.now(),
               cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    try:
        for epoch in range(start, 80):
            tick = time.perf_counter()
            previous_best = max((row['valid']['mAP50_95'] for row in history), default=-1.)
            record = train_epoch(model, criterion, opt, train, device)
            metrics, records, valid_sha = evaluate(model, criterion, valid, baseline, device)
            record.update(epoch=epoch + 1, valid=metrics, valid_batches_sha256=valid_sha,
                          wall_seconds=time.perf_counter() - tick)
            if epoch + 1 in STABLE_EPOCHS:
                record['prediction_archive'] = save_stable_predictions(
                    folder, epoch + 1, records, expected, expected_images)
            history.append(record)
            validate_history(history, epoch + 1)
            state = dict(binding=expected, initial_state_sha256=initial, initialization=identity,
                         model_state_dict=model.state_dict(), optimizer_state_dict=opt.state_dict(),
                         epoch=epoch + 1, rng=p.rng_state(train), history=history,
                         valid_loader_generator=valid.generator.get_state() if valid.generator is not None else None,
                         test_evaluated=False)
            if metrics['mAP50_95'] > previous_best:
                p.atomic_save(folder / 'best.pt', state)
                p.dump(folder / 'best_valid_metrics.json', metrics)
                p.dump(folder / 'best_valid_records.json', records)
            if epoch + 1 == 80:
                p.dump(folder / 'terminal_valid_metrics.json', metrics)
                p.dump(folder / 'terminal_valid_records.json', records)
            p.atomic_save(folder / 'latest.pt', state)
            p.dump(folder / 'epoch_log.json', history)
            p.dump(folder / 'status.json', dict(status='RUNNING', epoch=epoch + 1,
                   binding=expected, latest_valid={key: metrics[key] for key in METRICS}, updated_at=p.now()))
            print(f'{arm(backbone, head)} seed={seed} epoch={epoch + 1}/80 '
                  f'valid_mAP50_95={metrics["mAP50_95"]:.8f} seconds={record["wall_seconds"]:.2f}', flush=True)
        report = run_result(root, backbone, head, seed, write=True, config=config)
        p.dump(folder / 'status.json', dict(status='COMPLETE', epochs=80, binding=expected,
               best_epoch=report['best_epoch'], primary_checkpoint=report['primary_checkpoint'],
               stable_mAP50_95=report['stable_metrics']['mAP50_95'],
               stable_prediction_epochs=list(STABLE_EPOCHS),
               updated_at=p.now(), test_evaluated=False))
        return report
    except Exception as error:
        p.dump(folder / 'status.json', dict(status='FAILED_NO_AUTORETRY', completed_epochs=len(history),
               error_type=type(error).__name__, error=str(error), binding=expected, updated_at=p.now()))
        raise



def validate_formal_rows(rows):
    expected = {(backbone, head, seed) for backbone in BACKBONES for head in HEADS
                for seed in SEEDS_BY_BACKBONE[backbone]}
    observed = [(row['binding']['backbone'], row['binding']['head'], row['binding']['seed'])
                for row in rows]
    if (len(rows) != 16 or set(observed) != expected
            or any(not row.get('formal_eligible') for row in rows)):
        raise ValueError('All sixteen unique registered complete formal runs are required')
    for row in rows:
        if row['epochs'] != 80 or row.get('primary_checkpoint') != 'mean_epochs_76_80':
            raise ValueError('Formal primary metric requires all five stable epochs 76-80')
        validate_history(row['history'])
        expected_stable = stable_metrics_from_history(row['history'])
        if row.get('stable_metrics') != expected_stable or row['metrics'] != expected_stable:
            raise ValueError('Stable metrics must be the arithmetic mean of epochs 76-80')
        if (set(row.get('stable_prediction_paths', {})) != {str(epoch) for epoch in STABLE_EPOCHS}
                or set(row.get('stable_prediction_sha256', {})) != {str(epoch) for epoch in STABLE_EPOCHS}):
            raise ValueError('Every stable epoch requires saved full99 predictions and SHA256')
    for key in ('execution_manifest_sha256', 'environment_fingerprint', 'subset_contract_sha256', 'validation_target_sha256'):
        if len({row['binding'][key] for row in rows}) != 1:
            raise ValueError('Campaign source, environment or subset identity differs')
    verify_pairing(rows)


def factorial_summary(rows, decision, *, subset_contract=None, output_dir=None):
    """Training/source gate first; stable statistics have one separate owner."""
    validate_formal_rows(rows)
    if (decision.get('metric') != 'mAP50_95'
            or decision.get('primary_checkpoint') != 'mean_epochs_76_80'
            or decision.get('threshold_absolute') != .03):
        raise ValueError('Unregistered stable D17 decision protocol')
    if subset_contract is None:
        cfg, _ = load_config()
        subset_contract = load_subset_contract(cfg)
    from tools.sonar_head_factorial.stable_analysis import factorial_summary as analyze
    return analyze(rows, decision, subset_contract=subset_contract, output_dir=output_dir)


def summarize(root, config=CONFIG):
    cfg, baseline = verify_manifest(root, config)
    require_smoke(root)
    rows = [run_result(root, backbone, head, seed, config=config)
            for backbone, head, seed in queue_jobs()]
    summary = factorial_summary(rows, cfg['decision'], subset_contract=load_subset_contract(cfg),
                                output_dir=root / 'stable_analysis')
    summary['execution_manifest_sha256'] = p.sha(root / 'execution_manifest.json')
    summary['environment_fingerprint'] = read_json(root / 'environment_lock.json')['fingerprint']
    summary['subset_contract_sha256'] = cfg['subset_contract_sha256']
    summary['run_evidence'] = [dict(binding=row['binding'], best_epoch=row['best_epoch'],
                                   hashes=row['hashes'], stable_prediction_sha256=row['stable_prediction_sha256'])
                               for row in rows]
    p.dump(root / 'factorial_summary.json', summary)
    return summary


def queue_jobs(shard_index=0, shard_count=1):
    if shard_count not in (1, 2) or not 0 <= shard_index < shard_count:
        raise ValueError('Queue supports one or two registered shards')
    pairs = [(seed, backbone) for seed in ALL_SEEDS for backbone in BACKBONES
             if seed in SEEDS_BY_BACKBONE[backbone]]
    return [(backbone, head, seed) for index, (seed, backbone) in
            enumerate(pairs) if index % shard_count == shard_index
            for head in HEADS]


def queue(root, device, shard_index=0, shard_count=1, config=CONFIG):
    cfg, baseline = verify_manifest(root, config)
    require_smoke(root)
    if shard_count != 1 or shard_index != 0:
        raise ValueError('Fixed sixteen-run campaign requires its registered single-GPU queue')
    jobs = queue_jobs(shard_index, shard_count)
    status_path = root / f'queue_shard_{shard_index}_status.json'
    if status_path.exists():
        raise FileExistsError('Queue already invoked; no automatic retries or overwrite')
    status = dict(status='RUNNING', pid=os.getpid(), shard_index=shard_index,
                  shard_count=shard_count, jobs=jobs, completed=[], started_at=p.now(),
                  execution_manifest_sha256=p.sha(root / 'execution_manifest.json'),
                  test_evaluated=False)
    p.dump(status_path, status)
    try:
        for backbone, head, seed in jobs:
            folder = result_dir(root, backbone, head, seed, 'formal')
            if folder.exists():
                raise FileExistsError(f'Existing formal folder cannot be queued: {folder}')
            folder.mkdir(parents=True)
            command = [sys.executable, str(Path(__file__).resolve()), 'train', '--config', str(config),
                       '--output', str(root), '--backbone', backbone, '--head', head,
                       '--seed', str(seed), '--device', device]
            with (folder / 'stdout.log').open('x', encoding='utf-8') as handle:
                child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                status.update(active=[backbone, head, seed], child_pid=child.pid, updated_at=p.now())
                p.dump(status_path, status)
                try:
                    code = child.wait(timeout=float(cfg['queue']['maximum_hours_per_run']) * 3600)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)
                    raise RuntimeError('Bounded run timed out; no automatic retry')
            if code:
                raise RuntimeError(f'Child failed with exit code {code}; no automatic retry')
            report = run_result(root, backbone, head, seed, config=config)
            status['completed'].append([backbone, head, seed])
            p.dump(status_path, status)
        summary = summarize(root, config)
        status.update(status='COMPLETE', active=None, child_pid=None, updated_at=p.now())
        status['final_decision'] = summary['decision']
        p.dump(status_path, status)
        return status
    except Exception as error:
        status.update(status='FAILED_NO_AUTORETRY', error=str(error), error_type=type(error).__name__, updated_at=p.now())
        p.dump(status_path, status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'verify', 'smoke', 'train', 'queue', 'summarize'))
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', '--result-root', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--backbone', choices=BACKBONES, default=BACKBONES[0])
    parser.add_argument('--head', choices=HEADS, default=HEADS[0])
    parser.add_argument('--seed', type=int, choices=ALL_SEEDS, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    args = parser.parse_args()
    cfg, baseline = load_config(args.config)
    root = (args.output or ROOT / cfg['result_root']).resolve()
    torch.set_num_threads(2)
    if args.mode == 'prepare':
        prepare(root, args.config)
        result = dict(status='PREPARED', execution_manifest_sha256=p.sha(root / 'execution_manifest.json'))
    elif args.mode == 'verify':
        verify_manifest(root, args.config)
        result = dict(status='PREFLIGHT_PASS')
    elif args.mode == 'smoke':
        result = smoke(root, args.device, args.config)
    elif args.mode == 'train':
        result = run_training(root, args.backbone, args.head, args.seed, args.device, args.resume, args.config)
        result = dict(status='COMPLETE', binding=result['binding'], metrics=result['metrics'])
    elif args.mode == 'queue':
        result = queue(root, args.device, args.shard_index, args.shard_count, args.config)
    else:
        result = summarize(root, args.config)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
