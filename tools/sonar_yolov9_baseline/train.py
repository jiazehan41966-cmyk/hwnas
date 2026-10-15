"""Matched-budget L1 and pinned author YOLOv9-T, train/valid only."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
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
from tools.detector_b1_light.model import build_model as build_l1
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss, decode_detection_predictions

CONFIG = ROOT / 'configs/sonar_yolov9_baseline_20261005.yaml'
DEFAULT_RESULT = ROOT / 'results/sonar_yolov9_baseline_20261005'
GROUPS = ('L1_scratch', 'YOLOv9_T_scratch', 'YOLOv9_T_COCO')
METRICS = ('mAP50_95', 'AP50', 'AP75', 'Precision', 'Recall')


def load_config():
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    b = p.load_baseline()
    if (cfg['groups'] != list(GROUPS) or cfg['training']['seeds'] != [42]
            or cfg['training']['epochs'] != 80 or cfg['training']['batch_size'] != 4
            or cfg['training']['input'] != 640 or cfg['training']['amp']):
        raise ValueError('Unregistered comparison contract')
    ckpt = ROOT / cfg['author']['checkpoint']
    if not ckpt.is_file() or p.sha(ckpt) != cfg['author']['checkpoint_sha256']:
        raise ValueError('Pinned official COCO checkpoint absent or changed')
    return cfg, b


def source_hashes(cfg):
    paths = set((ROOT / 'src/hwnas_fpga').rglob('*.py'))
    for directory in ('tools/sonar_yolov9_baseline', 'tools/sonar_spec', 'tools/sonar_analysis',
                      'tools/detector_b1_light'):
        paths.update((ROOT / directory).glob('*.py'))
    original = json.loads(p.FREEZE_PATH.read_text(encoding='utf-8'))
    paths.update(ROOT / name.replace('\\', '/') for name in original['source_sha256'])
    author = ROOT / cfg['author']['source_root']
    for pattern in ('*.py', '*.yaml', '*.yml'):
        paths.update(author.rglob(pattern))
    paths.update((CONFIG, p.BASELINE_PATH, p.FREEZE_PATH,
                  ROOT / 'configs/detector_b1_light_model.yaml',
                  ROOT / 'configs/search_space_sonar_v0_1.yaml', author / 'LICENSE.md',
                  DEFAULT_RESULT / 'environment_spec.json',
                  DEFAULT_RESULT / 'env_overlay_lock.json'))
    return {path.relative_to(ROOT).as_posix(): p.sha(path) for path in sorted(paths)}


def prepare(result_root):
    cfg, b = load_config()
    identity = p.verify_inputs(b)
    try:
        commit = subprocess.check_output(['git', '-C', str(ROOT / cfg['author']['source_root']),
                                          'rev-parse', 'HEAD'], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        receipt = json.loads((ROOT / cfg['author']['source_root'] / 'source_receipt.json').read_text())
        commit = receipt['commit']
    if commit != cfg['author']['commit']:
        raise ValueError('Author source pin differs')
    manifest = dict(version=cfg['version'], config_sha256=p.sha(CONFIG), config=cfg,
                    author_commit=commit, source_sha256=source_hashes(cfg),
                    inputs=identity, test_evaluated=False)
    path = result_root / 'execution_manifest.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Existing campaign freeze differs; use a new result root')
    else:
        p.dump(path, manifest)
    return manifest


def verify_manifest(result_root):
    cfg, b = load_config()
    manifest = json.loads((result_root / 'execution_manifest.json').read_text(encoding='utf-8'))
    if (manifest['config_sha256'] != p.sha(CONFIG) or manifest['config'] != cfg
            or manifest['source_sha256'] != source_hashes(cfg)
            or manifest['inputs'] != p.verify_inputs(b)):
        raise ValueError('Frozen source/config/data binding drift')
    return cfg, b


def environment(device):
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'Pillow', 'PyYAML', 'opencv-python-headless',
                 'ipython', 'seaborn', 'requests', 'psutil', 'thop', 'setuptools'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    value = dict(python=platform.python_version(), interpreter=sys.executable,
                 packages=packages, device=device, cuda_runtime=torch.version.cuda,
                 deterministic_warn_only=True, amp=False, workers=0,
                 cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if device == 'cuda':
        value.update(gpu=torch.cuda.get_device_name(0),
                     gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
    return value


def result_dir(root, group, seed, mode):
    return root / mode / group / f'seed_{seed}'


def binding(root, group, seed, mode):
    return dict(execution_manifest_sha256=p.sha(root / 'execution_manifest.json'),
                group=group, seed=seed, mode=mode, test_evaluated=False)


def count_model(model, channels):
    # The project hook counter reports Conv/Linear MACs; both detectors use it.
    # Preserve RNG and every tensor including BN buffers across counting.
    saved = deepcopy(model.state_dict())
    old_mode = model.training
    device = next(model.parameters()).device
    cuda_devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=cuda_devices):
        info = p.frozen.get_model_info(model, (1, channels, 640, 640))
    model.load_state_dict(saved, strict=True)
    model.train(old_mode)
    info['FLOPs_2x_MACs'] = 2 * info['estimated_macs']
    info['counting_scope'] = 'Conv2d_Linear_only_no_NMS_latency_or_hardware_claim'
    return info


def make_run(root, group, seed, device, mode):
    cfg, b = verify_manifest(root)
    if group not in GROUPS or seed != 42 or mode not in ('smoke', 'formal'):
        raise ValueError('Unregistered run')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent fallback')
    folder = result_dir(root, group, seed, mode)
    folder.mkdir(parents=True, exist_ok=True)
    p.seed_all(seed)
    if group == 'L1_scratch':
        model = build_l1(prototype='L1', seed=seed).to(device)
        criterion = DenseAnchorDetectionLoss(model.loss_config(b)).to(device)
        init = model.initialization_identity()
        info = count_model(model, 1)
    else:
        from tools.sonar_yolov9_baseline import author_adapter as author
        condition = 'scratch' if group == 'YOLOv9_T_scratch' else 'coco_pretrained'
        model = author.build_author_model(ROOT / cfg['author']['source_root'], condition,
                                           ROOT / cfg['author']['checkpoint'] if condition == 'coco_pretrained' else None,
                                           num_classes=2, device=device)
        criterion, init = None, model.author_adapter_metadata
        info = dict(training_dual_graph=count_model(model, 3))
        main = author.build_main_inference_model(model)
        with torch.no_grad():
            image = torch.zeros((1, 3, 640, 640), device=device)
            model.eval(); main.eval()
            expected = model(image)[0][1]
            actual = main(image)[0]
            error = float((expected - actual).abs().max())
            if not torch.allclose(expected, actual, rtol=1e-5, atol=1e-4):
                raise ValueError(f'Main inference graph differs: {error}')
        info['main_inference_graph'] = count_model(main, 3)
        info['main_graph_max_abs_error'] = error
        del main
        gc.collect()
    initial_sha = p.state_hash(model)
    p.dump(folder / 'initialization_identity.json', init)
    p.dump(folder / 'model_info.json', info)
    # Model construction, COCO loading and graph counting consume no data RNG.
    p.seed_all(seed)
    p.dump(folder / 'training_rng_identity.json', p.training_rng_identity(seed))
    loaders = p.make_loaders(b, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    return folder, b, model, criterion, optimizer, *loaders, initial_sha, init


def batch_loss(group, model, criterion, images, targets, device):
    if group == 'L1_scratch':
        output = model(images.to(device))
        return criterion(output, targets)
    output = model(p.gray_to_three(images).to(device))
    from tools.sonar_yolov9_baseline import author_adapter as author
    loss, native_parts = author.author_loss(model, output, targets)
    return loss, dict(loss=float(loss.detach()), box_loss=float(native_parts['box']),
                      class_loss=float(native_parts['cls']), dfl_loss=float(native_parts['dfl']))


def train_epoch(group, model, criterion, optimizer, loader, device, max_batches=None):
    model.train()
    total = Counter(); n = 0; image_order = []
    started = time.perf_counter()
    for index, (images, targets) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        loss, parts = batch_loss(group, model, criterion, images, targets, device)
        if not torch.isfinite(loss) or not all(math.isfinite(float(v)) for v in parts.values()):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        grads = [param.grad for param in model.parameters() if param.grad is not None]
        if not grads or any(not torch.isfinite(grad).all() for grad in grads):
            raise ValueError('Nonfinite or absent gradients')
        optimizer.step()
        n += 1
        total.update({k: float(v) for k, v in parts.items()})
        image_order.extend(Path(target['path']).name for target in targets)
    if n == 0:
        raise ValueError('Empty train epoch')
    if device == 'cuda':
        torch.cuda.synchronize()
    return dict(batches=n, images=len(image_order), image_order_sha256=p.canonical_sha(image_order),
                losses={k: v / n for k, v in total.items()}, gradient_finite=True,
                lr=optimizer.param_groups[0]['lr'], train_seconds=time.perf_counter()-started)


@torch.no_grad()
def evaluate(group, model, criterion, loader, baseline, device, max_batches=None):
    model.eval()
    records = []
    data = ROOT / baseline['dataset']['data_dir']
    ep = baseline['evaluation_protocol']
    for index, (images, targets) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        if group == 'L1_scratch':
            output = model(images.to(device))
            predictions = decode_detection_predictions(output, config=criterion.config,
                score_threshold=ep['ap_decode_score_threshold'],
                pre_nms_topk=ep['pre_nms_topk'], nms_iou_threshold=ep['nms_iou_threshold'],
                max_detections=ep['post_nms_max_detections_per_image'])
        else:
            from tools.sonar_yolov9_baseline import author_adapter as author
            output = model(p.gray_to_three(images).to(device))
            predictions = author.decode_author(model, output,
                score_threshold=ep['ap_decode_score_threshold'],
                pre_nms_topk=ep['pre_nms_topk'], nms_iou_threshold=ep['nms_iou_threshold'],
                max_detections=ep['post_nms_max_detections_per_image'])
        records.extend(p.records_from_batch(predictions, targets, data))
    return p.metrics_from_records(records, baseline), records


def smoke(root, group, seed, device):
    folder = result_dir(root, group, seed, 'smoke')
    if (folder / 'smoke_report.json').exists():
        raise FileExistsError('Preserve completed smoke; use a new result root')
    folder, b, model, criterion, opt, train, valid, initial_sha, init = make_run(root, group, seed, device, 'smoke')
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        trained = train_epoch(group, model, criterion, opt, train, device, 2)
        metrics, records = evaluate(group, model, criterion, valid, b, device, 1)
        final_sha = p.state_hash(model)
        if final_sha == initial_sha or metrics['num_images'] != 4:
            raise ValueError('Smoke did not update model or cover a full validation batch')
        report = dict(status='PASS', binding=binding(root, group, seed, 'smoke'),
                      environment=environment(device), training=trained, valid=metrics,
                      initial_state_sha256=initial_sha, final_state_sha256=final_sha,
                      initialization_report=init, seconds=time.perf_counter()-started,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if device == 'cuda' else 0,
                      formal_eligible=False, test_evaluated=False)
        p.dump(folder / 'smoke_report.json', report)
        p.dump(folder / 'valid_records.json', records)
        print(json.dumps(report), flush=True)
    except Exception as error:
        p.dump(folder / 'smoke_failure.json', dict(status='FAILED', error=str(error),
               error_type=type(error).__name__, formal_eligible=False, test_evaluated=False))
        raise


def summarize(root, group, seed):
    folder = result_dir(root, group, seed, 'formal')
    expected = binding(root, group, seed, 'formal')
    state = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
    history = state['history']
    if (state['binding'] != expected or state['epoch'] != 80
            or [r['epoch'] for r in history] != list(range(1,81))
            or any(r['valid']['num_images'] != 99 for r in history)):
        raise ValueError('Incomplete or changed formal history')
    best = max(history, key=lambda record: record['valid']['mAP50_95'])
    best_state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    if (best_state['binding'] != expected or best_state['epoch'] != best['epoch']
            or best_state['history'] != history[:best['epoch']]
            or json.loads((folder / 'epoch_log.json').read_text()) != history):
        raise ValueError('Best checkpoint or epoch archive differs')
    metrics = json.loads((folder / 'best_valid_metrics.json').read_text())
    if metrics != best['valid']:
        raise ValueError('Best validation metrics differ')
    info = json.loads((folder / 'model_info.json').read_text())
    deploy = info.get('main_inference_graph', info)
    report = dict(group=group, seed=seed, epochs=80, best_epoch=best['epoch'],
                  **{key: metrics[key] for key in METRICS}, Params=deploy['total_params'],
                  MACs=deploy['estimated_macs'], FLOPs_2x_MACs=deploy['FLOPs_2x_MACs'],
                  model_info=info, initialization=state['initialization'],
                  best_checkpoint_sha256=p.sha(folder/'best.pt'),
                  latest_checkpoint_sha256=p.sha(folder/'latest.pt'),
                  epoch_log_sha256=p.sha(folder/'epoch_log.json'), binding=expected,
                  formal_eligible=True, test_evaluated=False,
                  interpretation='single_seed_unified_budget_not_author_recipe_reproduction')
    p.dump(folder / 'result.json', report)
    return report


def run_training(root, group, seed, device, resume=False):
    folder = result_dir(root, group, seed, 'formal')
    if folder.exists() and (folder / 'run_binding.json').exists() and not resume:
        raise FileExistsError('Existing formal invocation requires explicit resume')
    folder, b, model, criterion, opt, train, valid, initial_sha, init = make_run(root, group, seed, device, 'formal')
    expected = binding(root, group, seed, 'formal')
    history = []; start = 0
    if resume:
        state = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
        if (state['binding'] != expected or state['initial_state_sha256'] != initial_sha
                or state['initialization'] != init):
            raise ValueError('Resume source/data/model/initialization binding drift')
        model.load_state_dict(state['model_state_dict'], strict=True)
        opt.load_state_dict(state['optimizer_state_dict'])
        p.restore_rng(state['rng'], train)
        start, history = state['epoch'], state['history']
        if [r['epoch'] for r in history] != list(range(1, start+1)) or not 0 <= start <= 80:
            raise ValueError('Resume history inconsistent')
    p.dump(folder / 'run_binding.json', dict(binding=expected, initialization=init,
           initial_state_sha256=initial_sha, environment=environment(device), started_at=p.now()))
    try:
        for epoch in range(start, 80):
            tick = time.perf_counter()
            previous_best = max((r['valid']['mAP50_95'] for r in history), default=-1.)
            record = train_epoch(group, model, criterion, opt, train, device)
            metrics, records = evaluate(group, model, criterion, valid, b, device)
            if metrics['num_images'] != 99:
                raise ValueError('Validation did not cover all 99 images')
            record.update(epoch=epoch+1, valid=metrics, wall_seconds=time.perf_counter()-tick)
            history.append(record)
            state = dict(binding=expected, initial_state_sha256=initial_sha, initialization=init,
                         model_state_dict=model.state_dict(), optimizer_state_dict=opt.state_dict(),
                         epoch=epoch+1, rng=p.rng_state(train), history=history, test_evaluated=False)
            if metrics['mAP50_95'] > previous_best:
                p.atomic_save(folder/'best.pt', state)
                p.dump(folder/'best_valid_metrics.json', metrics)
                p.dump(folder/'best_valid_records.json', records)
            p.atomic_save(folder/'latest.pt', state)
            p.dump(folder/'epoch_log.json', history)
            p.dump(folder/'status.json', dict(status='RUNNING', epoch=epoch+1,
                    binding=expected, latest_valid={k:metrics[k] for k in METRICS}, updated_at=p.now()))
            print(f'{group} seed={seed} epoch={epoch+1}/80 loss={record["losses"]["loss"]:.6f} '
                  f'valid_mAP50_95={metrics["mAP50_95"]:.8f} seconds={record["wall_seconds"]:.2f}', flush=True)
        report = summarize(root, group, seed)
        p.dump(folder/'status.json', dict(status='COMPLETE', epochs=80, binding=expected,
               best_epoch=report['best_epoch'], updated_at=p.now(), test_evaluated=False))
        return report
    except Exception as error:
        p.dump(folder/'status.json', dict(status='FAILED_NO_AUTORETRY', completed_epochs=len(history),
               error_type=type(error).__name__, error=str(error), binding=expected, updated_at=p.now()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare','verify','smoke','train','summarize'))
    parser.add_argument('--result-root', type=Path, default=DEFAULT_RESULT)
    parser.add_argument('--group', choices=GROUPS, default=GROUPS[0])
    parser.add_argument('--seed', type=int, choices=(42,), default=42)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.result_root.resolve()
    if args.mode == 'prepare':
        manifest = prepare(root)
        print(json.dumps(dict(status='PREPARED', author_commit=manifest['author_commit'],
                              groups=list(GROUPS), manifest_sha256=p.sha(root/'execution_manifest.json'))))
    elif args.mode == 'verify':
        verify_manifest(root)
        print('PREFLIGHT PASS', p.sha(root/'execution_manifest.json'))
    elif args.mode == 'smoke':
        smoke(root,args.group,args.seed,args.device)
    elif args.mode == 'train':
        run_training(root,args.group,args.seed,args.device,args.resume)
    else:
        print(json.dumps(summarize(root,args.group,args.seed)))


if __name__ == '__main__':
    main()
