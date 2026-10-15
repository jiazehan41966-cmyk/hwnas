"""Controlled joint192 four-arm three-seed validation; frozen train/valid protocol.

Formal mode preserves the frozen Phase2F 80-epoch protocol. Smoke is explicitly
non-formal and runs two train batches plus one validation batch in its own path.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'tools/sonar_spec'),
                str(ROOT / 'tools/sonar_analysis'), str(ROOT / 'scripts')]
import baseline as frozen
from formal_runner import atomic_save, restore_rng, rng_state, train_epoch
from tools.sonar_joint192_validation.model import build_model, validate_architecture
from tools.detector_elastic_pilot.model import load_spec as load_pilot_spec
from tools.detector_elastic_pilot import train as pilot_train
from tools.detector_v0_2.train import verify_inputs as legacy_verify_inputs
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss
from hwnas_fpga.training.detection_coco_eval import evaluate_detection_predictions_coco

CONFIG = ROOT / 'configs/sonar_joint192_validation_20261008.yaml'
DEFAULT_ROOT = ROOT / 'results/sonar_joint192_validation_20261008_01'
EXPECTED_TRAINING = dict(epochs=80, input=640, batch_size=4, optimizer='AdamW',
    lr=.001, weight_decay=.0001, scheduler='none', amp=False, workers=0,
    initialization='shared_all_matching_reference_tensors_per_seed_fixed_blur_buffer',
    checkpoint_selection='earliest_best_valid_mAP50_95_within_80_epochs',
    resume_policy='epoch_boundary_strict_source_config_data_architecture_binding')
METRICS = ('mAP50_95', 'AP50', 'AP75', 'Precision', 'Recall')
PROTOTYPES = ('B0', 'B1', 'B2', 'B3')
SEEDS = (42, 43, 44)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   ensure_ascii=False).encode()).hexdigest()


def tensor_json(value):
    if isinstance(value, torch.Tensor):
        return {'dtype':str(value.dtype), 'shape':list(value.shape), 'values':value.cpu().tolist()}
    if isinstance(value, dict):
        return {key:tensor_json(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)):
        return [tensor_json(item) for item in value]
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def load_config():
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    if cfg['training'] != EXPECTED_TRAINING:
        raise ValueError('ELASTIC-inspired B1 pilot protocol differs from the preregistered full Phase2F training protocol')
    if tuple(cfg['prototypes']) != PROTOTYPES or tuple(cfg['seeds']) != SEEDS:
        raise ValueError('Only preregistered four arms and three seeds are permitted')
    for field in ('baseline', 'baseline_manifest'):
        if sha(ROOT / cfg[field]) != cfg[field + '_sha256']:
            raise ValueError(f'Frozen {field} hash drift')
    baseline = yaml.safe_load((ROOT / cfg['baseline']).read_text(encoding='utf-8'))
    for field in ('epochs', 'batch_size', 'lr', 'weight_decay', 'amp', 'scheduler'):
        if baseline['training'][field] != EXPECTED_TRAINING[field]:
            raise ValueError(f'Frozen baseline training differs: {field}')
    if baseline['evaluation_protocol']['development_splits'] != ['train', 'valid']:
        raise ValueError('Unexpected development splits')
    [validate_architecture(cfg['arms'][arm]) for arm in PROTOTYPES]
    load_pilot_spec()
    return cfg, baseline


def verify_inputs(cfg, baseline):
    # Preserve the historical helper and its source-glob audit. Extra unrelated
    # later files are dated audit information, not changes to frozen bytes.
    return legacy_verify_inputs(cfg, baseline)


def input_identity(value):
    return {key: item for key, item in value.items()
            if key not in ('additional_glob_files', 'historical_source_gate')}


def source_hashes(cfg):
    old_cfg, _ = pilot_train.load_config()
    paths = {ROOT / rel for rel in pilot_train.source_hashes(old_cfg)}
    paths.update(ROOT / rel for rel in (
        'tools/sonar_joint192_validation/model.py',
        'tools/sonar_joint192_validation/train.py',
        'tools/sonar_joint192_validation/queue_runner.py',
        'tests/test_sonar_joint192_model.py',
        cfg['model_spec']))
    paths.add(CONFIG)
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def prepare(result_root):
    cfg, baseline = load_config()
    inputs = verify_inputs(cfg, baseline)
    payload = dict(version=cfg['version'], config_sha256=sha(CONFIG),
        source_sha256=source_hashes(cfg), input_verification=inputs,
        training=cfg['training'], prototypes=list(PROTOTYPES), seeds=list(SEEDS),
        model_spec_sha256=sha(ROOT / cfg['model_spec']),
        evaluation_protocol=baseline['evaluation_protocol'],
        loss_config_inherited=baseline['training']['detection_loss'],
        augmentation=baseline['training']['detection_augmentation'],
        sampler=baseline['training']['detection_sampling'],
        initialization=cfg['training']['initialization'],
        namespace='sonar_joint192_validation',
        common_initialization_keys=['head_sha256','lateral_sha256'],
        architecture_arms=cfg['arms'], claims=cfg['claims'], staging=cfg['staging'],
        training_rng_policy='reset_all_RNG_to_seed_after_model_and_geometry_before_loaders',
        test_evaluated=False)
    path = Path(result_root) / 'execution_manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != payload:
        raise ValueError('Execution manifest drift; do not overwrite a frozen campaign')
    if not path.exists():
        dump(path, payload)
    return payload, baseline


def load_manifest(result_root, verify_data=True):
    path = Path(result_root) / 'execution_manifest.json'
    if not path.exists():
        raise FileNotFoundError('Run prepare before smoke or train')
    payload = json.loads(path.read_text(encoding='utf-8'))
    cfg, baseline = load_config()
    if payload['config_sha256'] != sha(CONFIG) or payload['source_sha256'] != source_hashes(cfg):
        raise ValueError('Source/config drift against frozen execution manifest')
    if payload['test_evaluated'] is not False:
        raise ValueError('Test seal violated')
    if verify_data and input_identity(payload['input_verification']) != input_identity(verify_inputs(cfg, baseline)):
        raise ValueError('Data/historical input drift against execution manifest')
    return payload, baseline


def binding(result_root, prototype, seed, formal):
    return dict(execution_manifest_sha256=sha(Path(result_root) / 'execution_manifest.json'),
        prototype=prototype, seed=seed, epochs=80 if formal else 0,
        formal_eligible=formal, namespace='sonar_joint192_validation', test_evaluated=False)


def result_dir(result_root, prototype, seed, formal=True):
    return Path(result_root) / ('formal' if formal else 'smoke') / prototype / f'seed_{seed}'


def environment(device):
    result = dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__,
        platform=platform.platform(), device=device, cudnn=torch.backends.cudnn.version(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic)
    if device == 'cuda':
        result.update(gpu=torch.cuda.get_device_name(0), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                      cuda_runtime=torch.version.cuda)
    return result


def make_run(result_root, prototype, seed, device, formal):
    if prototype not in PROTOTYPES or seed not in SEEDS:
        raise ValueError('Unregistered prototype or seed')
    manifest, baseline = load_manifest(result_root, True)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    cfg, _ = load_config()
    if device == 'cuda':
        spec = yaml.safe_load((ROOT/cfg['environment_spec']).read_text(encoding='utf-8'))
        actual = environment(device)
        for key,value in spec['expected'].items():
            if actual.get(key) != str(value):
                raise ValueError(f'Pinned server environment drift: {key}')
    folder = result_dir(result_root, prototype, seed, formal)
    folder.mkdir(parents=True, exist_ok=True)
    frozen.seed_all(seed)
    cfg, _ = load_config()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == 'cuda' else []):
        model = build_model(architecture=cfg['arms'][prototype], seed=seed).to(device)
    initialization_identity = model.initialization_identity()
    model._runner_initialization_identity = deepcopy(initialization_identity)
    dump(folder / 'initialization_identity.json', initialization_identity)
    initial_sha = frozen.state_hash(model)
    # Geometry counting must not advance the training RNG or update BN state.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == 'cuda' else []):
        info = frozen.get_model_info(model, (1, 1, 640, 640))
    if frozen.state_hash(model) != initial_sha:
        raise ValueError('Model counting changed initial state')
    fixed_blur_macs = 48 * 160 * 160 * 9 if cfg['arms'][prototype]['backbone']['stage1']['downsample'] == 'blur3_after_dense_dw' else 0
    info['conv_module_macs'] = info['estimated_macs']
    info['fixed_blur_macs'] = fixed_blur_macs
    info['estimated_macs'] += fixed_blur_macs
    info['FLOPs_2x_MACs'] = 2 * info['estimated_macs']
    info.update(prototype=prototype, levels=list(model.levels), feature_strides=list(model.feature_strides),
        architecture_contract=model.architecture_contract(), initialization_identity=initialization_identity)
    dump(folder / 'model_info.json', info)
    # Model/tower construction and geometry never alter the actual training stream.
    frozen.seed_all(seed)
    dump(folder / 'training_rng_identity.json', dict(seed=seed,
        torch_cpu_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        numpy_rng_sha256=canonical_sha(np.random.get_state()[1].tolist()),
        python_rng_sha256=canonical_sha(random.getstate()),
        policy='reset_after_construction_before_loaders'))
    train_loader = frozen.make_loader(baseline, ROOT / baseline['dataset']['data_dir'], 'train', seed, 0, train=True)
    valid_loader = frozen.make_loader(baseline, ROOT / baseline['dataset']['data_dir'], 'valid', seed, 0)
    criterion = DenseAnchorDetectionLoss(model.loss_config(baseline)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    return folder, baseline, model, optimizer, criterion, train_loader, valid_loader, initial_sha


def smoke(result_root, prototype, seed, device):
    folder = result_dir(result_root, prototype, seed, False)
    if (folder / 'smoke_report.json').exists():
        raise FileExistsError('Smoke already exists; preserve evidence, use a separate result root')
    folder, baseline, model, optimizer, criterion, train_loader, valid_loader, initial_sha = make_run(
        result_root, prototype, seed, device, False)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    model.train()
    batches = []
    for index, (images, targets) in enumerate(train_loader):
        if index >= 2:
            break
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = model(images.to(device)); frozen.finite_outputs(output)
        loss, parts = criterion(output, targets)
        if not torch.isfinite(loss) or not all(math.isfinite(v) for v in parts.values()):
            raise ValueError('Smoke nonfinite loss')
        loss.backward()
        gradients = frozen.gradient_summary(model)
        optimizer.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        batches.append(dict(batch=index, size=int(images.shape[0]), seconds=time.perf_counter()-tick,
            augmented_images_sha256=hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest(),
            targets_sha256=canonical_sha(tensor_json(targets)),
            loss=parts, gradients=gradients,
            prediction_shapes=[list(x.shape) for x in output['predictions']]))
    validation, _ = frozen.evaluate(model, valid_loader, criterion, baseline, device, max_batches=1)
    report = dict(status='PASS', binding=binding(result_root, prototype, seed, False),
        environment=environment(device), initial_state_sha256=initial_sha,
        final_state_sha256=frozen.state_hash(model), train_batches=batches,
        initialization_identity=model._runner_initialization_identity,
        architecture_contract=model.architecture_contract(),
        validation_smoke=validation, formal_eligible=False, test_evaluated=False,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else None,
        updated_at=now())
    dump(folder / 'smoke_report.json', report)
    print(json.dumps(dict(status='PASS', prototype=prototype, formal_eligible=False,
        batch4_seconds=[x['seconds'] for x in batches], peak_cuda_allocated_bytes=report['peak_cuda_allocated_bytes'])), flush=True)
    return report


def supplemental(records, baseline):
    predictions = [r['prediction'] for r in records]
    targets = [r['target'] for r in records]
    per_class = {}
    for label, name in enumerate(baseline['dataset']['class_names']):
        ps, ts = [], []
        for p, t in zip(predictions, targets):
            pi = [i for i, x in enumerate(p['labels']) if x == label]
            ti = [i for i, x in enumerate(t['labels']) if x == label]
            ps.append({k:[p[k][i] for i in pi] for k in ('boxes','scores','labels')})
            ts.append({k:[t[k][i] for i in ti] for k in ('boxes','labels')})
        per_class[name] = frozen.precision_recall(ps, ts, baseline['evaluation_protocol']['precision_recall_score_threshold'])
    area_metrics = evaluate_detection_predictions_coco(predictions, targets, num_classes=2,
        class_names=baseline['dataset']['class_names'], area_ranges={'all':(0.,float('inf')), 'small':(0.,1024.)},
        max_detections=baseline['evaluation_protocol']['max_detections_per_image_per_class'])
    return dict(per_class_precision_recall=per_class, all_and_small_area_ap=area_metrics,
        scope='custom_frozen_evaluator_in_640_input_coordinates_not_official_COCO_area_certification')


def summarize(result_root, prototype, seed):
    manifest, baseline = load_manifest(result_root, False)
    folder = result_dir(result_root, prototype, seed)
    state = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
    expected = binding(result_root, prototype, seed, True)
    if state['binding'] != expected or state['epoch'] != 80:
        raise ValueError('Complete checkpoint identity mismatch')
    history = state['history']
    if [r['epoch'] for r in history] != list(range(1, 81)):
        raise ValueError('Incomplete formal epoch history')
    if state.get('global_step') != 6640 or any(r.get('batches') != 83 for r in history):
        raise ValueError('Formal training batch/global-step count differs from frozen train330 protocol')
    best = max(history, key=lambda r:r['valid']['mAP50_95'])
    best_state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    if best_state['binding'] != expected or best_state['epoch'] != best['epoch']:
        raise ValueError('Best checkpoint does not match earliest best epoch')
    if best_state['history'] != history[:best['epoch']]:
        raise ValueError('Best checkpoint history is not the latest history prefix')
    if json.loads((folder/'epoch_log.json').read_text(encoding='utf-8')) != history:
        raise ValueError('Epoch log differs from completed checkpoint history')
    if any(r['valid']['num_images'] != 99 or not math.isfinite(r['valid']['mAP50_95']) for r in history):
        raise ValueError('Incomplete/nonfinite validation history')
    status = json.loads((folder / 'status.json').read_text(encoding='utf-8'))
    if status['status'] != 'COMPLETE' or status['binding'] != expected or status.get('epochs') != 80:
        raise ValueError('Completed checkpoint without matching completion status')
    initial_identity = json.loads((folder/'initialization_identity.json').read_text(encoding='utf-8'))
    if (state.get('initialization_identity') != initial_identity or
            best_state.get('initialization_identity') != initial_identity or
            state.get('architecture_contract') != best_state.get('architecture_contract')):
        raise ValueError('Checkpoint initialization/architecture identities differ')
    info = json.loads((folder / 'model_info.json').read_text(encoding='utf-8'))
    record = dict(prototype=prototype, seed=seed, best_epoch=best['epoch'],
        **{k:best['valid'][k] for k in METRICS}, endpoint_mAP50_95=history[-1]['valid']['mAP50_95'],
        Params=info['total_params'], MACs=info['estimated_macs'],
        best_checkpoint_sha256=sha(folder/'best.pt'), latest_checkpoint_sha256=sha(folder/'latest.pt'),
        history_sha256=sha(folder/'epoch_log.json'), execution_manifest_sha256=expected['execution_manifest_sha256'],
        formal_eligible=True, test_evaluated=False)
    best_records = json.loads((folder/'best_valid_records.json').read_text(encoding='utf-8'))
    record['supplemental'] = supplemental(best_records, baseline)
    dump(folder/'result.json', record)
    return record


def train(result_root, prototype, seed, device, resume=False):
    folder = result_dir(result_root, prototype, seed)
    if resume and not (folder/'latest.pt').is_file():
        raise FileNotFoundError('Explicit resume requires its own existing latest.pt')
    if (folder/'latest.pt').exists() and not resume:
        raise FileExistsError('Existing checkpoint requires --resume')
    if (folder/'run_binding.json').exists() and not (folder/'latest.pt').exists():
        raise RuntimeError('Incomplete invocation before first checkpoint requires independent audit; no silent overwrite')
    folder, baseline, model, optimizer, criterion, train_loader, valid_loader, initial_sha = make_run(
        result_root, prototype, seed, device, True)
    expected = binding(result_root, prototype, seed, True)
    start_epoch, history, parents = 0, [], []
    if resume and (folder/'latest.pt').exists():
        state = torch.load(folder/'latest.pt', map_location='cpu', weights_only=False)
        if (state['binding'] != expected or state['architecture'] != model.candidate_architecture or
                state.get('initialization_identity') != model._runner_initialization_identity or
                state['initial_state_sha256'] != initial_sha or
                state.get('architecture_contract') != model.architecture_contract()):
            raise ValueError('Resume binding/architecture mismatch')
        model.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        restore_rng(state['rng'], train_loader)
        start_epoch, history, initial_sha = state['epoch'], state['history'], state['initial_state_sha256']
        if [r['epoch'] for r in history] != list(range(1, start_epoch+1)):
            raise ValueError('Resume has inconsistent epoch history')
        parents = state['parents'] + [dict(checkpoint_sha256=sha(folder/'latest.pt'), epoch=start_epoch)]
    if start_epoch == 80:
        return summarize(result_root, prototype, seed)
    if not 0 <= start_epoch < 80:
        raise ValueError('Invalid checkpoint epoch')
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = 'unavailable_isolated_source_snapshot'
    dump(folder/'run_binding.json', dict(binding=expected, initial_state_sha256=initial_sha,
        initialization=EXPECTED_TRAINING['initialization'], start_epoch=start_epoch,
        initialization_identity=model._runner_initialization_identity,
        architecture_contract=model.architecture_contract(),
        git_commit=commit, environment=environment(device), loss_config=asdict(criterion.config),
        started_at=now(), test_evaluated=False))
    dump(folder/'status.json', dict(status='RUNNING', binding=expected, epoch=start_epoch, updated_at=now()))
    try:
        for epoch in range(start_epoch, 80):
            tick = time.perf_counter()
            old_best = max((r['valid']['mAP50_95'] for r in history), default=-1.)
            record = train_epoch(model, optimizer, train_loader, baseline, device, None)
            if record['batches'] != 83:
                raise ValueError('Formal epoch must contain all 83 frozen training batches')
            metrics, valid_records = frozen.evaluate(model, valid_loader, criterion, baseline, device)
            if metrics['num_images'] != 99:
                raise ValueError('Formal validation must evaluate all 99 frozen valid images')
            record.update(epoch=epoch+1, valid=metrics, wall_seconds=time.perf_counter()-tick)
            history.append(record)
            state = dict(binding=expected, architecture=model.candidate_architecture,
                initialization_identity=model._runner_initialization_identity,
                architecture_contract=model.architecture_contract(),
                prototype=prototype, seed=seed, epoch=epoch+1,
                global_step=sum(r['batches'] for r in history),
                model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                rng=rng_state(train_loader), history=history, parents=parents,
        initial_state_sha256=initial_sha, updated_at=now(), test_evaluated=False)
            if metrics['mAP50_95'] > old_best:
                atomic_save(folder/'best.pt', state)
                dump(folder/'best_valid_metrics.json', metrics)
                dump(folder/'best_valid_records.json', valid_records)
            atomic_save(folder/'latest.pt', state)
            dump(folder/'epoch_log.json', history)
            dump(folder/'status.json', dict(status='RUNNING', binding=expected, epoch=epoch+1,
                latest_valid={k:metrics[k] for k in METRICS}, elapsed_training_seconds=sum(r['wall_seconds'] for r in history),
                updated_at=now(), test_evaluated=False))
            print(prototype, f'seed={seed}', f'epoch={epoch+1}/80',
                f'loss={record["losses"]["loss"]:.6f}', f'valid_mAP50_95={metrics["mAP50_95"]:.8f}',
                f'seconds={record["wall_seconds"]:.2f}', flush=True)
        dump(folder/'status.json', dict(status='COMPLETE', binding=expected, epochs=80,
            elapsed_training_seconds=sum(r['wall_seconds'] for r in history), updated_at=now(), test_evaluated=False))
        return summarize(result_root, prototype, seed)
    except Exception as error:
        dump(folder/'status.json', dict(status='FAILED_NO_AUTORETRY', binding=expected,
            completed_epochs=len(history), error_type=type(error).__name__, error=str(error), updated_at=now(), test_evaluated=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare','preflight','smoke','train','summarize'])
    parser.add_argument('--prototype', choices=PROTOTYPES, default='B0')
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--result-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == 'prepare':
        payload, _ = prepare(args.result_root)
        print(json.dumps(dict(status='PREPARED', manifest_sha256=sha(args.result_root/'execution_manifest.json'),
            data_images={k:v['images'] for k,v in payload['input_verification']['datasets'].items()},
            historical_source_gate=payload['input_verification']['historical_source_gate'])), flush=True)
    elif args.mode == 'preflight':
        load_manifest(args.result_root, True)
        print('PREFLIGHT PASS', sha(args.result_root/'execution_manifest.json'))
    elif args.mode == 'smoke':
        smoke(args.result_root, args.prototype, args.seed, args.device)
    elif args.mode == 'train':
        train(args.result_root, args.prototype, args.seed, args.device, args.resume)
    else:
        print(json.dumps(summarize(args.result_root, args.prototype, args.seed)))


if __name__ == '__main__':
    main()
