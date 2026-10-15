"""Phase 2F independent multi-seed runner; exactly six fixed architectures."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from formal_protocol import ROOT, dump, load_protocol, sha, source_binding, verify_baseline, frozen
from formal_runner import atomic_save, restore_rng, rng_state, train_epoch
from hwnas_fpga.models.sonar_candidate import build_sonar_candidate
from hwnas_fpga.models.sonar_supernet import architecture_id, enumerate_architectures
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss
from phase2e_runner import state_hash


REPORT = ROOT / 'reports/phase2f_multiseed'
MANIFEST = REPORT / 'phase2f_manifest.json'
PHASE2E_FINAL = ROOT / 'reports/phase2e_multifidelity/final_results.csv'
FORMAL_CONFIG = ROOT / 'configs/supernet_sonar_v0_1_formal.yaml'
ARCHITECTURES = [
    ('A1', 'P3_P4_P5_s1k5_s2k3_s4k3_s5k3_s6k5_s7k5', 'final high-accuracy candidate'),
    ('A2', 'P3_P4_P5_s1k5_s2k3_s4k3_s5k3_s6k5_s7k3', 'stage7 matched control'),
    ('B1', 'P2_P3_P4_P5_s1k3_s2k5_s4k5_s5k5_s6k5_s7k5', 'matched P2 candidate'),
    ('B2', 'P3_P4_P5_s1k3_s2k5_s4k5_s5k5_s6k5_s7k5', 'matched P345 control'),
    ('C', 'P3_P4_P5_s1k3_s2k5_s4k5_s5k3_s6k5_s7k3', 'accuracy-compute candidate'),
    ('D', 'P3_P4_P5_s1k3_s2k3_s4k3_s5k3_s6k3_s7k3', 'all-k3 baseline-like control'),
]
METRICS = ('mAP50_95', 'AP50', 'AP75', 'Precision', 'Recall')


def now():
    return datetime.now(timezone.utc).isoformat()


def architecture_map():
    return {architecture_id(a): a for a in enumerate_architectures()}


def source_hashes():
    paths = [FORMAL_CONFIG, ROOT / 'tools/sonar_spec/phase2f_runner.py',
             ROOT / 'tools/sonar_spec/phase2f_pipeline.py',
             ROOT / 'tools/sonar_spec/phase2f_analysis.py']
    for package in ('sonar_candidate', 'sonar_supernet'):
        paths.extend(sorted((ROOT / 'src/hwnas_fpga/models' / package).rglob('*.py')))
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in paths}


def read_seed42():
    with PHASE2E_FINAL.open(newline='', encoding='utf-8') as handle:
        rows = {r['architecture_id']: r for r in csv.DictReader(handle)}
    selected = []
    for label, aid, role in ARCHITECTURES:
        if aid not in rows:
            raise ValueError(f'Frozen seed42 row missing: {aid}')
        row = rows[aid]
        if int(row['seed']) != 42 or int(row['budget']) != 80:
            raise ValueError(f'Invalid seed42/budget binding: {aid}')
        selected.append(dict(label=label, role=role, architecture_id=aid,
            best_epoch=int(row['best_epoch']), **{k: float(row[k]) for k in METRICS},
            endpoint_mAP50_95=float(row['endpoint_mAP50_95']), Params=int(row['Params']),
            MACs=int(row['MACs']), checkpoint_sha256=row['checkpoint_sha256'],
            history_sha256=row['source_history_sha256']))
    return selected


def prepare_manifest():
    protocol, baseline = load_protocol(FORMAL_CONFIG)
    legal = architecture_map()
    if len(legal) != 144:
        raise ValueError('Search Space V0.1 must contain exactly 144 architectures')
    for _, aid, _ in ARCHITECTURES:
        if aid not in legal:
            raise ValueError(f'Preregistered architecture is not legal: {aid}')
    payload = dict(version='sonar_phase2f_multiseed_v1', status='frozen_before_seed43_44',
        search_space='configs/search_space_sonar_v0_1.yaml',
        search_space_sha256=protocol['space_sha256'],
        baseline='configs/baseline_sonar_frozen.yaml',
        baseline_sha256=protocol['baseline_sha256'],
        baseline_manifest_sha256=protocol['baseline_manifest_sha256'],
        formal_config_sha256=sha(FORMAL_CONFIG), phase2e_final_results_sha256=sha(PHASE2E_FINAL),
        phase2f_protocol_sha256=sha(REPORT / 'PHASE2F_PROTOCOL.md'),
        source_sha256=source_hashes(), seeds=dict(existing=[42], new=[43, 44]), epochs=80,
        training=dict(input=640, batch_size=4, optimizer='AdamW', lr=.001,
            weight_decay=.0001, scheduler='none', amp=False, workers=0,
            initialization='independent_random_from_scratch_per_seed',
            checkpoint_selection='earliest_best_valid_mAP50_95_within_80_epochs'),
        architectures=read_seed42(), test_evaluated=False)
    if MANIFEST.exists() and json.loads(MANIFEST.read_text(encoding='utf-8')) != payload:
        raise ValueError('Frozen Phase2F manifest drift')
    dump(MANIFEST, payload)
    return payload, baseline


def load_manifest(verify_data=True):
    if not MANIFEST.exists():
        raise FileNotFoundError('Run phase2f_runner.py prepare before training')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    expected, baseline = prepare_manifest()
    if manifest != expected:
        raise ValueError('Phase2F manifest mismatch')
    if manifest['test_evaluated'] is not False:
        raise ValueError('Test seal violated')
    if verify_data:
        verify_baseline(baseline)
    return manifest, baseline


def binding(manifest, aid, seed):
    return dict(phase2f_manifest_sha256=sha(MANIFEST), source_sha256=manifest['source_sha256'],
        baseline_protocol_hash=manifest['baseline_sha256'],
        search_space_sha256=manifest['search_space_sha256'], architecture_id=aid, seed=seed,
        epochs=80, test_evaluated=False)


def result_dir(aid, seed):
    return ROOT / 'results/phase2f_multiseed' / aid / f'seed_{seed}'


def summarize(aid, seed):
    folder = result_dir(aid, seed)
    state = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if state['binding'] != binding(manifest, aid, seed) or state['epoch'] != 80:
        raise ValueError(f'Completed checkpoint identity mismatch: {aid} seed{seed}')
    history = state['history']
    if [r['epoch'] for r in history] != list(range(1, 81)):
        raise ValueError(f'Incomplete epoch history: {aid} seed{seed}')
    best = max(history, key=lambda r: r['valid']['mAP50_95'])
    info = json.loads((folder / 'model_info.json').read_text(encoding='utf-8'))
    status = json.loads((folder / 'status.json').read_text(encoding='utf-8'))
    row = dict(architecture_id=aid, seed=seed, best_epoch=best['epoch'],
        **{k: best['valid'][k] for k in METRICS},
        endpoint_mAP50_95=history[-1]['valid']['mAP50_95'],
        Params=info['total_params'], MACs=info['estimated_macs'],
        training_time_seconds=status['elapsed_training_seconds'],
        best_checkpoint=str(folder / 'best.pt'), latest_checkpoint=str(folder / 'latest.pt'),
        best_checkpoint_sha256=sha(folder / 'best.pt'), latest_checkpoint_sha256=sha(folder / 'latest.pt'),
        history_sha256=sha(folder / 'epoch_log.json'), test_evaluated=False)
    dump(folder / 'result.json', row)
    return row


def train(aid, seed, device):
    manifest, baseline = load_manifest(True)
    if seed not in manifest['seeds']['new']:
        raise ValueError('Phase2F permits only new seeds 43 and 44')
    legal = architecture_map()
    allowed = {x['architecture_id'] for x in manifest['architectures']}
    if aid not in allowed or aid not in legal:
        raise ValueError('Architecture is outside the frozen Phase2F set')
    folder = result_dir(aid, seed)
    folder.mkdir(parents=True, exist_ok=True)
    frozen.seed_all(seed)
    model = build_sonar_candidate(ROOT / manifest['search_space'], legal[aid]).to(device)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == 'cuda' else []):
        info = frozen.get_model_info(model, (1, 1, 640, 640))
    info['FLOPs_2x_MACs'] = 2 * info['estimated_macs']
    dump(folder / 'model_info.json', info)
    train_loader = frozen.make_loader(baseline, ROOT / baseline['dataset']['data_dir'], 'train', seed, 0, train=True)
    valid_loader = frozen.make_loader(baseline, ROOT / baseline['dataset']['data_dir'], 'valid', seed, 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    run_binding = binding(manifest, aid, seed)
    history, start_epoch, parents = [], 0, []
    initial_hash = frozen.state_hash(model)
    if (folder / 'latest.pt').exists():
        state = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
        if state['binding'] != run_binding or state['architecture'] != legal[aid]:
            raise ValueError('Phase2F resume binding mismatch')
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        restore_rng(state['rng'], train_loader)
        history, start_epoch = state['history'], state['epoch']
        initial_hash, parents = state['initial_state_sha256'], state['parents']
        if start_epoch < 80:
            parents = parents + [dict(checkpoint=str(folder / 'latest.pt'),
                sha256=sha(folder / 'latest.pt'), epoch=start_epoch)]
    if start_epoch >= 80:
        if not (folder / 'status.json').exists():
            raise ValueError('Epoch80 checkpoint exists without completion status')
        return summarize(aid, seed)
    criterion = DenseAnchorDetectionLoss(model.loss_config(baseline)).to(device)
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_commit = 'unavailable'
    gpu = torch.cuda.get_device_name(torch.cuda.current_device()) if device == 'cuda' else 'cpu'
    invocation_start, elapsed_before = now(), sum(float(r['wall_seconds']) for r in history)
    dump(folder / 'run_binding.json', dict(**run_binding, initialization_seed=seed,
        initialization='independent_from_scratch', git_commit=git_commit, gpu=gpu,
        torch=torch.__version__, resume_policy='epoch_boundary', start=invocation_start))
    for epoch in range(start_epoch, 80):
        tick = time.perf_counter()
        old_best = max((r['valid']['mAP50_95'] for r in history), default=-1.)
        row = train_epoch(model, optimizer, train_loader, baseline, device, None)
        metrics, _ = frozen.evaluate(model, valid_loader, criterion, baseline, device)
        row.update(epoch=epoch + 1, valid=metrics, wall_seconds=time.perf_counter() - tick)
        history.append(row)
        state = dict(binding=run_binding, architecture_id=aid, architecture=legal[aid], seed=seed,
            epoch=epoch + 1, global_step=sum(r['batches'] for r in history),
            model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
            optimizer_state_sha256=state_hash(optimizer.state_dict()), rng=rng_state(train_loader),
            history=history, parents=parents, initial_state_sha256=initial_hash,
            initialization='independent_from_scratch', git_commit=git_commit,
            invocation_start=invocation_start, updated_at=now(), test_evaluated=False)
        if metrics['mAP50_95'] > old_best:
            atomic_save(folder / 'best.pt', state)
        atomic_save(folder / 'latest.pt', state)
        dump(folder / 'epoch_log.json', history)
        print(aid, f'seed={seed}', f'epoch={epoch + 1}/80',
              f'loss={row["losses"]["loss"]:.6f}', flush=True)
    elapsed = elapsed_before + sum(float(r['wall_seconds']) for r in history[start_epoch:])
    dump(folder / 'status.json', dict(status='COMPLETE', architecture_id=aid, seed=seed,
        epochs=80, binding=run_binding, initialization='independent_from_scratch',
        checkpoint_selection='earliest_best_valid_mAP50_95', elapsed_training_seconds=elapsed,
        start=invocation_start, end=now(), test_evaluated=False))
    return summarize(aid, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'train', 'summarize'])
    parser.add_argument('--architecture-id')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == 'prepare':
        manifest, baseline = prepare_manifest()
        print(json.dumps(verify_baseline(baseline)))
        print('Phase2F manifest frozen:', sha(MANIFEST))
        return
    if not args.architecture_id or args.seed is None:
        parser.error('--architecture-id and --seed are required')
    if args.mode == 'train':
        train(args.architecture_id, args.seed, args.device)
    else:
        summarize(args.architecture_id, args.seed)


if __name__ == '__main__':
    main()
