"""One serial worker per explicitly assigned existing GPU; no automatic retry."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from tools.sonar_joint192_validation import train


def smoke_gate(result_root):
    reports = []
    for arm in train.PROTOTYPES:
        folder = train.result_dir(result_root, arm, 42, False)
        report = json.loads((folder / 'smoke_report.json').read_text())
        if (report['status'] != 'PASS' or report['binding'] != train.binding(result_root, arm, 42, False)
                or report['formal_eligible'] is not False or report['test_evaluated'] is not False
                or report['environment']['device'] != 'cuda'
                or len(report['train_batches']) != 2
                or any(b['size'] != 4 for b in report['train_batches'])
                or report['validation_smoke']['num_images'] != 4):
            raise ValueError('Actual bound two-batch CUDA smoke required: ' + arm)
        reports.append(report)
    for key in ('head_sha256', 'lateral_sha256'):
        if len({r['initialization_identity'][key] for r in reports}) != 1:
            raise ValueError('Shared initialization mismatch: ' + key)
    for i in range(2):
        for key in ('augmented_images_sha256', 'targets_sha256'):
            if len({r['train_batches'][i][key] for r in reports}) != 1:
                raise ValueError('Actual paired augmented batch mismatch: ' + key)
    rngs = [json.loads((train.result_dir(result_root, a, 42, False) / 'training_rng_identity.json').read_text())
            for a in train.PROTOTYPES]
    if any(r != rngs[0] for r in rngs):
        raise ValueError('Training RNG identity mismatch')
    cfg, _ = train.load_config()
    paired = {}
    for seed in train.SEEDS:
        states = {a: train.build_model(cfg['arms'][a], seed).state_dict() for a in train.PROTOTYPES}
        shared = set.intersection(*(set(s) for s in states.values()))
        for name in shared:
            if any(not torch.equal(states['B0'][name], states[a][name]) for a in train.PROTOTYPES):
                raise ValueError(f'Unpaired initialization: seed{seed}/{name}')
        paired[str(seed)] = dict(common_tensors=len(shared), all_identical=True)
    return dict(smoke_sha256={a: train.sha(train.result_dir(result_root, a, 42, False) / 'smoke_report.json')
                             for a in train.PROTOTYPES}, shared_initialization=paired, paired_augmented_batches=True)


def run(result_root, gpu):
    train.load_manifest(result_root, True)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('Physical GPU assignment must match CUDA_VISIBLE_DEVICES')
    cfg, _ = train.load_config()
    arms = cfg['staging']['workers'][gpu]
    gate = smoke_gate(result_root)
    worker = result_root / f'worker_gpu{gpu}'
    worker.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (worker / 'queue.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = worker / 'queue_status.json'
        if path.exists():
            raise FileExistsError('Worker already invoked; preserve its evidence')
        state = dict(status='RUNNING', pid=os.getpid(), gpu=gpu, arms=arms, seeds=list(train.SEEDS),
                     execution_manifest_sha256=train.sha(result_root / 'execution_manifest.json'),
                     completed=[], active=None, gate=gate, started_at=train.now(), test_evaluated=False)
        train.dump(path, state)
        try:
            for seed in train.SEEDS:
                for arm in arms:
                    folder = train.result_dir(result_root, arm, seed)
                    folder.mkdir(parents=True, exist_ok=True)
                    if (folder / 'run_binding.json').exists() or (folder / 'latest.pt').exists():
                        raise FileExistsError('Existing training invocation refused')
                    command = [sys.executable, str(Path(__file__).with_name('train.py')), 'train',
                               '--prototype', arm, '--seed', str(seed), '--device', 'cuda',
                               '--result-root', str(result_root)]
                    with (folder / 'stdout.log').open('a') as log:
                        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                        state.update(active=dict(arm=arm, seed=seed), child_pid=child.pid, updated_at=train.now())
                        train.dump(path, state)
                        if child.wait():
                            raise RuntimeError(f'{arm}/seed{seed} failed; no automatic retry')
                    train.summarize(result_root, arm, seed)
                    state['completed'].append(dict(arm=arm, seed=seed, result_sha256=train.sha(folder / 'result.json')))
                    state.update(active=None, child_pid=None, updated_at=train.now())
                    train.dump(path, state)
            state.update(status='COMPLETE', updated_at=train.now())
            train.dump(path, state)
        except Exception as error:
            state.update(status='FAILED_NO_AUTORETRY', error=str(error), updated_at=train.now())
            train.dump(path, state)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', type=Path, default=train.DEFAULT_ROOT)
    parser.add_argument('--gpu', type=int, choices=[2, 3], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.result_root.resolve(), args.gpu)
