"""Copy and rebind the paused v1 state without altering its numerical contents."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

from tools.sonar_fcos_pilot.common import ROOT, read_json, resolve_path, sha256_file, utc_now, write_json
from tools.sonar_fcos_pilot.recovery import state_digest
from tools.sonar_fcos_pilot.train import _save_checkpoint

PARENT_ROOT = Path('/home/jiajia/experiments/sonar_fcos_quality_20261012_01')
PARENT_RESULT = 'results/sonar_fcos_quality_20261012_01'
PARENT_CONFIG = 'configs/sonar_fcos_quality_20261012.json'
PARENT_EXECUTION_SHA = '7f3efcee40a56304cb963baa0e9874b35ea0027cef38a277cc66836d813014b8'
NUMERICAL_KEYS = ('model', 'optimizer', 'scaler', 'rng', 'history', 'best_map', 'epoch', 'best_epoch')


def numerical_digest(checkpoint):
    return state_digest({key: checkpoint[key] for key in NUMERICAL_KEYS if key in checkpoint})


def rebind_checkpoint(checkpoint, binding, best_artifacts=None):
    """Bindings/paths may change; model, optimizer, RNG and complete history may not."""
    before = numerical_digest(checkpoint)
    migrated = copy.deepcopy(checkpoint)
    migrated['binding'] = copy.deepcopy(binding)
    if best_artifacts is not None:
        migrated['best_artifacts'] = copy.deepcopy(best_artifacts)
    if numerical_digest(migrated) != before:
        raise RuntimeError('Rebinding changed numerical training state')
    return migrated


def retention_check(completed_epochs):
    if completed_epochs < 1 or completed_epochs > 30:
        raise ValueError('Migration requires a valid committed parent epoch')
    return {'completed_epochs': completed_epochs,
            'late_retention_gap': completed_epochs >= 26,
            'late_retention_epochs': list(range(26, 31)),
            'late_retention_fully_prospective': completed_epochs < 26}


def ensure_parent_stopped(parent):
    parent = Path(parent).resolve()
    result = parent / PARENT_RESULT
    if not (result / 'STOP_QUEUE').is_file():
        raise RuntimeError('The parent needs its explicit graceful-stop request')
    live = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if (entry / 'cwd').resolve() == parent:
                args = (entry / 'cmdline').read_bytes().split(bytes([0]))
                if any(b'tools.sonar_fcos_quality' in arg for arg in args):
                    live.append(int(entry.name))
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            pass
    if live:
        raise RuntimeError('Parent training or scheduler still alive; migration refused')
    status = read_json(result / 'queue_status.json')
    if status.get('status') != 'PAUSED_BY_REQUEST':
        raise RuntimeError('Parent queue has not acknowledged the pause')
    return {'status': 'PARENT_PAUSED', 'live_parent_pids': [], 'forced_signal_sent': False}


def checked_copy(source, target, expected_sha, root):
    source, target, root = Path(source), Path(target), Path(root).resolve()
    if not target.resolve().is_relative_to(root):
        raise ValueError('Migration target escapes the new isolated root')
    if sha256_file(source) != expected_sha:
        raise ValueError('Prepared migration artifact changed')
    if target.exists():
        if sha256_file(target) != expected_sha:
            raise FileExistsError('Existing migration target differs: ' + str(target))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.migration-copy')
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != expected_sha:
        raise RuntimeError('Migration artifact copy differs')
    temporary.replace(target)


def migrate(config_path):
    import torch
    from .campaign import load_config, source_binding
    from .train import make_run_receipt
    config = load_config(config_path)
    new_root = ROOT.resolve()
    output = resolve_path(config['output_dir'])
    checkpoint_root = resolve_path(config['checkpoint_root'])
    if not output.is_relative_to(new_root) or not checkpoint_root.is_relative_to(new_root):
        raise ValueError('Migration writes only the new isolated campaign root')
    if new_root == PARENT_ROOT.resolve():
        raise RuntimeError('Migration must never execute in the parent root')
    parent = PARENT_ROOT
    parent_output = parent / PARENT_RESULT
    paused = ensure_parent_stopped(parent)
    parent_manifest_path = parent_output / 'execution_manifest.json'
    if sha256_file(parent_manifest_path) != PARENT_EXECUTION_SHA:
        raise RuntimeError('Parent execution manifest differs from the preregistration exposure receipt')
    previous = read_json(parent_manifest_path)
    parent_source = previous['source_binding']
    for relative, digest in parent_source['sources'].items():
        if sha256_file(parent / relative) != digest or sha256_file(new_root / relative) != digest:
            raise RuntimeError('Parent helper/model implementation differs: ' + relative)
    parent_config = read_json(parent / PARENT_CONFIG)
    for key in ('quality', 'variants', 'seeds', 'epochs', 'max_formal_runs', 'expected_manifest_sha256'):
        if config[key] != parent_config[key]:
            raise RuntimeError('Migration may not change training/model/budget: ' + key)
    if list((parent_output / 'runs').glob('QFL_DFL_seed*/run_manifest.json')):
        raise RuntimeError('A quality-variant parent run already started; prospective migration gate fails')
    execution = read_json(output / 'execution_manifest.json')
    source = source_binding(config)
    if execution['source_binding'] != source or execution['environment'] != previous['environment']:
        raise RuntimeError('New execution source or runtime does not match the prepared migration')
    launch = read_json(parent_output / 'launch.json')
    if launch.get('eligible_gpu_indices') != [0]:
        raise RuntimeError('Parent physical GPU affinity is not the verified GPU 0')
    receipt_path = output / 'migration.json'
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt.get('status') != 'MIGRATION_PASS' or receipt['new_source_binding'] != source:
            raise RuntimeError('Existing migration binding differs')
        for artifact in receipt['artifacts']:
            if sha256_file(artifact['new_path']) != artifact['new_sha256']:
                raise RuntimeError('Committed migration artifact changed')
        return receipt
    transaction_path = output / 'migration_staging' / 'transaction.json'
    if transaction_path.exists():
        transaction = read_json(transaction_path)
        if transaction['receipt']['new_source_binding'] != source:
            raise RuntimeError('Prepared transaction belongs to different source')
        for original in transaction['parent_inputs']:
            if sha256_file(original['path']) != original['sha256']:
                raise RuntimeError('Parent input changed since transaction preparation')
    else:
        runs = sorted((parent_output / 'runs').glob('AAA_seed*/run_manifest.json'))
        if not runs:
            raise RuntimeError('Expected at least one committed parent AAA run')
        if len(runs) > 3:
            raise RuntimeError('Unexpected parent run budget')
        staging = output / 'migration_staging' / 'prepared'
        artifacts, migrated_runs, writes, parent_inputs = [], [], [], []

        def prepared(path, final):
            digest = sha256_file(path)
            writes.append({'staged_path': str(path), 'final_path': str(final), 'sha256': digest})
            return {'path': str(final), 'sha256': digest}

        for run_manifest_path in runs:
            old_run = run_manifest_path.parent
            old_receipt = read_json(run_manifest_path)
            variant, seed = old_receipt['variant'], old_receipt['seed']
            run_id = f'{variant}_seed{seed}'
            if variant != 'AAA' or seed not in (42, 43, 44) or old_run.name != run_id:
                raise RuntimeError('Unexpected parent run identity')
            old_binding = old_receipt['binding']
            if old_binding['source_binding'] != parent_source or old_binding['environment'] != previous['environment']:
                raise RuntimeError('Parent run binding changed')
            parent_cp_root = Path(parent_config['checkpoint_root'])
            if not parent_cp_root.is_absolute():
                parent_cp_root = parent / parent_cp_root
            old_checkpoint = parent_cp_root / run_id / 'last.pt'
            saved = torch.load(old_checkpoint, map_location='cpu', weights_only=False)
            if saved['binding'] != old_binding:
                raise RuntimeError('Parent committed checkpoint binding differs')
            completed = saved['epoch'] + 1
            retention = retention_check(completed)
            if retention['late_retention_gap']:
                write_json(output / 'migration_retention_gap.json', {'status': 'BLOCKED_LATE_RETENTION_GAP',
                    'run_id': run_id, **retention})
                raise RuntimeError('Parent already reached epoch 26; required late snapshots were not retained')
            if len(saved['history']) != completed or read_json(old_run / 'history.json') != saved['history']:
                raise RuntimeError('Parent history sidecar differs from committed numerical state')
            new_receipt = make_run_receipt(config, variant, seed, source, execution['environment'],
                old_receipt['model'], old_receipt['training_images'], old_receipt['validation_images'])
            binding = new_receipt['binding']
            new_run, new_cp = output / 'runs' / run_id, checkpoint_root / run_id
            stage_run, stage_cp = staging / run_id / 'run', staging / run_id / 'checkpoints'
            stage_run.mkdir(parents=True, exist_ok=True)
            stage_cp.mkdir(parents=True, exist_ok=True)
            best_artifacts = {}
            for key, item in saved['best_artifacts'].items():
                old_path = Path(item['path']).resolve()
                if sha256_file(old_path) != item['sha256']:
                    raise RuntimeError('Parent selected artifact changed')
                parent_inputs.append({'path': str(old_path), 'sha256': item['sha256']})
                if key == 'checkpoint':
                    if not old_path.is_relative_to((parent_cp_root / run_id).resolve()):
                        raise ValueError('Parent checkpoint escapes its run')
                    blob = torch.load(old_path, map_location='cpu', weights_only=False)
                    if blob['binding'] != old_binding:
                        raise RuntimeError('Parent selected checkpoint binding differs')
                    migrated_best = rebind_checkpoint(blob, binding)
                    new_path, stage_path = new_cp / old_path.name, stage_cp / old_path.name
                    _save_checkpoint(stage_path, migrated_best)
                    if numerical_digest(torch.load(stage_path, map_location='cpu', weights_only=False)) != numerical_digest(blob):
                        raise RuntimeError('Selected checkpoint numerical contents changed during save')
                else:
                    if not old_path.is_relative_to(old_run.resolve()):
                        raise ValueError('Parent selected metadata escapes its run')
                    relative = old_path.relative_to(old_run.resolve())
                    new_path, stage_path = new_run / relative, stage_run / relative
                    stage_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(old_path, stage_path)
                    if sha256_file(stage_path) != item['sha256']:
                        raise RuntimeError('Selected prediction/metric bytes changed')
                best_artifacts[key] = prepared(stage_path, new_path)
                artifacts.append({'kind': key, 'run_id': run_id, 'old_path': str(old_path),
                    'old_sha256': item['sha256'], 'new_path': str(new_path), 'new_sha256': sha256_file(stage_path)})
            migrated_last = rebind_checkpoint(saved, binding, best_artifacts)
            migrated_last['retained_epochs'] = {}
            staged_last = stage_cp / 'last.pt'
            _save_checkpoint(staged_last, migrated_last)
            state_before = numerical_digest(saved)
            state_after = numerical_digest(torch.load(staged_last, map_location='cpu', weights_only=False))
            if state_after != state_before:
                raise RuntimeError('Full model/optimizer/scaler/RNG/history changed during migration')
            prepared(staged_last, new_cp / 'last.pt')
            artifacts.append({'kind': 'last_checkpoint', 'run_id': run_id, 'old_path': str(old_checkpoint),
                'old_sha256': sha256_file(old_checkpoint), 'new_path': str(new_cp / 'last.pt'),
                'new_sha256': sha256_file(staged_last)})
            parent_inputs.append({'path': str(old_checkpoint), 'sha256': sha256_file(old_checkpoint)})
            for name in ('history.json', 'numerical_events.json'):
                if (old_run / name).exists():
                    shutil.copy2(old_run / name, stage_run / name)
                    prepared(stage_run / name, new_run / name)
                    parent_inputs.append({'path': str(old_run / name), 'sha256': sha256_file(old_run / name)})
            write_json(stage_run / 'run_manifest.json', new_receipt)
            prepared(stage_run / 'run_manifest.json', new_run / 'run_manifest.json')
            write_json(stage_run / 'status.json', {'status': 'MIGRATED_PARTIAL', 'completed_epochs': completed,
                'variant': variant, 'seed': seed, 'updated_at': utc_now(), 'resume_from_last_complete_epoch': True})
            prepared(stage_run / 'status.json', new_run / 'status.json')
            migrated_runs.append({'run_id': run_id, 'variant': variant, 'seed': seed, 'physical_gpu': 0,
                'completed_epochs': completed, 'resume_at_epoch': completed + 1,
                'numerical_state_sha256_before': state_before, 'numerical_state_sha256_after': state_after,
                'numerical_state_exact': True, 'old_binding': old_binding, 'new_binding': binding,
                'new_candidate_run': False, **retention})
        receipt = {'status': 'MIGRATION_PASS', 'created_at': utc_now(), 'parent_root': str(parent),
            'new_root': str(new_root), 'parent_execution_manifest_sha256': PARENT_EXECUTION_SHA,
            'new_execution_manifest_sha256': sha256_file(output / 'execution_manifest.json'),
            'new_source_binding': source, 'parent_source_binding': parent_source, 'parent_pause': paused,
            'migrated_runs': migrated_runs, 'artifacts': artifacts, 'total_logical_runs': 6,
            'new_candidate_runs_from_migration': 0,
            'remaining_epoch_budget': 180 - sum(r['completed_epochs'] for r in migrated_runs),
            'validation_outputs_exist': True, 'validation_metric_values_examined': False,
            'qfl_parent_started': False, 'parent_files_modified': False,
            'incomplete_parent_epoch_discarded': True}
        transaction = {'receipt': receipt, 'writes': writes, 'parent_inputs': parent_inputs}
        write_json(transaction_path, transaction)
    # Prepared transaction is resumable; only matching new-root files are accepted.
    for item in transaction['writes']:
        staged = Path(item['staged_path']).resolve()
        if not staged.is_relative_to((output / 'migration_staging').resolve()):
            raise ValueError('Prepared migration source escapes its staging area')
        checked_copy(staged, item['final_path'], item['sha256'], new_root)
    ensure_parent_stopped(parent)
    for original in transaction['parent_inputs']:
        if sha256_file(original['path']) != original['sha256']:
            raise RuntimeError('Parent state changed during migration')
    write_json(receipt_path, transaction['receipt'])
    return transaction['receipt']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    result = migrate(args.config)
    # Keep prospective monitoring free of validation outcomes.
    print(json.dumps({'status': result['status'], 'migrated_runs': len(result['migrated_runs']),
        'completed_parent_epochs': sum(r['completed_epochs'] for r in result['migrated_runs']),
        'remaining_epoch_budget': result['remaining_epoch_budget'], 'numerical_state_exact': True,
        'validation_metric_values_examined': False}))


if __name__ == '__main__':
    main()
