"""Resumable, low-priority FCOS quality experiment; never reads the test split."""
from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path
import time

import torch

from tools.sonar_fcos_pilot.common import (environment, read_json, require_manifest, resolve_path,
    seed_everything, sha256_file, utc_now, write_json)
from tools.sonar_fcos_pilot.data import SonarDataset, predictions_to_original
from tools.sonar_fcos_pilot.train import loader, learning_rate, _save_checkpoint
from tools.sonar_fcos_pilot.numerics import AccumulationGroup, numerical_policy, rng_state, restore_rng
from .campaign import DEFAULT_CONFIG, load_config, source_binding, prepare, summarize, verify_development_inputs
from .model import build_model, model_contract
from .diagnostics import StageDiagnosticTracer, evaluate_quality_predictions


class YieldToPrimary(RuntimeError):
    pass


def check_pause(pause_file):
    if pause_file and Path(pause_file).exists():
        raise YieldToPrimary('Primary workload requested GPU; incomplete epoch will replay from last commit')


@torch.inference_mode()
def predict(model, dataset, config, *, trace=False, pause_file=None):
    model.eval()
    predictions = []
    device = torch.device(config['device'])
    records = {int(r['image_id']): r for r in dataset.records}
    tracer = StageDiagnosticTracer(model) if trace else None
    from contextlib import nullcontext
    with tracer if tracer else nullcontext():
        for images, targets in loader(dataset, config):
            check_pause(pause_file)
            if tracer:
                tracer.begin_batch(targets, [records[int(t['image_id'].reshape(-1)[0])] for t in targets])
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=config['amp'] and device.type == 'cuda'):
                outputs = model([image.to(device, non_blocking=True) for image in images])
            predictions.extend(predictions_to_original(t, p) for t, p in zip(targets, outputs))
    return predictions, (tracer.summary(dataset.records) if tracer else None)


def artifact(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha256_file(path)}


def verify_artifacts(artifacts):
    for value in artifacts.values():
        if sha256_file(value['path']) != value['sha256']:
            raise RuntimeError('Committed artifact changed: ' + value['path'])


def train_one(config, variant, seed, epoch_budget=None, pause_file=None):
    if variant not in config['variants'] or seed not in config['seeds']:
        raise ValueError('Run is outside the authorized six-run budget')
    if epoch_budget is not None and epoch_budget < 1:
        raise ValueError('epoch_budget must be positive')
    output = resolve_path(config['output_dir'])
    execution = read_json(output / 'execution_manifest.json')
    source = source_binding(config)
    if source != execution['source_binding'] or environment() != execution['environment']:
        raise RuntimeError('Frozen execution source/config/environment changed')
    smoke_gate = read_json(output / 'smoke_gate.json')
    if (smoke_gate['status'] != 'PASS' or smoke_gate['source_binding'] != source
            or smoke_gate['execution_manifest_sha256'] != sha256_file(output / 'execution_manifest.json')
            or smoke_gate['environment'] != execution['environment']):
        raise RuntimeError('Real GPU smoke does not match this campaign')
    manifest_path, manifest = require_manifest(config)
    torch.set_num_threads(config['torch_threads'])
    seed_everything(seed)
    directory = output / 'runs' / f'{variant}_seed{seed}'
    checkpoint_dir = resolve_path(config['checkpoint_root']) / f'{variant}_seed{seed}'
    binding = {'source_binding': source, 'environment': execution['environment'], 'variant': variant, 'seed': seed}
    summary_path = directory / 'summary.json'
    if summary_path.exists():
        result = read_json(summary_path)
        if result['binding'] != binding:
            raise RuntimeError('Completed run binding changed')
        verify_artifacts(result['best_artifacts'])
        return result
    directory.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / 'last.pt'
    if not last_path.exists():
        verify_development_inputs(manifest)
    train_data = SonarDataset(manifest_path, 'train', config['image_size'])
    val_data = SonarDataset(manifest_path, 'val', config['image_size'])
    model = build_model(variant, len(manifest['classes']) + 1, seed,
        pretrained=config['pretrained'], image_size=config['image_size'], cache_dir=resolve_path(config['weight_cache']))
    contract = model_contract(model)
    if contract['pretrained']['checkpoint_sha256'] != source['pretrained_checkpoint_sha256']:
        raise RuntimeError('Public pretrained weights differ from the frozen execution manifest')
    receipt = {**binding, 'binding': binding, 'model': contract, 'epochs': 30,
               'training_images': len(train_data), 'validation_images': len(val_data),
               'test_evaluated': False, 'numerical_policy': numerical_policy()}
    receipt_path = directory / 'run_manifest.json'
    if receipt_path.exists():
        if read_json(receipt_path) != receipt:
            raise RuntimeError('Run receipt changed')
    else:
        if len(list((output / 'runs').glob('*/run_manifest.json'))) >= 6:
            raise RuntimeError('Six-run budget exhausted')
        write_json(receipt_path, receipt)
    device = torch.device(config['device'])
    model.to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'],
                               momentum=config['momentum'], weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', init_scale=config['amp_initial_scale'],
                                 enabled=config['amp'] and device.type == 'cuda')
    history, best_map, start_epoch, best_artifacts, best_epoch = [], -math.inf, 0, None, None
    if last_path.exists():
        saved = torch.load(last_path, map_location='cpu', weights_only=False)
        if saved['binding'] != binding:
            raise RuntimeError('Resume checkpoint binding changed')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scaler.load_state_dict(saved['scaler'])
        history, best_map, start_epoch = saved['history'], saved['best_map'], saved['epoch'] + 1
        best_artifacts, best_epoch = saved['best_artifacts'], saved['best_epoch']
        verify_artifacts(best_artifacts)
        restore_rng(saved['rng'])
        del saved
    completed_epochs = start_epoch
    events_path = directory / 'numerical_events.json'
    events = read_json(events_path) if events_path.exists() else []

    def record_event(event):
        events.append(event)
        write_json(events_path, events)

    def status(state, **kw):
        write_json(directory / 'status.json', {'status': state, 'completed_epochs': completed_epochs,
                   'updated_at': utc_now(), 'variant': variant, 'seed': seed, **kw})

    try:
        stop_epoch = min(30, start_epoch + epoch_budget) if epoch_budget else 30
        for epoch in range(start_epoch, stop_epoch):
            check_pause(pause_file)
            started = time.monotonic()
            model.train()
            data_loader = loader(train_data, config, shuffle=True, seed=seed + epoch * 100003)
            steps, accumulation = len(data_loader), config['gradient_accumulation']
            optimizer.zero_grad(set_to_none=True)
            total_loss, updates, retries, group = 0., 0, 0, None
            for step, (images, targets) in enumerate(data_loader):
                rate = learning_rate(config, epoch, step, steps)
                group_start = (step // accumulation) * accumulation
                divisor = min(accumulation, steps - group_start)
                if group is None:
                    check_pause(pause_file)
                    group = AccumulationGroup(model, optimizer, scaler, device, divisor, epoch=epoch + 1,
                        group_index=step // accumulation + 1, use_amp=config['amp'] and device.type == 'cuda',
                        event_callback=record_event)
                loss = group.backward(images, targets, step=step, learning_rate=rate)
                if (step + 1) % accumulation == 0 or step + 1 == steps:
                    result = group.finish()
                    total_loss += sum(result['loss_values'])
                    updates += result['optimizer_steps']
                    retries += result['retries']
                    group = None
                if step % 100 == 0 or step + 1 == steps:
                    status('TRAINING', epoch=epoch + 1, step=step + 1, steps=steps, loss=loss,
                           optimizer_steps=updates, overflow_retries=retries, amp_scale=float(scaler.get_scale()))
            if updates != math.ceil(steps / accumulation):
                raise RuntimeError('Incorrect optimizer update count')
            status('VALIDATING', epoch=epoch + 1)
            predictions, _ = predict(model, val_data, config, pause_file=pause_file)
            metrics = evaluate_quality_predictions(val_data.records, predictions, len(manifest['classes']))
            if metrics['map'] is None or not math.isfinite(metrics['map']):
                raise RuntimeError('Validation mAP is undefined')
            improved = metrics['map'] > best_map
            history.append({'epoch': epoch + 1, 'loss': total_loss / steps, 'validation': metrics,
                            'learning_rate': rate, 'optimizer_steps': updates, 'overflow_retries': retries,
                            'amp_scale': float(scaler.get_scale()), 'seconds': time.monotonic() - started})
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            if improved:
                best_map, best_epoch = metrics['map'], epoch + 1
                best_path = checkpoint_dir / f'best_epoch_{epoch + 1:03d}.pt'
                pred_path = directory / 'validation' / f'epoch_{epoch + 1:03d}_predictions.json'
                metric_path = directory / 'validation' / f'epoch_{epoch + 1:03d}_metrics.json'
                _save_checkpoint(best_path, {'model': state, 'binding': binding, 'epoch': epoch,
                                             'model_contract': contract, 'classes': manifest['classes']})
                write_json(pred_path, predictions)
                write_json(metric_path, metrics)
                best_artifacts = {'checkpoint': artifact(best_path), 'predictions': artifact(pred_path), 'metrics': artifact(metric_path)}
            _save_checkpoint(last_path, {'model': state, 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                'rng': rng_state(), 'history': history, 'best_map': best_map, 'epoch': epoch,
                'binding': binding, 'best_artifacts': best_artifacts, 'best_epoch': best_epoch})
            completed_epochs = epoch + 1
            write_json(directory / 'history.json', history)
            status('PARTIAL', best_epoch=best_epoch, best_map=best_map)
            print(f'{variant}_seed{seed} epoch={epoch + 1}/30 loss={total_loss / steps:.5f} '
                  f'AP75={metrics["ap75"]:.4f} AP85={metrics["ap85"]:.4f} mAP={metrics["map"]:.4f}', flush=True)
            del state, predictions
        if completed_epochs < 30:
            return {'status': 'PARTIAL', 'completed_epochs': completed_epochs}
        status('DIAGNOSING_SELECTED_CHECKPOINT', best_epoch=best_epoch)
        selected = torch.load(best_artifacts['checkpoint']['path'], map_location='cpu', weights_only=False)
        model.load_state_dict(selected['model'])
        del selected
        predictions, diagnostics = predict(model, val_data, config, trace=True, pause_file=pause_file)
        metrics = evaluate_quality_predictions(val_data.records, predictions, len(manifest['classes']))
        archived_predictions = read_json(best_artifacts['predictions']['path'])
        # The tracer must preserve the model's output, not substitute a second evaluator.
        if predictions != archived_predictions:
            raise RuntimeError('Selected-checkpoint replay differs from committed validation predictions')
        diagnostics['wrong_class_fp'] = metrics['wrong_class_fp']
        diagnostic_path = directory / 'diagnostics.json'
        write_json(diagnostic_path, diagnostics)
        best_artifacts['diagnostics'] = artifact(diagnostic_path)
        result = {**receipt, 'status': 'COMPLETE', 'trained_epochs': 30, 'best_epoch': best_epoch,
                  'validation': metrics, 'final_epoch_validation': history[-1]['validation'],
                  'diagnostics': diagnostics, 'best_artifacts': best_artifacts, 'completed_at': utc_now()}
        write_json(summary_path, result)
        status('COMPLETE', best_epoch=best_epoch)
        return result
    except YieldToPrimary:
        status('PAUSED_FOR_PRIMARY', resume_from_last_complete_epoch=last_path.exists())
        raise
    except BaseException as error:
        status('ERROR', error=str(error), resume_from_last_complete_epoch=last_path.exists())
        raise
    finally:
        del model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def smoke(config):
    """Real train-image backward/update and val-image traced eval, no formal state."""
    torch.set_num_threads(config['torch_threads'])
    manifest_path, manifest = require_manifest(config)
    train_data, val_data = [SonarDataset(manifest_path, s, 640) for s in ('train', 'val')]
    device = torch.device(config['device'])
    witnesses = []
    for variant in config['variants']:
        model = build_model(variant, len(manifest['classes']) + 1, 42,
                            pretrained=True, cache_dir=resolve_path(config['weight_cache'])).to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
        scaler = torch.amp.GradScaler('cuda', init_scale=config['amp_initial_scale'], enabled=config['amp'] and device.type == 'cuda')
        group = AccumulationGroup(model, optimizer, scaler, device, 4, epoch=0, group_index=0,
                                  use_amp=config['amp'] and device.type == 'cuda')
        for index in range(4):
            sample, target = train_data[index]
            group.backward([sample], [target], step=index, learning_rate=config['learning_rate'])
        updated = group.finish()
        if updated['optimizer_steps'] != 1:
            raise RuntimeError('Smoke has no optimizer update')
        probe_data = SonarDataset(manifest_path, 'val', 640)
        probe_data.samples = probe_data.samples[:2]
        probe_data.records = probe_data.samples
        predictions, diagnostics = predict(model, probe_data, config, trace=True)
        metrics = evaluate_quality_predictions(probe_data.records, predictions, len(manifest['classes']))
        witnesses.append({'variant': variant, 'model': model_contract(model), 'update': updated,
                          'validation_images': len(predictions), 'metrics': metrics, 'diagnostics': diagnostics})
        del model, optimizer, scaler, group
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {'status': 'PASS', 'source_binding': source_binding(config), 'environment': environment(),
              'execution_manifest_sha256': sha256_file(resolve_path(config['output_dir']) / 'execution_manifest.json'),
              'witnesses': witnesses, 'formal_training': False, 'test_evaluated': False}
    write_json(resolve_path(config['output_dir']) / 'smoke_gate.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'smoke', 'run', 'summarize'))
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--variant', choices=('AAA', 'QFL_DFL'))
    parser.add_argument('--seed', type=int, choices=(42, 43, 44))
    parser.add_argument('--epoch-budget', type=int)
    parser.add_argument('--pause-file', default=os.environ.get('SONAR_FCOS_QUALITY_PAUSE_FILE'))
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        if args.mode == 'run':
            result = train_one(config, args.variant, args.seed, args.epoch_budget, args.pause_file)
        else:
            result = globals()[args.mode](config)
        print({'status': result.get('status', result.get('decision', {}).get('status')), 'mode': args.mode}, flush=True)
    except YieldToPrimary:
        raise SystemExit(75)


if __name__ == '__main__':
    main()
