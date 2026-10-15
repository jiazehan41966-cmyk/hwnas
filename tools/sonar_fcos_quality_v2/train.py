"""Preregistered endpoint retention; unchanged v1 training mathematics.

Best, epoch 30, and every epoch 26--30 are retained independently. All diagnostic
traces preserve native model outputs and are evaluation-only. Weights are an FCOS
archive only. C01c is deferred and requires separate E0+quality UATD pretraining;
these FCOS weights cannot initialize the E0/B1 C01c experiment.
"""
from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from tools.sonar_fcos_pilot.common import (environment, read_json, require_manifest, resolve_path,
    seed_everything, sha256_file, utc_now, write_json)
from tools.sonar_fcos_pilot.data import SonarDataset, predictions_to_original
from tools.sonar_fcos_pilot.train import loader, learning_rate, _save_checkpoint
from tools.sonar_fcos_pilot.numerics import AccumulationGroup, numerical_policy, rng_state, restore_rng
from .campaign import DEFAULT_CONFIG, load_config, source_binding, prepare, summarize, verify_development_inputs
from tools.sonar_fcos_quality.model import build_model, model_contract
from tools.sonar_fcos_quality.diagnostics import (StageDiagnosticTracer,
    evaluate_quality_predictions, stage_rows_for_image, _iou, STAGES)
from .analysis import evaluate_checkpoint


RETAINED_EPOCHS = (26, 27, 28, 29, 30)


def add_stage_score_maxima(rows, record, raw_boxes, raw_scores, candidate_boxes,
                           candidate_scores, candidate_labels, nms_keep, top_keep, threshold):
    """Add sufficient per-GT score statistics for a post-hoc operating point.

    Native NMS/TopK scores and box geometry remain unchanged. The maximum score
    among same-class good boxes allows a stricter score cutoff to be replayed
    without retaining all dense boxes. ``raw`` coverage remains score-free.
    """
    labels = np.asarray(candidate_labels)
    scores = np.asarray(candidate_scores)
    nms_mask, top_mask = np.zeros(len(labels), dtype=bool), np.zeros(len(labels), dtype=bool)
    nms_mask[np.asarray(nms_keep, dtype=np.int64)] = True
    top_mask[np.asarray(top_keep, dtype=np.int64)] = True
    valid = ((candidate_boxes[:, 2] > candidate_boxes[:, 0])
             & (candidate_boxes[:, 3] > candidate_boxes[:, 1]) & (labels > 0))
    for row, gt, label in zip(rows, record['boxes'], record['labels']):
        raw_iou, candidate_iou = _iou(raw_boxes, gt), _iou(candidate_boxes, gt)
        masks = {'raw': np.ones(len(raw_boxes), dtype=bool),
            'threshold': raw_scores[:, label] > threshold, 'topk': labels == label,
            'nms': (labels == label) & nms_mask, 'top100': (labels == label) & top_mask,
            'final': (labels == label) & top_mask & valid}
        for stage in STAGES:
            overlaps = raw_iou if stage in ('raw', 'threshold') else candidate_iou
            values = raw_scores[:, label] if stage in ('raw', 'threshold') else scores
            for cutoff, name in ((.75, 'good75_max_score'), (.85, 'good85_max_score')):
                selected = masks[stage] & (overlaps >= cutoff)
                row['stages'][stage][name] = float(values[selected].max()) if selected.any() else None
    return rows


class ScoreStageTracer(StageDiagnosticTracer):
    """Add score maxima to immutable v1 stage definitions and native parity."""

    def _trace(self, heads, anchors, image_shapes, official):
        from torchvision.ops import boxes as box_ops
        if self._batch is None:
            raise RuntimeError('begin_batch must precede traced inference')
        targets, records = self._batch
        if len(targets) != len(official):
            raise ValueError('Traced inference batch changed size')
        for image_index, (target, record) in enumerate(zip(targets, records)):
            raw_boxes, raw_scores, boxes, scores, labels = [], [], [], [], []
            for level, anchor in enumerate(anchors[image_index]):
                logits, regression = heads['cls_logits'][level][image_index], heads['bbox_regression'][level][image_index]
                if self.score_mode == 'quality_sigmoid':
                    logits, regression, anchor = logits.float(), regression.float(), anchor.float()
                probabilities = logits.sigmoid()
                native_scores = (probabilities if self.score_mode == 'quality_sigmoid' else
                    torch.sqrt(probabilities * heads['bbox_ctrness'][level][image_index].sigmoid()))
                flat_scores = native_scores.flatten()
                passed = flat_scores > self.model.score_thresh
                indices = torch.where(passed)[0]
                selected_scores, order = flat_scores[passed].topk(min(len(indices), self.model.topk_candidates))
                indices = indices[order]
                points = torch.div(indices, logits.shape[-1], rounding_mode='floor')
                candidate_labels = indices % logits.shape[-1]
                candidate_boxes = box_ops.clip_boxes_to_image(
                    self.model.box_coder.decode(regression[points], anchor[points]), image_shapes[image_index])
                raw = box_ops.clip_boxes_to_image(self.model.box_coder.decode(regression, anchor), image_shapes[image_index])
                raw_boxes.append(self._inverse(raw, target))
                raw_scores.append(native_scores.detach().cpu().numpy())
                boxes.append(candidate_boxes)
                scores.append(selected_scores)
                labels.append(candidate_labels)
            boxes, scores, labels = torch.cat(boxes), torch.cat(scores), torch.cat(labels)
            nms_keep = box_ops.batched_nms(boxes, scores, labels, self.model.nms_thresh)
            top_keep = nms_keep[:self.model.detections_per_img]
            replay = {'boxes': boxes[top_keep], 'scores': scores[top_keep], 'labels': labels[top_keep]}
            if not all(torch.equal(official[image_index][key], value) for key, value in replay.items()):
                raise RuntimeError('Diagnostic reconstruction differs from authoritative postprocessing')
            raw_boxes, raw_scores = np.concatenate(raw_boxes), np.concatenate(raw_scores)
            candidate_boxes = self._inverse(boxes, target)
            candidate_scores, candidate_labels = scores.detach().cpu().numpy(), labels.detach().cpu().numpy()
            nms_array, top_array = nms_keep.detach().cpu().numpy(), top_keep.detach().cpu().numpy()
            threshold = float(self.model.score_thresh)
            rows = stage_rows_for_image(record, raw_boxes, raw_scores, candidate_boxes,
                                        candidate_labels, nms_array, top_array, threshold)
            self.rows.extend(add_stage_score_maxima(rows, record, raw_boxes, raw_scores,
                candidate_boxes, candidate_scores, candidate_labels, nms_array, top_array, threshold))
            self.images.append(record['image_id'])
        self._batch = None


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
    tracer = ScoreStageTracer(model) if trace else None
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
    diagnostics = tracer.summary(dataset.records) if tracer else None
    if diagnostics is not None:
        diagnostics['stage_rows'] = tracer.rows
        diagnostics['final_no_good_box'] = diagnostics['ground_truths'] - diagnostics['stage_covered_gt']['final']
    return predictions, diagnostics


def artifact(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha256_file(path)}


def verify_artifacts(artifacts):
    for value in artifacts.values():
        if sha256_file(value['path']) != value['sha256']:
            raise RuntimeError('Committed artifact changed: ' + value['path'])


def make_run_receipt(config, variant, seed, source, execution_environment, contract,
                     training_images, validation_images):
    """Single receipt schema shared by training and explicit v1 migration."""
    binding = {'source_binding': source, 'environment': execution_environment, 'variant': variant, 'seed': seed}
    return {**binding, 'binding': binding, 'model': contract, 'epochs': 30,
            'training_images': training_images, 'validation_images': validation_images,
            'test_evaluated': False, 'numerical_policy': numerical_policy(),
            'retention': {'epochs': list(RETAINED_EPOCHS), 'best_selector': 'maximum_validation_map',
                          'primary_endpoints': ['best', 'epoch30'],
                          'stability_endpoint': 'arithmetic_mean_epochs26_30',
                          'retained_per_epoch': ['checkpoint', 'predictions', 'metrics', 'stage_rows', 'evaluation']}}


def save_evaluation(directory, records, predictions, num_classes, diagnostics, *, selection, epoch):
    """Archive per-GT evidence and independently identified endpoint metrics."""
    if diagnostics is None or 'stage_rows' not in diagnostics:
        raise RuntimeError('A retained endpoint requires complete native per-GT stage rows')
    rows = diagnostics['stage_rows']
    # Supplementary analysis must not change the next epoch's stochastic state,
    # even if a future analysis implementation uses randomized computations.
    analysis_rng = rng_state()
    try:
        value = evaluate_checkpoint(records, predictions, num_classes, rows,
                                    selection=selection, epoch=epoch)
    finally:
        restore_rng(analysis_rng)
    expected_gt = sum(len(record['labels']) for record in records)
    final_covered = diagnostics['stage_covered_gt']['final']
    if diagnostics['ground_truths'] != expected_gt or len(rows) != expected_gt:
        raise RuntimeError('Endpoint stage rows do not cover every validation ground truth')
    for key in ('nms_first_loss', 'raw_no_good_box'):
        if value['diagnostics'][key] != diagnostics[key]:
            raise RuntimeError('Endpoint analysis changed the native candidate-coverage diagnostic: ' + key)
    if value['diagnostics'].get('final_no_good_box', expected_gt - final_covered) != expected_gt - final_covered:
        raise RuntimeError('Endpoint analysis disagrees with native final candidate coverage')
    value['diagnostics']['final_no_good_box'] = expected_gt - final_covered
    value['diagnostics']['ground_truths'] = expected_gt
    value['checkpoint_policy'] = {'selection': selection, 'epoch': epoch,
        'test_evaluated': False, 'transfer_use': 'archived_FCOS_only_C01c_deferred_not_E0'}
    stage_path, evaluation_path = directory / 'stage_rows.json', directory / 'evaluation.json'
    write_json(stage_path, rows)
    write_json(evaluation_path, value)
    return value, {'stage_rows': artifact(stage_path), 'evaluation': artifact(evaluation_path)}


def late_epoch_mean(retained_epochs):
    """Arithmetic mean of five separately evaluated checkpoints, not an ensemble."""
    if set(retained_epochs) != {str(epoch) for epoch in RETAINED_EPOCHS}:
        raise RuntimeError('Every epoch 26 through 30 must be retained before completion')
    evaluations = []
    for epoch in RETAINED_EPOCHS:
        artifacts = retained_epochs[str(epoch)]
        verify_artifacts(artifacts)
        evaluations.append(read_json(artifacts['evaluation']['path']))
    metric_keys = ('map', 'ap75', 'ap85')
    diagnostic_keys = ('nms_first_loss', 'raw_no_good_box', 'final_no_good_box', 'wrong_class_fp')
    return {'selection': 'epochs26_30_mean', 'epochs': list(RETAINED_EPOCHS),
        'aggregation': 'arithmetic mean of five independent checkpoint evaluations; not pooled predictions or weight averaging',
        'metrics': {key: sum(value['metrics'][key] for value in evaluations) / 5 for key in metric_keys},
        'diagnostics': {key: sum(value['diagnostics'][key] for value in evaluations) / 5 for key in diagnostic_keys},
        'per_epoch_evaluation_artifacts': {str(epoch): retained_epochs[str(epoch)]['evaluation'] for epoch in RETAINED_EPOCHS}}


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
        for artifacts in result['retained_epochs'].values():
            verify_artifacts(artifacts)
        verify_artifacts(result['evaluation_manifests'])
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
    receipt = make_run_receipt(config, variant, seed, source, execution['environment'], contract,
                               len(train_data), len(val_data))
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
    retained_epochs = {}
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
        retained_epochs = saved.get('retained_epochs', {})
        expected_retained = {str(epoch) for epoch in RETAINED_EPOCHS if epoch <= start_epoch}
        if set(retained_epochs) != expected_retained:
            raise RuntimeError('Resume checkpoint lacks required epoch26--30 retained artifacts')
        verify_artifacts(best_artifacts)
        for artifacts in retained_epochs.values():
            verify_artifacts(artifacts)
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
            retain = epoch + 1 in RETAINED_EPOCHS
            predictions, epoch_diagnostics = predict(model, val_data, config, trace=retain, pause_file=pause_file)
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
            if retain:
                retained_path = checkpoint_dir / f'epoch_{epoch + 1:03d}.pt'
                retained_directory = directory / 'retained' / f'epoch_{epoch + 1:03d}'
                pred_path, metric_path = retained_directory / 'predictions.json', retained_directory / 'metrics.json'
                _save_checkpoint(retained_path, {'model': state, 'binding': binding, 'epoch': epoch,
                    'model_contract': contract, 'classes': manifest['classes']})
                write_json(pred_path, predictions)
                write_json(metric_path, metrics)
                epoch_evaluation, evaluation_artifacts = save_evaluation(retained_directory, val_data.records, predictions,
                    len(manifest['classes']), epoch_diagnostics, selection=f'epoch{epoch + 1}', epoch=epoch + 1)
                if epoch_evaluation['metrics'] != metrics:
                    raise RuntimeError('Retained endpoint metrics differ from authoritative epoch validation')
                retained_epochs[str(epoch + 1)] = {'checkpoint': artifact(retained_path),
                    'predictions': artifact(pred_path), 'metrics': artifact(metric_path), **evaluation_artifacts}
            _save_checkpoint(last_path, {'model': state, 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                'rng': rng_state(), 'history': history, 'best_map': best_map, 'epoch': epoch,
                'binding': binding, 'best_artifacts': best_artifacts, 'best_epoch': best_epoch,
                'retained_epochs': retained_epochs})
            completed_epochs = epoch + 1
            write_json(directory / 'history.json', history)
            status('PARTIAL', best_epoch=best_epoch, best_map=best_map)
            print(f'{variant}_seed{seed} epoch={epoch + 1}/30 loss={total_loss / steps:.5f} '
                  f'AP75={metrics["ap75"]:.4f} AP85={metrics["ap85"]:.4f} mAP={metrics["map"]:.4f}', flush=True)
            del state, predictions, epoch_diagnostics
        if completed_epochs < 30:
            return {'status': 'PARTIAL', 'completed_epochs': completed_epochs}
        status('DIAGNOSING_SELECTED_CHECKPOINT', best_epoch=best_epoch)
        selected = torch.load(best_artifacts['checkpoint']['path'], map_location='cpu', weights_only=False)
        model.load_state_dict(selected['model'])
        del selected
        predictions, diagnostics = predict(model, val_data, config, trace=True, pause_file=pause_file)
        archived_predictions = read_json(best_artifacts['predictions']['path'])
        # The tracer must preserve the model's output, not substitute a second evaluator.
        if predictions != archived_predictions:
            raise RuntimeError('Selected-checkpoint replay differs from committed validation predictions')
        best_evaluation, endpoint_artifacts = save_evaluation(directory / 'evaluations' / 'best',
            val_data.records, predictions, len(manifest['classes']), diagnostics, selection='best', epoch=best_epoch)
        if best_evaluation['metrics'] != read_json(best_artifacts['metrics']['path']):
            raise RuntimeError('Selected-checkpoint evaluation metrics differ from committed validation')
        best_artifacts.update(endpoint_artifacts)
        final_artifacts = retained_epochs['30']
        final_evaluation = read_json(final_artifacts['evaluation']['path'])
        final_predictions = read_json(final_artifacts['predictions']['path'])
        if final_evaluation['metrics'] != read_json(final_artifacts['metrics']['path']):
            raise RuntimeError('Epoch30 evaluation metrics differ from committed validation metrics')
        stability = late_epoch_mean(retained_epochs)
        stability_path = directory / 'evaluations' / 'epochs26_30_mean.json'
        write_json(stability_path, stability)
        evaluation_manifests = {'best': endpoint_artifacts['evaluation'],
                               'epoch30': final_artifacts['evaluation'],
                               'epochs26_30_mean': artifact(stability_path)}
        metrics, diagnostics = best_evaluation['metrics'], best_evaluation['diagnostics']
        if final_evaluation['epoch'] != 30 or len(final_predictions) != len(val_data):
            raise RuntimeError('Final endpoint does not cover epoch30 and the complete validation split')
        result = {**receipt, 'status': 'COMPLETE', 'trained_epochs': 30, 'best_epoch': best_epoch,
                  'validation': metrics, 'final_epoch_validation': history[-1]['validation'],
                  'diagnostics': diagnostics, 'best_artifacts': best_artifacts,
                  'endpoints': {'best': best_evaluation, 'epoch30': final_evaluation,
                                'epochs26_30_mean': stability},
                  'retained_epochs': retained_epochs, 'evaluation_manifests': evaluation_manifests,
                  'completed_at': utc_now()}
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
