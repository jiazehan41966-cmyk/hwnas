"""CPU-only paired seed-and-image bootstrap for the frozen late-epoch protocol.

All-area 101-point AP uses the project's exact per-image matcher. Bootstrap
copies retain their own GT/detections; global score sorting uses torch's same
argsort (including its tie ordering), and recall/precision remain float32.
Every actual input is checked against the slow public evaluator before any CI
is reported. No model, checkpoint inference, training, or remote access occurs.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from hwnas_fpga.training import detection_coco_eval as coco

EPOCHS = (76, 77, 78, 79, 80)
HEADS = ('obj_softmax', 'quality_aware')
SEEDS = {'E0': (42, 43, 44, 45, 46), 'YOLOv9_T': (42, 43, 44)}
COMMON_SEEDS = (42, 43, 44)
NAMES = ('cylider', 'manta')
METRICS = ('mAP50_95', 'AP50', 'AP75', 'cylider_AP50', 'cylider_AP75',
           'cylider_AP50_95', 'manta_AP50', 'manta_AP75', 'manta_AP50_95')
PARITY_TOLERANCE = 2e-6
RECALL_GRID = np.asarray(coco.COCO_RECALL_POINTS, dtype=np.float32)


class EvaluatorParityError(ValueError):
    """No result is publishable when the accelerator differs from the oracle."""


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')


def _metric_vector(ap):
    """AP shape (2,10); absent classes are omitted, as in the public evaluator."""
    present = ~np.isnan(ap).all(axis=1)
    total = ap[present].mean(axis=0, dtype=np.float64) if present.any() else np.zeros(10)
    out = [float(total.mean()), float(total[0]), float(total[5])]
    for values in ap:
        out += [float(values[0]), float(values[5]), float(values.mean(dtype=np.float64))]
    return np.asarray(out, dtype=np.float64)


def reference_metrics(records, indices=None):
    """Slow independent public evaluator, including duplicated sampled images."""
    selected = records if indices is None else [records[int(i)] for i in indices]
    result = coco.evaluate_detection_predictions_coco(
        [r['prediction'] for r in selected], [r['target'] for r in selected],
        num_classes=2, class_names=NAMES, area_ranges={'all': (0., float('inf'))},
        max_detections=100)
    ap = np.full((2, 10), np.nan, dtype=np.float64)
    for cls, name in enumerate(NAMES):
        by_iou = result['per_class_ap']['all'].get(name)
        if by_iou:
            ap[cls] = [by_iou[f'{threshold:.2f}'] for threshold in coco.COCO_IOU_THRESHOLDS]
    return _metric_vector(ap)


class PreparedAP:
    """Cache exact per-image matches, then replay original global ranking.

    This avoids repeating IoU/matching 2,000 times. It deliberately does not
    compress distinct score ties or replace float32 recall comparisons with
    rational arithmetic. Sample order and duplicate copies remain explicit.
    """
    def __init__(self, records):
        self.records = records
        self.n = len(records)
        predictions = coco._stack([r['prediction'] for r in records], with_scores=True)
        targets = coco._stack([r['target'] for r in records], with_scores=False)
        self.classes = []
        for cls in range(2):
            counts = np.asarray([min(int((p['labels'] == cls).sum()), 100)
                                 for p in predictions], dtype=np.int64)
            starts = np.r_[0, counts.cumsum()]
            positives = np.asarray([int((t['labels'] == cls).sum()) for t in targets], dtype=np.int64)
            flags, scores0 = [], None
            for threshold in coco.COCO_IOU_THRESHOLDS:
                scores, tp, fp, total = coco._evaluate_class_area(
                    predictions, targets, class_index=cls, iou_threshold=threshold,
                    area_range=(0., float('inf')), max_detections=100)
                if (len(scores) != int(counts.sum()) or total != int(positives.sum())
                        or not torch.equal(tp + fp, torch.ones_like(tp))):
                    raise EvaluatorParityError('Unexpected ignored all-area detections; use public evaluator')
                if scores0 is None:
                    scores0 = scores.numpy().copy()
                elif not np.array_equal(scores0, scores.numpy()):
                    raise EvaluatorParityError('Score order changed across IoU thresholds')
                flags.append(tp.numpy())
            self.classes.append(dict(scores=scores0, tp=np.stack(flags, axis=1),
                gt=positives, indices=[np.arange(starts[i], starts[i+1]) for i in range(self.n)],
                has_score_ties=len(np.unique(scores0)) != len(scores0)))

    def evaluate_one(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= self.n):
            raise ValueError('Image sample indices must be a nonempty in-range vector')
        ap = np.full((2, 10), np.nan, dtype=np.float64)
        for cls, item in enumerate(self.classes):
            positives = int(item['gt'][indices].sum())
            if positives == 0:
                continue
            copied = np.concatenate([item['indices'][int(i)] for i in indices])
            if not len(copied):
                ap[cls] = 0.
                continue
            # Exactly the same global unstable tie ordering as the project evaluator.
            order = torch.from_numpy(item['scores'][copied]).argsort(descending=True).numpy()
            tp = np.cumsum(item['tp'][copied[order]], axis=0, dtype=np.float32)
            denominator = np.arange(1, len(copied)+1, dtype=np.float32)[:, None]
            precision = tp / denominator
            recall = tp / np.float32(positives)
            envelope = np.maximum.accumulate(precision[::-1], axis=0)[::-1]
            for q in range(10):
                positions = np.searchsorted(recall[:, q], RECALL_GRID, side='left')
                sampled = np.zeros(101, dtype=np.float32)
                ok = positions < len(copied)
                sampled[ok] = envelope[positions[ok], q]
                ap[cls, q] = float(sampled.mean(dtype=np.float32))
        return _metric_vector(ap)

    def evaluate(self, draws):
        draws = np.asarray(draws)
        if draws.ndim != 2:
            raise ValueError('Image draws must be a 2D matrix')
        return np.stack([self.evaluate_one(row) for row in draws])

    def validate(self, samples):
        checks = []
        for name, indices in samples.items():
            exact = reference_metrics(self.records, indices)
            replay = self.evaluate_one(indices)
            if not np.array_equal(np.isnan(exact), np.isnan(replay)):
                raise EvaluatorParityError(f'{name}: absent-class semantics differ from public evaluator')
            finite = np.isfinite(exact)
            error = float(np.max(np.abs(exact[finite] - replay[finite]))) if finite.any() else 0.
            if error > PARITY_TOLERANCE:
                raise EvaluatorParityError(f'{name}: original evaluator parity error {error:g}')
            checks.append(dict(sample=name, max_absolute_error=error, passed=True,
                               duplicate_images=len(set(map(int, indices))) < len(indices)))
        return checks


def bootstrap_plan(image_indices, seed_ids, *, replicates=2000, random_seed=20261012):
    """One image draw per replicate is shared across all seeds, heads and epochs."""
    ids = np.asarray(image_indices, dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0 or len(set(ids.tolist())) != len(ids):
        raise ValueError('Bootstrap population must contain unique image indices')
    if replicates < 1 or len(seed_ids) < 2 or len(set(seed_ids)) != len(seed_ids):
        raise ValueError('Bootstrap requires positive replicates and distinct paired seeds')
    child = np.random.SeedSequence(random_seed).spawn(2)
    image_rng, seed_rng = map(np.random.default_rng, child)
    image_draws = ids[image_rng.integers(0, len(ids), size=(replicates, len(ids)))]
    seed_draws = seed_rng.integers(0, len(seed_ids), size=(replicates, len(seed_ids)))
    return dict(image_draws=image_draws, seed_draws=seed_draws,
                seed_ids=list(seed_ids), random_seed=int(random_seed), replicates=int(replicates))


def reduce_paired_bootstrap(seed_epoch_values, seed_draws):
    """(replicate, paired_seed, epoch, metric) -> both CI sample definitions."""
    values = np.asarray(seed_epoch_values, dtype=np.float64)
    draws = np.asarray(seed_draws, dtype=np.int64)
    if (values.ndim != 4 or draws.shape != values.shape[:2]
            or np.any(draws < 0) or np.any(draws >= values.shape[1])):
        raise ValueError('Paired seed bootstrap shape/index mismatch')
    stable = values.mean(axis=2)
    image_only = stable.mean(axis=1)
    hierarchical = stable[np.arange(len(stable))[:, None], draws].mean(axis=1)
    return hierarchical, image_only


def interval(values, confidence=.95):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return dict(low=None, high=None, confidence=confidence, n_valid=0,
                    excludes_zero=False, conclusion='UNDEFINED_NO_CLASS_GT')
    low, high = np.quantile(finite, [(1-confidence)/2, (1+confidence)/2], method='linear')
    excludes_zero = bool(low > 0 or high < 0)
    return dict(low=float(low), high=float(high), confidence=float(confidence),
                n_valid=len(finite), excludes_zero=excludes_zero,
                conclusion=('POSITIVE_DIFFERENCE' if low > 0 else 'NEGATIVE_DIFFERENCE')
                if excludes_zero else 'INCONCLUSIVE_NOT_EQUIVALENCE')


def d17_decision(mean, hierarchical_ci, threshold=.03):
    """Full99 mean threshold AND strictly positive CI lower endpoint; no OR gate."""
    threshold_met = bool(mean >= threshold - 1e-12)
    interval_positive = hierarchical_ci.get('low') is not None and hierarchical_ci['low'] > 0.
    passed = bool(threshold_met and interval_positive)
    return dict(metric='mAP50_95', primary_checkpoint='mean_epochs_76_80', primary_subset='full99',
                E0_delta=float(mean), E0_delta_percentage_points=float(mean*100),
                threshold_absolute=float(threshold), threshold_met=threshold_met,
                hierarchical_ci=hierarchical_ci, ci_lower_strictly_positive=bool(interval_positive),
                final_pass=passed, decision_status='PASS' if passed else 'DO_NOT_PROMOTE',
                criterion='full99 stable mean delta >= 0.03 AND hierarchical paired CI95 lower > 0',
                ci_crossing_zero_means='INCONCLUSIVE_NOT_EQUIVALENCE',
                automatic_promotion_executed=False, automatic_search_executed=False)


def _basename(value):
    return str(value).replace('\\', '/').split('/')[-1]


def validate_contract(contract):
    names = {}
    for key, size in (('full_images', 99), ('kept_images', 89), ('excluded_images', 10)):
        values = contract.get(key)
        if (not isinstance(values, list) or len(values) != size or len(set(values)) != size
                or any(v != _basename(v) for v in values) or values != sorted(values)):
            raise ValueError(f'Subset contract {key} must contain {size} sorted unique basenames')
        names[key] = values
    if (set(names['kept_images']) & set(names['excluded_images'])
            or set(names['kept_images']) | set(names['excluded_images']) != set(names['full_images'])):
        raise ValueError('Subset contract must partition exactly the 99 validation images')
    if (contract.get('selection_uses_model_predictions') is not False
            or contract.get('selection_uses_performance') is not False):
        raise ValueError('Subset contract must be fixed independently of predictions and performance')
    return names


def validate_rows(rows):
    expected = {(b, h, s) for b, seeds in SEEDS.items() for h in HEADS for s in seeds}
    result = {}
    for row in rows:
        binding = row.get('binding', {})
        key = (binding.get('backbone'), binding.get('head'), binding.get('seed'))
        if key in result or key not in expected:
            raise ValueError('Exactly sixteen registered backbone/head/seed cells required; duplicate or unexpected seed')
        if row.get('primary_checkpoint') != 'mean_epochs_76_80':
            raise ValueError('Primary metric must use mean_epochs_76_80, not best or terminal checkpoint')
        if row.get('formal_eligible') is not True or row.get('epochs') != 80:
            raise ValueError('All sixteen formal runs must complete epoch 80')
        for field in ('stable_prediction_paths', 'stable_prediction_sha256'):
            if set(map(str, row.get(field, {}))) != set(map(str, EPOCHS)):
                raise ValueError(f'{field}: missing or extra stable epoch; require all epochs 76-80')
        history = row.get('history', [])
        if ([r.get('epoch') for r in history] != list(range(1, 81))
                or any(not np.isfinite(float(r.get('valid', {}).get('mAP50_95', float('nan'))))
                       for r in history)):
            raise ValueError('Full finite history epochs 1-80 required; no missing epoch allowed')
        result[key] = row
    if set(result) != expected:
        raise ValueError('Exactly sixteen registered cells required: E0 seeds42-46; YOLOv9_T seeds42-44')
    for field in ('environment_fingerprint', 'execution_manifest_sha256'):
        values = {r['binding'].get(field) for r in rows}
        if None in values or len(values) != 1:
            raise ValueError(f'Run identity differs or is missing: {field}')
    return result


def load_records(path, expected_images, expected_targets=None):
    records = read_json(path)
    if isinstance(records, dict):
        records = records['records']
    if not isinstance(records, list) or len(records) != 99:
        raise ValueError('Each stable epoch must contain 99 complete validation records')
    by_name = {}
    for record in records:
        name = _basename(record.get('image', record.get('path', '')))
        if not name or name in by_name:
            raise ValueError('Validation image name is empty or duplicated')
        for kind in ('target', 'prediction'):
            d = record.get(kind, {})
            boxes = np.asarray(d.get('boxes', []), dtype=np.float32).reshape(-1, 4)
            labels = np.asarray(d.get('labels', []))
            if (labels.ndim != 1 or len(boxes) != len(labels) or not np.isfinite(boxes).all()
                    or not np.isin(labels, [0, 1]).all() or np.any(boxes[:, 2:] < boxes[:, :2])):
                raise ValueError(f'Invalid {kind} boxes/labels: {name}')
            if kind == 'prediction':
                scores = np.asarray(d.get('scores', []), dtype=np.float32)
                if scores.ndim != 1 or len(scores) != len(boxes) or not np.isfinite(scores).all():
                    raise ValueError(f'Invalid prediction scores: {name}')
        if expected_targets is not None:
            target = expected_targets.get(name)
            if (target is None or record['target']['labels'] != target['labels']
                    or not np.array_equal(np.asarray(record['target']['boxes'], dtype=np.float32),
                                          np.asarray(target['boxes'], dtype=np.float32))):
                raise ValueError(f'Validation GT differs across seed/head/epoch: {name}')
        by_name[name] = dict(record, image=name)
    if set(by_name) != set(expected_images):
        raise ValueError('Validation image set differs from the frozen 99-image contract')
    return [by_name[name] for name in expected_images]


def _mean_sd(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(mean=float(values.mean()), sd=float(values.std(ddof=1)) if len(values)>1 else None,
                n_seeds=len(values), per_seed=values.tolist())


def _effect_report(point, replicate_values, seed_draws, seeds):
    hierarchical, image_only = reduce_paired_bootstrap(replicate_values, seed_draws)
    stable_point = np.asarray(point).mean(axis=1)
    result = {}
    for j, name in enumerate(METRICS):
        result[name] = dict(**_mean_sd(stable_point[:, j]), seeds=list(seeds),
            hierarchical_ci=interval(hierarchical[:, j]), image_only_ci=interval(image_only[:, j]))
    return result


def common_seed_contrasts(subset_values, seed_draws):
    """Paired 42-44 structure effects and difference-in-differences.

    Structure effects are YOLOv9_T minus E0 at each head. The marginal
    structure effect is the predeclared equal (1/2, 1/2) average of the two
    head-specific differences. E0's additional seeds45-46 never enter here.
    """
    raw = {}
    for head in HEADS:
        raw[head] = np.stack([subset_values['YOLOv9_T',head,seed]
                             - subset_values['E0',head,seed]
                             for seed in COMMON_SEEDS], axis=1)
    raw['equal_head_mean'] = (raw[HEADS[0]] + raw[HEADS[1]]) / 2.
    structure = {name:_effect_report(delta[0],delta[1:],seed_draws,COMMON_SEEDS)
                 for name,delta in raw.items()}
    interaction = raw[HEADS[1]] - raw[HEADS[0]]
    return dict(structure_effects=structure,
        structure_definition='YOLOv9_T minus E0; each head separately; equal_head_mean = 0.5 * (obj_softmax difference + quality_aware difference)',
        interaction=_effect_report(interaction[0],interaction[1:],seed_draws,COMMON_SEEDS))


def factorial_summary(rows, decision, *, subset_contract, output_dir=None):
    """Analyze all 16 complete runs from immutable five-epoch predictions.

    Rows come from train.run_result and must supply binding, history, epochs,
    formal_eligible, primary_checkpoint, stable_prediction_paths and their SHA256.
    ``subset_contract`` is the checked JSON mapping with full/kept/excluded names.
    """
    cells = validate_rows(rows)
    names = validate_contract(subset_contract)
    if (decision.get('metric') != 'mAP50_95' or decision.get('threshold_absolute') != .03
            or decision.get('primary_checkpoint', 'mean_epochs_76_80') != 'mean_epochs_76_80'):
        raise ValueError('Unregistered stable decision metric, epoch window, or D17 threshold')
    boot = decision.get('bootstrap', {})
    repetitions = int(boot.get('replicates', 2000))
    random_seed = int(boot.get('random_seed', 20261012))
    if repetitions < 1 or boot.get('confidence', .95) != .95:
        raise ValueError('Positive bootstrap replicates and 95% confidence required')
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    positions = {name:i for i,name in enumerate(names['full_images'])}
    populations = {'full99':np.arange(99),
                   'without_similar89':np.asarray([positions[n] for n in names['kept_images']])}
    plans = {}
    for k, (subset, population) in enumerate(populations.items()):
        e0_plan = bootstrap_plan(population, SEEDS['E0'], replicates=repetitions,
                                 random_seed=random_seed+100*k)
        # A distinct seed RNG per backbone; the image matrix is explicitly shared.
        yolo_plan = bootstrap_plan(population, COMMON_SEEDS, replicates=repetitions,
                                   random_seed=random_seed+100*k+1)
        plans[subset] = dict(image_draws=e0_plan['image_draws'],
                             E0_seed_draws=e0_plan['seed_draws'],
                             YOLOv9_T_seed_draws=yolo_plan['seed_draws'])
    values = {subset:{} for subset in populations}
    sources, parity = [], []
    reference_targets = None
    for key in sorted(cells):
        row = cells[key]
        run_values = {subset:[] for subset in populations}
        for epoch in EPOCHS:
            path = Path(row['stable_prediction_paths'][str(epoch)])
            source_hash = sha256(path)
            if source_hash != row['stable_prediction_sha256'][str(epoch)]:
                raise ValueError(f'Stable prediction source hash drift: {key} epoch {epoch}')
            records = load_records(path, names['full_images'], reference_targets)
            if reference_targets is None:
                reference_targets = {r['image']:r['target'] for r in records}
                class_counts = Counter(c for r in records for c in r['target']['labels'])
                if class_counts != {0:70, 1:29} or any(len(r['target']['labels']) != 1 for r in records):
                    raise ValueError('Frozen valid99 target contract requires one GT/image and class counts70/29')
            prepared = PreparedAP(records)
            oracle_samples = dict(full99=populations['full99'], without_similar89=populations['without_similar89'],
                duplicate_bootstrap=plans['full99']['image_draws'][0],
                one_class_repeated=np.repeat(0,99))
            checks = prepared.validate(oracle_samples)
            parity.append(dict(backbone=key[0], head=key[1], seed=key[2], epoch=epoch,
                               has_score_ties=any(c['has_score_ties'] for c in prepared.classes), checks=checks))
            for subset, population in populations.items():
                point = prepared.evaluate_one(population)
                if subset == 'full99':
                    logged = float(row['history'][epoch-1]['valid']['mAP50_95'])
                    if abs(point[0]-logged) > PARITY_TOLERANCE:
                        raise ValueError(f'Logged stable mAP does not match epoch predictions: {key}/{epoch}')
                samples = prepared.evaluate(plans[subset]['image_draws'])
                run_values[subset].append(np.vstack((point, samples)))
            sources.append(dict(backbone=key[0], head=key[1], seed=key[2], epoch=epoch,
                                path=str(path.resolve()), sha256=source_hash))
        for subset in populations:
            # [point + replicates, five epochs, metrics]
            values[subset][key] = np.stack(run_values[subset], axis=1)
        print(f'stable analysis {key}: epochs76-80 checked, both subsets bootstrapped', flush=True)
    reports = {}
    for subset, population in populations.items():
        subset_values = values[subset]
        per_epoch, per_seed, arms, effects = [], [], {}, {}
        for backbone, seeds in SEEDS.items():
            for head in HEADS:
                point = np.stack([subset_values[backbone,head,seed][0] for seed in seeds])
                stable = point.mean(axis=1)
                arms[f'{backbone}__{head}'] = {name:dict(**_mean_sd(stable[:,j]),seeds=list(seeds))
                                                   for j,name in enumerate(METRICS)}
                for i, seed in enumerate(seeds):
                    per_seed.append(dict(backbone=backbone,head=head,seed=seed,
                                         **dict(zip(METRICS,stable[i].tolist()))))
                    for q, epoch in enumerate(EPOCHS):
                        per_epoch.append(dict(backbone=backbone,head=head,seed=seed,epoch=epoch,
                                              **dict(zip(METRICS,point[i,q].tolist()))))
            delta = np.stack([subset_values[backbone,HEADS[1],seed]-subset_values[backbone,HEADS[0],seed]
                              for seed in seeds],axis=1)
            effects[backbone] = _effect_report(delta[0],delta[1:],plans[subset][f'{backbone}_seed_draws'],seeds)
        common = common_seed_contrasts(subset_values,plans[subset]['YOLOv9_T_seed_draws'])
        reports[subset] = dict(n_images=len(population),images=[names['full_images'][i] for i in population],
            per_epoch=per_epoch,per_seed=per_seed,arms=arms,head_effects=effects,
            structure_effects_common_seeds_42_44=common['structure_effects'],
            structure_effect_definition=common['structure_definition'],
            interaction_common_seeds_42_44=common['interaction'])
    full = reports['full99']
    primary = full['head_effects']['E0']['mAP50_95']
    verdict = d17_decision(primary['mean'],primary['hierarchical_ci'])
    direction = {}
    for backbone in SEEDS:
        a = full['head_effects'][backbone]['mAP50_95']['mean']
        b = reports['without_similar89']['head_effects'][backbone]['mAP50_95']['mean']
        direction[backbone] = dict(full99_delta=a,without_similar89_delta=b,
                                   same_direction=bool(np.sign(a)==np.sign(b)),
                                   used_in_D17_gate=False)
    verdict['direction_consistency_independent'] = direction
    report = dict(schema_version=1,status='COMPLETE',metric='all-area COCO-style AP101 mAP50-95',
        primary_checkpoint='mean_epochs_76_80',epochs=list(EPOCHS),n_runs=16,
        seeds_by_backbone={k:list(v) for k,v in SEEDS.items()},
        test_evaluated=False,training_or_inference_performed=False,
        bootstrap=dict(method='hierarchical_paired_seed_and_image_percentile',replicates=repetitions,
            random_seed=random_seed,confidence=.95,quantile_method='linear',
            image_sampling='One n-image draw shared across every seed, head, and all five epochs per replicate',
            seed_sampling='Paired seed IDs sampled with replacement independently per backbone; interaction shares common42-44 IDs',
            epoch_reduction='Recompute AP per sampled epoch, mean5 epochs per seed, then mean sampled seeds',
            image_only='Same image draws; fixed observed training seeds; auxiliary interval',
            limitation='Validation-development evidence; five/three observed seeds give limited training-population coverage'),
        subsets=reports,direction_consistency=direction,decision=verdict,
        arms=full['arms'],paired_head_effect_YOLOv9_T=full['head_effects']['YOLOv9_T'],
        structure_effects_common_seeds_42_44=full['structure_effects_common_seeds_42_44'],
        interaction_YOLO_head_effect_minus_E0_head_effect=full['interaction_common_seeds_42_44'],
        evaluator_parity=dict(tolerance=PARITY_TOLERANCE,checks=parity,all_passed=True),
        source_records=sources,subset_contract_canonical_sha256=hashlib.sha256(
            json.dumps(subset_contract,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        stable_analysis_sha256=sha256(__file__))
    if any(sha256(item['path']) != item['sha256'] for item in sources):
        raise ValueError('Prediction source changed during stable analysis')
    if output_dir is not None:
        for subset, plan in plans.items():
            np.savez_compressed(output_dir/f'{subset}_bootstrap_draws.npz',**plan)
            keys = sorted(values[subset])
            np.savez_compressed(output_dir/f'{subset}_epoch_bootstrap_values.npz',
                values=np.stack([values[subset][key] for key in keys]),
                keys=np.asarray(['/'.join(map(str,key)) for key in keys]),metrics=np.asarray(METRICS),epochs=np.asarray(EPOCHS))
        write_json(output_dir/'stable_analysis.json',report)
        write_json(output_dir/'evaluator_parity.json',report['evaluator_parity'])
    return _json_safe(report)
