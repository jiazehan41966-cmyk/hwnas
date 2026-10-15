"""Frozen recipe-selection and paired-AP tests; synthetic data are not evidence."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sonar_recipe_check import analysis
from tools.sonar_head_factorial import stable_analysis as stable


def selection(means, lows=None):
    arms = dict(zip(analysis.RECIPES, means))
    lows = [.001, .001, .001] if lows is None else lows
    effects = {r: dict(mean=arms[r] - arms['R0'], hierarchical_ci=dict(low=low, high=.2))
               for r, low in zip(analysis.CANDIDATES, lows)}
    return analysis.select_recipe(arms, effects)


def synthetic_rows():
    rows = []
    for recipe in analysis.RECIPES:
        for seed in analysis.SEEDS:
            rows.append(dict(recipe=recipe, binding=dict(backbone='YOLOv9_T', head='quality_aware',
                seed=seed, mode='formal', test_evaluated=False, environment_fingerprint='env',
                subset_contract_sha256='subset', validation_target_sha256='gt',
                execution_manifest_sha256='C01' if recipe == 'R0' else 'new'),
                epochs=80, formal_eligible=True, test_evaluated=False,
                primary_checkpoint='mean_epochs_76_80',
                stable_prediction_paths={str(e): f'{recipe}/{seed}/{e}.json' for e in analysis.EPOCHS},
                stable_prediction_sha256={str(e): 'unused' for e in analysis.EPOCHS},
                history=[dict(epoch=e, valid=dict(mAP50_95=.2)) for e in range(1, 81)],
                metrics=dict(mAP50_95=.2), stable_metrics=dict(mAP50_95=.2),
                initial_state_sha256=f'initial-{seed}', initialization=dict(seed=seed, head=f'h-{seed}'),
                model_info=dict(backbone='YOLOv9_T', head='quality_aware', parameters=dict(head=19),
                                initialization=dict(seed=seed))))
    return rows


def contract():
    names = [f'image_{i:03d}.jpg' for i in range(99)]
    return dict(full_images=names, kept_images=names[10:], excluded_images=names[:10],
                selection_uses_model_predictions=False, selection_uses_performance=False)


def example_records(recipe=0, seed=0, epoch=0):
    records = []
    for i in range(4):
        label = i % 2
        # Tied TP/FP scores, missing detections, and repeated image sampling.
        hit = (i + recipe + seed + epoch) % 3 != 0
        boxes = [[20., 20., 30., 30.]] + ([[0., 0., 10., 10.]] if hit else [])
        records.append(dict(image=f'image_{i}.jpg', target=dict(boxes=[[0., 0., 10., 10.]], labels=[label]),
            prediction=dict(boxes=boxes, scores=[.8] * len(boxes), labels=[label] * len(boxes))))
    return records


class DecisionTests(unittest.TestCase):
    def test_requires_inclusive_one_point_and_strict_positive_lower(self):
        self.assertEqual(selection([.1, .11, .1, .1])['selected_recipe'], 'R1')
        self.assertEqual(selection([.1, .109999, .1, .1])['selected_recipe'], 'R0')
        self.assertEqual(selection([.1, .14, .1, .1], [0., .001, .001])['selected_recipe'], 'R0')
        self.assertEqual(selection([.1, .14, .1, .1], [-.001, .001, .001])['selected_recipe'], 'R0')

    def test_simplest_rule_inclusive_half_point_boundary(self):
        self.assertEqual(selection([.1, .115, .12, .116])['selected_recipe'], 'R1')
        self.assertEqual(selection([.1, .114999, .12, .121])['selected_recipe'], 'R2')
        self.assertEqual(selection([.1, .115, .12, .126])['selected_recipe'], 'R3')

    def test_inconclusive_best_does_not_control_effective_recipe_choice(self):
        result = selection([.1, .112, .113, .17], [.001, .001, -.01])
        self.assertEqual(result['best_effective_recipe'], 'R2')
        self.assertEqual(result['selected_recipe'], 'R1')
        self.assertFalse(result['multiple_comparisons_adjusted'])
        self.assertFalse(result['auxiliary89_used_for_selection'])

    def test_undefined_ci_and_all_noneffective_retain_R0(self):
        result = selection([.1, .12, .13, .14], [None, 0., -.01])
        self.assertEqual(result['selected_recipe'], 'R0')
        self.assertEqual(result['effective_candidates'], [])

    def test_mismatched_mean_and_unregistered_bootstrap_rejected(self):
        with self.assertRaisesRegex(ValueError, 'paired stable mean'):
            analysis.select_recipe(dict(R0=.1, R1=.2, R2=.2, R3=.2),
                {r: dict(mean=.03, hierarchical_ci=dict(low=.001, high=.2)) for r in analysis.CANDIDATES})
        analysis.validate_decision(deepcopy(analysis.DECISION))
        for field, value in (('replicates', 1999), ('random_seed', 42), ('confidence', .9)):
            config = deepcopy(analysis.DECISION)
            config['bootstrap'][field] = value
            with self.assertRaisesRegex(ValueError, 'bootstrap'):
                analysis.validate_decision(config)


class ContractTests(unittest.TestCase):
    def test_full_grid_and_wrapped_C01_rows(self):
        rows = synthetic_rows()
        self.assertEqual(len(analysis.validate_rows(rows)), 12)
        groups = {recipe: [] for recipe in analysis.RECIPES}
        for row in rows:
            groups[row.pop('recipe')].append(row)
        self.assertEqual(len(analysis.validate_rows(groups)), 12)
        for rows in (synthetic_rows()[:-1], synthetic_rows() + [synthetic_rows()[0]]):
            with self.assertRaisesRegex(ValueError, 'twelve'):
                analysis.validate_rows(rows)

    def test_history_missing_predictions_and_stable_mean_fail_closed(self):
        for field in ('stable_prediction_paths', 'stable_prediction_sha256'):
            rows = synthetic_rows()
            del rows[0][field]['78']
            with self.assertRaisesRegex(ValueError, 'epochs76'):
                analysis.validate_rows(rows)
        rows = synthetic_rows()
        rows[0]['history'].pop(15)
        with self.assertRaisesRegex(ValueError, 'history'):
            analysis.validate_rows(rows)
        rows = synthetic_rows()
        rows[0]['stable_metrics']['mAP50_95'] = .3
        with self.assertRaisesRegex(ValueError, 'Stable metric'):
            analysis.validate_rows(rows)
        rows[0]['stable_metrics']['mAP50_95'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Stable metric'):
            analysis.validate_rows(rows)

    def test_initialization_structure_and_environment_drift_rejected(self):
        for field in ('initialization', 'initial_state_sha256'):
            rows = synthetic_rows()
            rows[3][field] = 'changed'
            with self.assertRaisesRegex(ValueError, 'initialization differs'):
                analysis.validate_rows(rows)
        rows = synthetic_rows()
        rows[3]['model_info']['parameters']['head'] = 20
        with self.assertRaisesRegex(ValueError, 'structure'):
            analysis.validate_rows(rows)
        rows = synthetic_rows()
        rows[3]['binding']['environment_fingerprint'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'environment'):
            analysis.validate_rows(rows)

    def test_only_explicit_location_fields_normalized(self):
        rows = synthetic_rows()
        for i, row in enumerate(rows):
            row['model_info']['source_root'] = f'/root/checkout{i}'
        self.assertEqual(len(analysis.validate_rows(rows)), 12)
        rows[3]['model_info']['arbitrary_note'] = 'cannot silently ignore'
        with self.assertRaisesRegex(ValueError, 'structure'):
            analysis.validate_rows(rows)

    def test_prediction_hash_drift_before_any_statistics(self):
        rows = synthetic_rows()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'records.json'
            path.write_text('[]', encoding='utf-8')
            rows[0]['stable_prediction_paths']['76'] = str(path)
            with self.assertRaisesRegex(ValueError, 'source hash drift'):
                analysis.recipe_summary(rows, subset_contract=contract())

    def test_gt_source_hash_rejected_even_when_all_rows_agree(self):
        rows = synthetic_rows()
        records = [dict(image=name, target=dict(boxes=[[0., 0., 10., 10.]], labels=[0 if i < 70 else 1]),
                   prediction=dict(boxes=[], scores=[], labels=[]))
                   for i, name in enumerate(contract()['full_images'])]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'records.json'
            stable.write_json(path, records)
            rows[0]['stable_prediction_paths']['76'] = str(path)
            rows[0]['stable_prediction_sha256']['76'] = stable.sha256(path)
            with self.assertRaisesRegex(ValueError, 'frozen source-label'):
                analysis.recipe_summary(rows, subset_contract=contract())


class PairedReplayTests(unittest.TestCase):
    def test_fixed_shared_draws_are_reproducible_and_seed_resampling_is_real(self):
        plan = stable.bootstrap_plan(range(99), analysis.SEEDS, replicates=analysis.REPLICATES,
                                     random_seed=analysis.RANDOM_SEED)
        again = stable.bootstrap_plan(range(99), analysis.SEEDS, replicates=analysis.REPLICATES,
                                      random_seed=analysis.RANDOM_SEED)
        np.testing.assert_array_equal(plan['image_draws'], again['image_draws'])
        np.testing.assert_array_equal(plan['seed_draws'], again['seed_draws'])
        self.assertTrue(any(len(set(row)) < 3 for row in plan['seed_draws']))
        self.assertTrue(all(len(set(row)) < 99 for row in plan['image_draws']))

    def test_repeated_images_exact_AP_and_paired_epoch_seed_oracle(self):
        # The same repeated-image matrix and seed draws are reused by all cells.
        draws = np.array([[3, 0, 0, 1], [2, 2, 1, 3], [1, 1, 1, 0]])
        seed_draws = np.array([[2, 2, 0], [1, 0, 1], [0, 0, 2]])
        values = {}
        oracle = {}
        for r, recipe in enumerate(analysis.RECIPES):
            for s, seed in enumerate(analysis.SEEDS):
                collected, exact = [], []
                for epoch in range(5):
                    records = example_records(r, s, epoch)
                    prepared = stable.PreparedAP(records)
                    sample_indices = np.vstack((np.arange(4), draws))
                    fast = prepared.evaluate(sample_indices)
                    slow = np.stack([stable.reference_metrics(records, indices) for indices in sample_indices])
                    np.testing.assert_allclose(fast, slow, atol=stable.PARITY_TOLERANCE, rtol=0, equal_nan=True)
                    collected.append(fast)
                    exact.append(slow)
                values[recipe, seed] = np.stack(collected, axis=1)
                oracle[recipe, seed] = np.stack(exact, axis=1)
        report = analysis.summarize_values(values, seed_draws)
        for recipe in analysis.CANDIDATES:
            effects = report['candidate_effects_vs_R0'][recipe]
            for j, name in enumerate(analysis.METRICS):
                expected_point = np.mean([np.mean(oracle[recipe, s][0, :, j] - oracle['R0', s][0, :, j])
                                          for s in analysis.SEEDS])
                expected_samples = []
                for b, sampled_seeds in enumerate(seed_draws):
                    expected_samples.append(np.mean([np.mean(oracle[recipe, analysis.SEEDS[s]][b + 1, :, j]
                        - oracle['R0', analysis.SEEDS[s]][b + 1, :, j]) for s in sampled_seeds]))
                self.assertAlmostEqual(effects[name]['mean'], expected_point, places=6)
                exact_ci = stable.interval(expected_samples)
                self.assertAlmostEqual(effects[name]['hierarchical_ci']['low'], exact_ci['low'], places=6)
                self.assertAlmostEqual(effects[name]['hierarchical_ci']['high'], exact_ci['high'], places=6)

    def test_auxiliary_report_cannot_override_primary_selection(self):
        # Deliberate seed heterogeneity also distinguishes the hierarchical CI.
        values = {}
        draws = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        for recipe in analysis.RECIPES:
            for j, seed in enumerate(analysis.SEEDS):
                delta = 0 if recipe == 'R0' else [.0, .03, .09][j]
                values[recipe, seed] = np.full((4, 5, len(analysis.METRICS)), .1 + delta)
        report = analysis.summarize_values(values, draws)
        effect = report['candidate_effects_vs_R0']['R1']['mAP50_95']
        self.assertGreater(effect['hierarchical_ci']['high'] - effect['hierarchical_ci']['low'], .05)
        self.assertAlmostEqual(effect['image_only_ci']['high'] - effect['image_only_ci']['low'], 0.)


class DecisionArtifactTests(unittest.TestCase):
    def test_D14_formally_records_R0_when_no_candidate_passes(self):
        report = dict(status='COMPLETE', n_runs=12, decision=selection([.1, .103, .109, .101]),
                      analysis_sha256='source', subset_contract_canonical_sha256='subset')
        with tempfile.TemporaryDirectory() as tmp:
            written = analysis.write_d14(report, tmp)
            self.assertEqual(written['selected_recipe'], 'R0')
            self.assertEqual(written['decision_id'], 'D14')
            archived = json.loads((Path(tmp) / 'D14.json').read_text(encoding='utf-8'))
            self.assertEqual(archived, written)
            markdown = (Path(tmp) / 'D14.md').read_text(encoding='utf-8')
            self.assertIn('**R0**', markdown)
            self.assertIn('未做多重比较校正', markdown)
            self.assertIn('89 图子集', markdown)

    def test_D14_cannot_be_written_from_partial_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'twelve complete'):
                analysis.write_d14(dict(status='PARTIAL', n_runs=11), tmp)
            self.assertFalse((Path(tmp) / 'D14.json').exists())


if __name__ == '__main__':
    stable.torch.set_num_threads(2)
    unittest.main(verbosity=2)
