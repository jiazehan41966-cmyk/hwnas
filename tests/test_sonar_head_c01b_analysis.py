"""C01b gate, provenance, and paired reduction tests; no experiment evidence."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sonar_head_c01b import analysis


def subset_contract():
    names = [f'image_{i:03d}.jpg' for i in range(99)]
    return dict(full_images=names, kept_images=names[10:], excluded_images=names[:10],
                selection_uses_model_predictions=False, selection_uses_performance=False)


def decision():
    return dict(primary_checkpoint='mean_epochs_76_80', primary_subset='full99',
                direction='C01b_minus_C01', ci_used_in_gate=False,
                minimum_delta_absolute={'mAP50_95': -.005, 'AP75': -.005},
                bootstrap=dict(replicates=2000, random_seed=20261013, confidence=.95))


def rows():
    result = {variant: [] for variant in analysis.VARIANTS}
    for variant in analysis.VARIANTS:
        for seed in analysis.SEEDS:
            result[variant].append(dict(
                binding=dict(backbone='E0', head='quality_aware', seed=seed,
                    environment_fingerprint='shared-environment',
                    execution_manifest_sha256=f'manifest-{variant}',
                    validation_target_sha256='same-gt', subset_contract_sha256='same-subset'),
                comparison_contract_sha256='same-comparison',
                epochs=80, formal_eligible=True, test_evaluated=False,
                primary_checkpoint='mean_epochs_76_80', metrics=dict(mAP50_95=.3, AP75=.32),
                model_info=dict(head_contract=dict(raw_output_channels_per_scale=
                    [66] * 4 if variant == 'C01' else [6] * 4)),
                stable_prediction_paths={str(e): f'{variant}_{seed}_{e}.json' for e in analysis.EPOCHS},
                stable_prediction_sha256={str(e): 'pending' for e in analysis.EPOCHS},
                history=[dict(epoch=e, valid=dict(mAP50_95=.3, AP75=.32)) for e in range(1, 81)]))
    return result


class GateTests(unittest.TestCase):
    def test_both_metrics_and_inclusive_boundary(self):
        chosen = analysis.channel_decision(dict(mAP50_95=-.005, AP75=-.005), channels_C01=66)
        self.assertTrue(chosen['gate_pass'])
        self.assertEqual(chosen['selected_raw_output_channels_per_scale'], [6] * 4)
        # An arithmetic representation of exactly 0.5 percentage point is inclusive.
        self.assertTrue(analysis.channel_decision(
            dict(mAP50_95=.195 - .2, AP75=0.), channels_C01=66)['gate_pass'])
        for values in (dict(mAP50_95=-.005001, AP75=.2),
                       dict(mAP50_95=.2, AP75=-.005001)):
            chosen = analysis.channel_decision(values, channels_C01=66)
            self.assertFalse(chosen['gate_pass'])
            self.assertEqual(chosen['selected_raw_output_channels'], 66)
            self.assertEqual(chosen['selected_variant'], 'C01')

    def test_channel_count_comes_from_actual_contract(self):
        values = dict(mAP50_95=-.01, AP75=.1)
        self.assertEqual(analysis.channel_decision(values, channels_C01=34)
                         ['selected_raw_output_channels'], 34)
        supplied = dict(zip(analysis.SCALES, [66] * 4))
        self.assertEqual(analysis.channel_decision(values, channels_C01=supplied)
                         ['selected_raw_output_channels'], 66)
        for bad in (True, 0, [66] * 3, [66, 66, 34, 66], {'P2': 66}):
            with self.assertRaises(ValueError):
                analysis.channel_decision(values, channels_C01=bad)

    def test_bootstrap_interval_is_not_an_extra_gate_or_noninferiority_claim(self):
        chosen = analysis.channel_decision(dict(mAP50_95=-.004, AP75=-.004), channels_C01=66)
        self.assertTrue(chosen['gate_pass'])
        self.assertFalse(chosen['ci_used_in_gate'])
        self.assertFalse(chosen['statistical_noninferiority_established'])
        self.assertNotIn('hierarchical_ci', chosen['conditions']['mAP50_95'])

    def test_nonfinite_delta_cannot_choose_channels(self):
        for bad in (float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'Finite paired stable delta'):
                analysis.channel_decision(dict(mAP50_95=bad, AP75=.1), channels_C01=66)


class ContractTests(unittest.TestCase):
    def test_two_variant_manifests_allowed_shared_environment_required(self):
        cells, common, channels = analysis.validate_rows(rows(), channels_C01=66)
        self.assertEqual(len(cells), 6)
        self.assertEqual(len(channels), 6)
        self.assertEqual(common['comparison_contract_sha256'], 'same-comparison')
        for field, in_binding in (('environment_fingerprint', True),
                                  ('comparison_contract_sha256', False),
                                  ('validation_target_sha256', True)):
            bad = rows()
            target = bad['C01b'][0]['binding'] if in_binding else bad['C01b'][0]
            target[field] = 'different'
            with self.assertRaisesRegex(ValueError, field):
                analysis.validate_rows(bad, channels_C01=66)
        bad = rows(); bad['C01b'][0]['binding']['execution_manifest_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'execution_manifest'):
            analysis.validate_rows(bad, channels_C01=66)

    def test_exact_six_cells_and_complete_five_epoch_archives(self):
        cases = []
        bad = rows(); bad['C01'].pop(); cases.append(bad)
        bad = rows(); bad['C01'][2]['binding']['seed'] = 42; cases.append(bad)
        bad = rows(); bad['C01b'][2]['binding']['seed'] = 45; cases.append(bad)
        bad = rows(); del bad['C01b'][0]['stable_prediction_paths']['78']; cases.append(bad)
        bad = rows(); bad['C01b'][0]['history'].pop(77); cases.append(bad)
        bad = rows(); bad['C01b'][0]['formal_eligible'] = False; cases.append(bad)
        for bad in cases:
            with self.assertRaises(ValueError):
                analysis.validate_rows(bad, channels_C01=66)

    def test_AP75_history_metrics_and_measured_channels_are_checked(self):
        bad = rows(); del bad['C01'][0]['history'][77]['valid']['AP75']
        with self.assertRaisesRegex(ValueError, 'AP75'):
            analysis.validate_rows(bad, channels_C01=66)
        bad = rows(); bad['C01'][0]['metrics']['AP75'] = .8
        with self.assertRaisesRegex(ValueError, 'stable metrics'):
            analysis.validate_rows(bad, channels_C01=66)
        bad = rows(); bad['C01b'][0]['model_info']['head_contract']['raw_output_channels_per_scale'] = [66] * 4
        with self.assertRaisesRegex(ValueError, 'output channels'):
            analysis.validate_rows(bad, channels_C01=66)

    def test_immutable_registered_gate_and_bootstrap(self):
        self.assertEqual(analysis.validate_decision(decision())['bootstrap']['replicates'], 2000)
        for name, bad_value in (('primary_subset', 'without_similar89'),
                                ('primary_checkpoint', 'best'), ('ci_used_in_gate', True)):
            bad = decision(); bad[name] = bad_value
            with self.assertRaises(ValueError):
                analysis.validate_decision(bad)
        bad = decision(); bad['minimum_delta_absolute']['AP75'] = -.01
        with self.assertRaises(ValueError):
            analysis.validate_decision(bad)
        bad = decision(); bad['bootstrap']['random_seed'] = 123
        with self.assertRaises(ValueError):
            analysis.validate_decision(bad)

    def test_paired_training_witness_cannot_drift(self):
        supplied = rows()
        for variant in analysis.VARIANTS:
            supplied[variant][0]['history'][0]['image_order_sha256'] = 'same-order'
        cells, _, _ = analysis.validate_rows(supplied, channels_C01=66)
        self.assertEqual(analysis._pairing_witnesses(cells)['n_checks'], 1)
        cells['C01b', 42]['history'][0]['image_order_sha256'] = 'different-order'
        with self.assertRaisesRegex(ValueError, 'witness differs'):
            analysis._pairing_witnesses(cells)


class PairedReplayTests(unittest.TestCase):
    def make_archives(self, directory):
        supplied = rows()
        for variant in analysis.VARIANTS:
            for row in supplied[variant]:
                seed = row['binding']['seed']
                for epoch in analysis.EPOCHS:
                    base = .3 + (epoch - 76) * .001 + (seed - 42) * .01
                    change = 0. if variant == 'C01' else -.003 - (seed - 42) * .001
                    marker = base + change
                    records = [dict(image=name,
                        target=dict(boxes=[[0., 0., 10., 10.]], labels=[0 if i < 70 else 1]),
                        prediction=dict(boxes=[], scores=[], labels=[]),
                        synthetic_test_metric=marker)
                        for i, name in enumerate(subset_contract()['full_images'])]
                    path = directory / f'{variant}_{seed}_{epoch}.json'
                    path.write_text(json.dumps(records), encoding='utf-8')
                    row['stable_prediction_paths'][str(epoch)] = str(path)
                    row['stable_prediction_sha256'][str(epoch)] = analysis.stable.sha256(path)
                    row['history'][epoch-1]['valid'] = dict(mAP50_95=marker + .00049, AP75=marker + .02049)
                row['metrics'] = {name: np.mean([row['history'][e-1]['valid'][name]
                                  for e in analysis.EPOCHS]) for name in analysis.PRIMARY_METRICS}
        return supplied

    def fake_prepared(self, instances):
        class FakePrepared:
            def __init__(self, records):
                self.records = records
                self.base = records[0]['synthetic_test_metric']
                self.classes = [{'has_score_ties': False}]
                self.draws = []
                instances.append(self)

            def evaluate_one(self, indices):
                value = np.full(len(analysis.METRICS), self.base + np.mean(indices) * .00001)
                value[analysis.METRICS.index('AP75')] += .02
                return value

            def evaluate(self, draws):
                self.draws.append(draws)
                value = np.full((len(draws), len(analysis.METRICS)), self.base)
                value += np.mean(draws, axis=1)[:, None] * .00001
                value[:, analysis.METRICS.index('AP75')] += .02
                return value

            def validate(self, samples):
                return [dict(sample=name, passed=True, max_absolute_error=0.) for name in samples]
        return FakePrepared

    def test_separate_epoch_AP_mean_shared_draws_paired_seed_CI_and_saved_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            supplied = self.make_archives(directory)
            instances = []
            with patch.object(analysis.stable, 'PreparedAP', self.fake_prepared(instances)):
                result = analysis.c01b_summary(supplied, decision(), subset_contract=subset_contract(),
                                               channels_C01=66, output_dir=directory / 'audit')
            self.assertEqual(len(instances), 30)  # One preparation per run/epoch; no pooling.
            self.assertTrue(all(len(p.records) == 99 for p in instances))
            self.assertTrue(all(p.draws[0] is instances[0].draws[0] for p in instances))
            self.assertTrue(all(p.draws[1] is instances[0].draws[1] for p in instances))
            self.assertEqual(instances[0].draws[0].shape, (2000, 99))
            self.assertEqual(instances[0].draws[1].shape, (2000, 89))
            self.assertTrue(result['decision']['gate_pass'])
            delta = result['paired_difference_C01b_minus_C01']['mAP50_95']
            np.testing.assert_allclose(delta['per_seed'], [-.003, -.004, -.005], atol=1e-12)
            self.assertAlmostEqual(delta['mean'], -.004)
            self.assertGreater(delta['hierarchical_ci']['high'] - delta['hierarchical_ci']['low'], .0005)
            self.assertAlmostEqual(delta['image_only_ci']['high'] - delta['image_only_ci']['low'], 0., places=12)
            self.assertEqual(len(result['source_records']), 30)
            self.assertEqual(len(result['evaluator_parity']['checks']), 30)
            self.assertTrue(all(len(item['checks']) == 4 for item in result['evaluator_parity']['checks']))
            with np.load(directory / 'audit' / 'full99_epoch_bootstrap_values.npz') as saved:
                self.assertEqual(saved['values'].shape, (6, 2001, 5, len(analysis.METRICS)))
            archived = json.loads((directory / 'audit' / 'c01b_analysis.json').read_text(encoding='utf-8'))
            self.assertEqual(archived['decision']['selected_raw_output_channels'], 6)

    def test_prediction_hash_drift_and_AP75_log_disagreement_fail_before_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            supplied = self.make_archives(Path(tmp))
            bad = deepcopy(supplied); bad['C01'][0]['stable_prediction_sha256']['76'] = 'drift'
            with self.assertRaisesRegex(ValueError, 'source hash drift'):
                analysis.c01b_summary(bad, decision(), subset_contract=subset_contract(), channels_C01=66)
            supplied['C01'][0]['history'][75]['valid']['AP75'] += .01
            supplied['C01'][0]['metrics']['AP75'] += .002
            with patch.object(analysis.stable, 'PreparedAP', self.fake_prepared([])):
                with self.assertRaisesRegex(ValueError, 'Logged stable AP75'):
                    analysis.c01b_summary(supplied, decision(), subset_contract=subset_contract(), channels_C01=66)


if __name__ == '__main__':
    unittest.main(verbosity=2)
