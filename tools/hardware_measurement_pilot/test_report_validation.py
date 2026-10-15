"""Import checks against the immutable first real board timing snapshot."""
from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

from tools.hardware_measurement_pilot.measurement import summarize_csv
from tools.hardware_measurement_pilot.report import compare_replayed_summary, validate_timing_snapshot, sha


BASE = Path(__file__).resolve().parents[2] / 'results/hardware_measurement_pilot_20261012/attempt05'
SNAPSHOT = BASE / 'board_snapshots/body8_high_timing01'
PACKAGE = BASE / 'board_packages/body8_to_head16_reference_high/20261012T141130951567Z/case'


@unittest.skipUnless((SNAPSHOT / 'evidence/output/summary.json').is_file(), 'real board snapshot not present')
class RealTimingReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        output = SNAPSHOT / 'evidence/output'
        cls.stored = json.loads((output / 'summary.json').read_text())
        cls.replay = summarize_csv(output / 'timing_raw.csv', json.loads((output / 'timing_manifest.json').read_text()))
        cls.replay.update(correctness_fixtures_passed=15, integrated_phase_validation='PASS_BOUND_ARTIFACT',
            board_bundle_sha256=sha(PACKAGE / 'board_bundle.json'), protocol_status='DRAFT', formal_cost_eligible=False)

    def test_real_snapshot_identity_fixtures_and_statistics(self):
        result = validate_timing_snapshot(SNAPSHOT, PACKAGE)
        self.assertEqual(result['status'], 'PASS_IMPORT_AUDIT')
        self.assertEqual(result['numerical_status'], 'PASS')
        self.assertIs(result['hardware_measurement_eligible'], False)
        self.assertEqual(result['cpu_governor_observations']['batch1']['observed_policy_values'], ['userspace'])
        self.assertTrue(all(row['ulp_distance'] <= 8 for row in result['replay_comparison']['tolerated_float_differences']))

    def test_json_key_serialization_is_normalized(self):
        self.assertIn(1, self.replay['cases'][0]['host_group_size_distribution'])
        self.assertIn('1', self.stored['cases'][0]['host_group_size_distribution'])
        self.assertEqual(compare_replayed_summary(self.stored, self.replay)['status'], 'PASS_REPLAY_EQUIVALENCE')

    def test_count_identity_and_pass_gate_mutations_rejected(self):
        for mutation in ('count', 'identity', 'raw_hash', 'case_status', 'overall_status', 'eligibility'):
            value = deepcopy(self.stored)
            if mutation == 'count': value['cases'][0]['measured_rows'] += 1
            elif mutation == 'identity': value['cases'][0]['case_id'] += '_wrong'
            elif mutation == 'raw_hash': value['raw_csv_sha256'] = '0' * 64
            elif mutation == 'case_status': value['cases'][0]['status'] = 'FAIL'
            elif mutation == 'overall_status': value['status'] = 'FAIL'
            else: value['hardware_measurement_eligible'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compare_replayed_summary(value, self.replay)

    def test_obvious_statistic_mutation_rejected(self):
        value = deepcopy(self.stored)
        value['cases'][0]['scopes']['transfer']['sample_std_ns'] += 1.0
        with self.assertRaises(ValueError):
            compare_replayed_summary(value, self.replay)

    def test_eight_ulp_boundary_not_nine(self):
        canonical = json.loads(json.dumps(self.replay))
        original = canonical['cases'][0]['scopes']['transfer']['sample_std_ns']
        changed = original
        for _ in range(8): changed = math.nextafter(changed, math.inf)
        canonical['cases'][0]['scopes']['transfer']['sample_std_ns'] = changed
        result = compare_replayed_summary(canonical, self.replay)
        self.assertEqual(result['tolerated_float_differences'][0]['ulp_distance'], 8)
        canonical['cases'][0]['scopes']['transfer']['sample_std_ns'] = math.nextafter(changed, math.inf)
        with self.assertRaises(ValueError):
            compare_replayed_summary(canonical, self.replay)

    def test_clock_and_threshold_float_are_exact(self):
        for field in ('clock', 'threshold'):
            value = deepcopy(self.stored)
            if field == 'clock':
                meta = value['cases'][0]['batches'][0]['metadata']
                meta['clock_hz'] = math.nextafter(meta['clock_hz'], math.inf)
            else:
                value['cv_limit'] = math.nextafter(value['cv_limit'], math.inf)
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare_replayed_summary(value, self.replay)

    def test_schema_type_and_nonfinite_are_rejected(self):
        for mutation in ('keys', 'boolean_count', 'nonfinite'):
            value = deepcopy(self.stored)
            if mutation == 'keys': value['cases'][0]['extra'] = 1
            elif mutation == 'boolean_count': value['cases'][0]['scopes']['transfer']['n'] = True
            else: value['cases'][0]['scopes']['transfer']['sample_std_ns'] = float('nan')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compare_replayed_summary(value, self.replay)

    def test_metadata_cannot_impersonate_statistical_path(self):
        for nested in ('scopes', 'host_scopes_by_group_size', 'host_median_stability'):
            expected = json.loads(json.dumps(self.replay))
            metadata = expected['cases'][0]['batches'][0]['metadata']
            metadata[nested] = {'transfer': {'cv': 0.1}} if nested != 'host_median_stability' else {'maximum_relative_deviation': 0.1}
            actual = deepcopy(expected)
            target = actual['cases'][0]['batches'][0]['metadata'][nested]
            if nested == 'host_median_stability': target['maximum_relative_deviation'] = math.nextafter(0.1, math.inf)
            else: target['transfer']['cv'] = math.nextafter(0.1, math.inf)
            with self.subTest(nested=nested), self.assertRaises(ValueError):
                compare_replayed_summary(actual, expected)


if __name__ == '__main__':
    unittest.main()
