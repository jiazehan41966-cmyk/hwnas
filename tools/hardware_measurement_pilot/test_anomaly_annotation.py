import json
import unittest

from tools.hardware_measurement_pilot.anomaly_annotation import annotate_rows, annotate_csv, CSV_FIELDS


def group_rows(values, sizes=None, *, phase='measured', batch='b1', compute=None):
    sizes = sizes or [1] * len(values)
    result = []
    for group, (value, size) in enumerate(zip(values, sizes)):
        for _ in range(size):
            item = dict(case_id='case', iteration=str(len(result)), phase=phase, case_sha256='1'*64,
                numeric_contract_sha256='2'*64, bitstream_sha256='3'*64, expected_output_sha256='4'*64,
                clock_hz='187498125.0', board_id='board', compute_ns=str(compute[group] if compute else 100),
                transfer_ns='200', call_ns=str(float(value)), compared_elements='100', mismatched_elements='0',
                observed_output_sha256='4'*64, batch_id=batch, group_id='group'+str(group),
                group_size=str(size), group_call_total_ns=str(value*size))
            result.append(item)
    return result


class AnnotationTests(unittest.TestCase):
    def test_host_unique_groups_not_repeated_member_weight(self):
        rows = group_rows([1000,2000,3000,4000,100000], [1,1,1,1,100])
        result, rules = annotate_rows(rows)
        host = next(row for row in rules['strata'] if row['scope']=='call')
        self.assertEqual((host['observation_count'], host['median_ns'],host['mad_ns']), (5,3000,1000))
        self.assertEqual(host['high_latency_observations'],1)
        self.assertEqual(host['high_latency_mapped_rows'],100)
        self.assertEqual(sum(row['ann_call_observation_weight'] for row in result),5)
        self.assertTrue(all(row['ann_call_high_latency'] is True for row in result[4:]))
        self.assertEqual(len(result),104)

    def test_controlled_pl_slow_value_is_labelled_without_deletion(self):
        rows = group_rows([1000]*5, compute=[10,11,12,13,100])
        result, rules = annotate_rows(rows)
        baseline = next(row for row in rules['strata'] if row['scope']=='compute')
        self.assertEqual(baseline['median_ns'],12)
        self.assertEqual(baseline['mad_ns'],1)
        self.assertTrue(result[-1]['ann_compute_high_latency'])
        self.assertEqual(rules['samples_deleted'],0)
        for original, annotated in zip(rows,result):
            self.assertEqual(original, {key:annotated[key] for key in CSV_FIELDS})

    def test_mad_zero_is_not_a_robust_anomaly_decision(self):
        rows = group_rows([1000]*4, compute=[1,1,1,100])
        result, rules = annotate_rows(rows)
        baseline = next(row for row in rules['strata'] if row['scope']=='compute')
        self.assertEqual(baseline['status'],'MAD_ZERO_NO_ROBUST_DECISION')
        self.assertIsNone(baseline['threshold_ns'])
        self.assertEqual(baseline['high_latency_observations'],0)
        self.assertEqual(result[-1]['ann_compute_high_latency'],'')
        self.assertTrue(result[-1]['ann_compute_above_median'])
        self.assertIn('COMPUTE_ABOVE_MEDIAN_MAD_ZERO_NO_ROBUST_DECISION',json.loads(result[-1]['ann_flags_json']))

    def test_phase_and_batch_do_not_share_baselines_or_group_keys(self):
        rows=group_rows([1000,2000,3000],phase='warmup')+group_rows([4000,5000,6000])+group_rows([7000,8000,9000],batch='b2')
        _,rules=annotate_rows(rows)
        hosts={(row['batch_id'],row['phase']):row for row in rules['strata'] if row['scope']=='call'}
        self.assertEqual(len(hosts),3)
        self.assertEqual(hosts[('b1','warmup')]['median_ns'],2000)
        self.assertEqual(hosts[('b1','measured')]['median_ns'],5000)
        self.assertEqual(hosts[('b2','measured')]['median_ns'],8000)
        self.assertEqual(rules['invalid_rows'],0)

    def test_invalid_values_and_groups_are_retained(self):
        rows=group_rows([1000,2000,3000])
        rows[0]['compute_ns']='NaN'
        rows[1]['group_size']='2'
        rows[1]['call_ns']='1000'
        rows[2]['__anomaly_field_count']=len(CSV_FIELDS)+1
        rows[2]['__anomaly_extra_fields']=['preserve_extra']
        result,rules=annotate_rows(rows)
        self.assertEqual(len(result),3)
        self.assertEqual(rules['invalid_rows'],3)
        self.assertEqual(result[0]['compute_ns'],'NaN')
        self.assertEqual(result[0]['ann_compute_status'],'INVALID_NUMERIC_VALUE')
        self.assertEqual(result[1]['ann_call_status'],'INVALID_HOST_GROUP')
        self.assertEqual(json.loads(result[2]['ann_extra_fields_json']),['preserve_extra'])

    def test_duplicate_iteration_flagged(self):
        rows=group_rows([1000,2000])
        rows[1]['iteration']=rows[0]['iteration']
        result,rules=annotate_rows(rows)
        self.assertEqual(rules['invalid_rows'],2)
        self.assertTrue(all('INVALID_DUPLICATE_PHASE_ITERATION' in json.loads(row['ann_flags_json']) for row in result))

    def test_schema_conflicts_and_input_output_alias_rejected(self):
        for fields in (list(CSV_FIELDS)+['ann_flags_json'], list(CSV_FIELDS)+['case_id'], []):
            with self.assertRaises(ValueError):
                annotate_rows([],input_fields=fields)
        with self.assertRaises(ValueError):
            annotate_csv('source.csv','source.csv','new_rules.json')
        with self.assertRaises(ValueError):
            annotate_csv('source.csv','new_output.csv','new_output.csv')
        with self.assertRaises(ValueError):
            annotate_csv('source.csv','manifest.json','rules.json',manifest_path='manifest.json')


if __name__=='__main__':
    unittest.main()
