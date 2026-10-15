from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.hardware_measurement_pilot.schedule_scenarios import schedule_pipeline, repeated_build_mix
from tools.hardware_measurement_pilot.report import _pilot_schedule, _schedule_markdown, _d07_rows


class ScheduleTests(unittest.TestCase):
    def jobs(self, board=4):
        return [dict(job_id=str(i), build_seconds=10, board_seconds=board, board_accepted=True) for i in range(6)]

    def test_three_build_workers_one_board(self):
        result = schedule_pipeline(self.jobs(), implementation_coverage_complete=True, physical_bindings_complete=True)
        self.assertEqual(result['conditional_build_makespan_seconds'], 20)
        self.assertEqual(result['board_occupancy_seconds'], 24)
        self.assertEqual(result['full_measurement_makespan_seconds'], 34)
        windows = sorted((r['board_start_seconds'], r['board_finish_seconds']) for r in result['jobs'])
        self.assertTrue(all(windows[i][1] <= windows[i+1][0] for i in range(5)))
        self.assertTrue(all(r['build_finish_seconds'] <= r['board_start_seconds'] for r in result['jobs']))

    def test_board_bottleneck_never_divided_by_workers(self):
        result = schedule_pipeline(self.jobs(20))
        self.assertEqual(result['conditional_pipeline_makespan_seconds'], 130)
        self.assertIsNone(result['full_measurement_makespan_seconds'])

    def test_missing_or_unaccepted_board_cannot_be_zero(self):
        for key, value in [('board_seconds', None), ('board_accepted', False)]:
            jobs = self.jobs()
            jobs[2][key] = value
            result = schedule_pipeline(jobs, implementation_coverage_complete=True, physical_bindings_complete=True)
            self.assertEqual(result['conditional_build_makespan_seconds'], 20)
            self.assertIsNone(result['conditional_pipeline_makespan_seconds'])
            self.assertIsNone(result['full_measurement_makespan_seconds'])

    def test_build_mix_is_condition_only(self):
        result = repeated_build_mix({'long': 9, 'short': 3}, 4, build_workers=2)
        self.assertEqual(result['conditional_build_makespan_seconds'], 12)
        self.assertIsNone(result['full_measurement_makespan_seconds'])
        self.assertEqual(result['board_workers'], 1)

    def test_observed_cpu_deviation_has_conditional_time_without_acceptance(self):
        jobs = self.jobs()
        for job in jobs:
            job.update(board_accepted=False, board_observation_validated=True)
        result = schedule_pipeline(jobs, allow_observed_protocol_deviations=True,
            implementation_coverage_complete=True, physical_bindings_complete=True)
        self.assertEqual(result['conditional_pipeline_makespan_seconds'],34)
        self.assertIs(result['accepted_board_occupancy_complete'],False)
        self.assertIs(result['observed_board_occupancy_complete'],True)
        self.assertIsNone(result['full_measurement_makespan_seconds'])
        self.assertIn('BOARD_PROTOCOL_CONDITIONS_NOT_MET_OBSERVATIONS_RETAINED',result['missing_evidence'])

    def test_each_evidence_gate_requires_boolean_true(self):
        for field in ('implementation_coverage_complete', 'physical_bindings_complete'):
            for value in (False, None, 'false', 'true', 1):
                gates = dict(implementation_coverage_complete=True, physical_bindings_complete=True)
                gates[field] = value
                result = schedule_pipeline(self.jobs(), **gates)
                self.assertEqual(result['conditional_pipeline_makespan_seconds'], 34)
                self.assertIsNone(result['full_measurement_makespan_seconds'])

    def test_invalid_duration_or_ids_rejected(self):
        for value in (float('nan'), float('inf'), 0, -1, True):
            with self.assertRaises(ValueError):
                schedule_pipeline([dict(job_id='a', build_seconds=value)])
        with self.assertRaises(ValueError):
            schedule_pipeline([dict(job_id='a', build_seconds=1)] * 2)


class PilotScheduleGateTests(unittest.TestCase):
    def setUp(self):
        self.cases = [dict(case_id=f'case{index:02}', case_sha256='a' * 64,
            hls_wall_seconds=5, route_wall_seconds=5, board_runtime_seconds=2, probe_wall_seconds=2,
            board_correctness='PASS_ALL_BATCH_FIXTURES', board_timing='PASS',
            phase_clock_evidence_verified=True, phase_evidence={'status': 'PASS_BOUND_PHASE_EVIDENCE'},
            static_interface_acceptance='PASS_STATIC_ROUTE_RESET_IDENTITY',
            board='STATISTICS_PASS_WITH_CPU_GOVERNOR_DEVIATION', hardware_measurement_eligible=False,
            board_condition_deviations=['CPU_GOVERNOR_UNAVAILABLE'], hls_status_source='hls.json',
            route_status_source='route.json', board_runtime_source='timing.json', probe_runtime_source='probe.json')
            for index in range(14)]
        self.inventory = dict(pilot_case_count=14, unique_primitive_count=65,
            pilot_cases=[{'case_id': row['case_id']} for row in self.cases])
        self.attempt = Path(__file__).resolve().parent / 'unused_test_attempt'
        read_patch = patch('tools.hardware_measurement_pilot.report.read_json', return_value={'jobs': 3})
        read_patch.start()
        self.addCleanup(read_patch.stop)

    def test_complete_cpu_deviation_mix_has_only_conditional_time(self):
        result = _pilot_schedule(self.cases, self.inventory, self.attempt)
        self.assertEqual(result['observed_case_count'], 14)
        self.assertEqual(len(result['scenarios']), 3)
        self.assertTrue(all(row['board_accepted'] is False for row in result['observations']))
        self.assertEqual([row['conditional_pipeline_hours'] * 3600 for row in result['scenarios']], [530, 790, 1570])
        for row in result['scenarios']:
            self.assertIsNone(row['full_measurement_calendar_days'])
            self.assertIsNone(row['schedule']['full_measurement_makespan_seconds'])
            self.assertIs(row['schedule']['accepted_board_occupancy_complete'], False)
            self.assertIs(row['schedule']['observed_board_occupancy_complete'], True)

    def test_missing_inventory_cases_never_become_a_complete_mix(self):
        for count in (0, 1, 13):
            with self.subTest(count=count):
                result = _pilot_schedule(self.cases[:count], self.inventory, self.attempt)
                self.assertEqual(result['status'], 'WAIT_COMPLETE_14_CASE_PILOT_MIX')
                self.assertEqual(result['expected_pilot_case_count'], 14)
                self.assertEqual(len(result['missing_case_ids']), 14 - count)
                self.assertEqual(result['scenarios'], [])

    def test_cpu_deviation_status_cannot_be_promoted_by_inconsistent_eligibility(self):
        cases = deepcopy(self.cases)
        for row in cases:
            row['hardware_measurement_eligible'] = True
        result = _pilot_schedule(cases, self.inventory, self.attempt)
        self.assertTrue(all(row['board_accepted'] is False for row in result['observations']))
        for row in result['scenarios']:
            self.assertIs(row['schedule']['accepted_board_occupancy_complete'], False)
            self.assertIsNone(row['schedule']['full_measurement_makespan_seconds'])

    def test_duplicate_unexpected_and_malformed_inventory_ids_rejected(self):
        for cases in (self.cases + [deepcopy(self.cases[0])],
                      self.cases + [dict(self.cases[0], case_id='unexpected')]):
            with self.assertRaises(ValueError):
                _pilot_schedule(cases, self.inventory, self.attempt)
        for mutation in ('count', 'missing', 'duplicate'):
            inventory = deepcopy(self.inventory)
            if mutation == 'count': inventory['pilot_case_count'] = 13
            elif mutation == 'missing': inventory['pilot_cases'].pop()
            else: inventory['pilot_cases'][-1] = dict(inventory['pilot_cases'][0])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                _pilot_schedule(self.cases, inventory, self.attempt)

    def test_observation_order_follows_bound_inventory(self):
        result = _pilot_schedule(list(reversed(self.cases)), self.inventory, self.attempt)
        self.assertEqual([row['case_id'] for row in result['observations']],
                         [row['case_id'] for row in self.inventory['pilot_cases']])

    def test_timing_pass_cannot_bypass_binding_or_duration_gates(self):
        mutations = [('phase_evidence', {'status': 'NOT_RUN'}),
            ('static_interface_acceptance', 'RESET_INTERFACE_INVALID_NOT_BOARD_ACCEPTED'),
            ('board', 'TIMING_PASS_PHASE_PENDING'), ('phase_clock_evidence_verified', False),
            ('board_correctness', 'FAIL'), ('board_timing', 'FAIL'),
            ('hls_wall_seconds', True), ('route_wall_seconds', float('nan')),
            ('board_runtime_seconds', None), ('probe_wall_seconds', 0)]
        for field, value in mutations:
            cases = deepcopy(self.cases)
            cases[0][field] = value
            with self.subTest(field=field):
                result = _pilot_schedule(cases, self.inventory, self.attempt)
                self.assertEqual(result['scenarios'], [])
                self.assertEqual(result['missing_case_ids'], ['case00'])

    def test_over_21_conditional_workdays_does_not_fill_full_calendar(self):
        cases = deepcopy(self.cases)
        for row in cases:
            row.update(board_runtime_seconds=3600, probe_wall_seconds=3600)
        result = _pilot_schedule(cases, self.inventory, self.attempt)
        self.assertIs(result['conditional_time_trigger_over_21_workdays'], True)
        self.assertIsNone(result['full_measurement_time_trigger_over_21_days'])
        self.assertIsNone(result['full_measurement_calendar_days'])
        data = {'extrapolation': {'conditional_pilot_schedule': result}}
        table = _schedule_markdown(data)
        self.assertIn('已测工具+板端worker阶段条件工作日（8h/日）', table)
        self.assertIn('已测工具+板端worker阶段条件工作日（24h/日）', table)
        self.assertIn('NOT_ESTIMABLE', _d07_rows(data)[1][1])


if __name__ == '__main__':
    unittest.main()
