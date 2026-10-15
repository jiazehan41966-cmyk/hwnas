"""Numerical deployment-budget checks; never a substitute for bound board proof.

The preregistered full-route/board/measurement-quality gates remain separate
from HLS operator screening and from the ten-call performance contract.
"""
from __future__ import annotations

import math


def assess(metrics, preregistration):
    checks = {}

    def check(name, value, bound, direction='max', minimum=None):
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        if minimum is not None:
            valid = valid and value >= minimum
        passed = valid and (value <= bound if direction == 'max' else value >= bound if direction == 'min' else value == bound)
        checks[name] = dict(value=value, bound=bound, direction=direction,
                            status='PASS' if passed else 'FAIL' if valid else 'MISSING_OR_INVALID')

    resource = preregistration['resource_gate']
    timing = preregistration['timing_gate']
    board = preregistration['board_performance_gate']
    quality = preregistration['measurement_quality_gate']
    for key, source in (('LUT', 'lut_max'), ('DSP', 'dsp_max'), ('BRAM18', 'bram18_max'), ('URAM', 'uram_max')):
        check(key, metrics.get(key), resource[source], minimum=0)
    check('wns_ns', metrics.get('wns_ns'), timing['wns_ns_min'], 'min')
    check('tns_ns', metrics.get('tns_ns'), timing['tns_ns_required'], 'equal')
    check('implemented_period_ns', metrics.get('implemented_period_ns'), timing['clock_period_ns'], minimum=0)
    check('board_e2e_ms', metrics.get('board_e2e_ms'), board['latency_board_ms_max'], minimum=0)
    check('dynamic_power_w', metrics.get('dynamic_power_w'), board['dynamic_power_w_max'], minimum=0)
    check('dynamic_energy_mj', metrics.get('dynamic_energy_mj'), board['dynamic_energy_mj_per_op_max'], minimum=0)
    check('junction_temperature_c', metrics.get('junction_temperature_c'), preregistration['thermal_gate']['junction_temperature_c_max'])
    check('ambient_temperature_c', metrics.get('ambient_temperature_c'), preregistration['thermal_gate']['ambient_temperature_c_max'])
    for field, source in (
        ('idle_active_pair_repetitions', 'idle_active_pair_repetitions_min'),
        ('capture_duration_s', 'capture_duration_s_min'),
        ('idle_sample_count', 'idle_sample_count_min'),
        ('active_sample_count', 'active_sample_count_min'),
        ('active_completed_ops', 'active_completed_ops_min'),
        ('latency_timed_ops', 'latency_timed_ops_min')):
        check(field, metrics.get(field), quality[source], 'min', minimum=0)
    check('coefficient_of_variation', metrics.get('coefficient_of_variation'), quality['coefficient_of_variation_max'], minimum=0)
    for field in ('same_integer_valid99', 'matched_candidate_quant_mapping_bitstream', 'power_above_noise_floor'):
        value = metrics.get(field)
        checks[field] = dict(value=value, required=True, status='PASS' if value is True else 'FAIL' if value is False else 'MISSING_OR_INVALID')
    status = ('BUDGET_FAIL' if any(c['status'] == 'FAIL' for c in checks.values()) else
              'INCOMPLETE_BUDGET_EVIDENCE' if any(c['status'] != 'PASS' for c in checks.values()) else 'NUMERICAL_BUDGET_PASS')
    return dict(status=status, checks=checks, full_deployment_acceptance=False,
                scope='numerical_engineering_gate_assessment_requires_separate_artifact_admission')
