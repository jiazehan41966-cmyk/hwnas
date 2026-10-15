"""Engineering gates on synthetic measurements; these are not board evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.sonar_joint_v2.deployment_budget import assess

ROOT = Path(__file__).resolve().parents[1]
PREREG = json.loads((ROOT / 'artifacts/zcu104_full_campaign_v1/control/hardware_gate_preregistration_20260810.json').read_text())


def measurements():
    return dict(LUT=10000, DSP=100, BRAM18=100, URAM=0, wns_ns=.25, tns_ns=0,
                implemented_period_ns=5, board_e2e_ms=20, dynamic_power_w=5,
                dynamic_energy_mj=100, junction_temperature_c=50, ambient_temperature_c=25,
                idle_active_pair_repetitions=5, capture_duration_s=60, idle_sample_count=100,
                active_sample_count=100, active_completed_ops=1000, latency_timed_ops=1000,
                coefficient_of_variation=.01, same_integer_valid99=True,
                matched_candidate_quant_mapping_bitstream=True, power_above_noise_floor=True)


def test_all_numerical_gates_do_not_self_authorize_deployment():
    result = assess(measurements(), PREREG)
    assert result['status'] == 'NUMERICAL_BUDGET_PASS'
    assert result['full_deployment_acceptance'] is False


@pytest.mark.parametrize('changes', [
    dict(board_e2e_ms=131911), dict(dynamic_energy_mj=27197),
    dict(wns_ns=.161), dict(implemented_period_ns=5.333),
    dict(latency_timed_ops=10, active_completed_ops=10),
    dict(idle_active_pair_repetitions=1, idle_sample_count=50),
    dict(BRAM18=437), dict(power_above_noise_floor=False),
])
def test_functional_or_ten_call_pass_cannot_bypass_engineering_budget(changes):
    values = measurements(); values.update(changes)
    assert assess(values, PREREG)['status'] == 'BUDGET_FAIL'


@pytest.mark.parametrize('changes', [dict(dynamic_energy_mj=-1), dict(dynamic_power_w=float('nan')),
                                    dict(matched_candidate_quant_mapping_bitstream=None),
                                    dict(same_integer_valid99=None)])
def test_invalid_or_unbound_evidence_never_passes(changes):
    values = measurements(); values.update(changes)
    assert assess(values, PREREG)['status'] == 'INCOMPLETE_BUDGET_EVIDENCE'


def test_budget_checker_does_not_mutate_measurements_or_preregistration():
    values = measurements(); before = deepcopy(values); gate = deepcopy(PREREG)
    assess(values, gate)
    assert values == before and gate == PREREG
