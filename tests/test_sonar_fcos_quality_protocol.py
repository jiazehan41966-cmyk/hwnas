"""Acceptance gates use all six fresh paired runs and percentage-point units."""
from copy import deepcopy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.sonar_fcos_quality.campaign import paired_decision


def rows():
    return [{'variant': variant, 'seed': seed, 'trained_epochs': 30,
             'ap75': 40. + (2. if variant == 'QFL_DFL' else 0.),
             'ap85': 9., 'map': 48., 'nms_first_loss': 300 if variant == 'QFL_DFL' else 400,
             'raw_no_good_box': 448, 'wrong_class_fp': 577}
            for seed in (42, 43, 44) for variant in ('AAA', 'QFL_DFL')]


def test_exact_user_thresholds_pass_and_historical_333_is_not_denominator():
    result = paired_decision(rows())
    assert result['status'] == 'PASS'
    assert result['mean_ap75_gain_pp'] == 2.
    assert result['mean_nms_first_loss_relative_reduction'] == .25
    assert result['paired_ap75_gains_pp'] == [2., 2., 2.]


def test_both_gates_required_and_partial_never_passes():
    values = rows()
    values[1]['ap75'] = 41.999
    assert paired_decision(values)['status'] == 'FAIL'
    values = rows()
    values[1]['nms_first_loss'] = 301
    assert paired_decision(values)['status'] == 'FAIL'
    assert paired_decision(values[:-1])['status'] == 'PENDING'
    values[-1]['trained_epochs'] = 29
    assert paired_decision(values)['status'] == 'PENDING'


def test_duplicate_or_zero_denominator_does_not_pass():
    values = rows()
    with pytest.raises(ValueError, match='Duplicate'):
        paired_decision(values + [deepcopy(values[0])])
    for value in values:
        value['nms_first_loss'] = 0
    result = paired_decision(values)
    assert result['status'] == 'FAIL'
    assert result['mean_nms_first_loss_relative_reduction'] is None
