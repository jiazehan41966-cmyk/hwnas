"""Tests of diagnostic matching, score bounds and sealed split guard."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('confidence_followup', Path(__file__).resolve().parents[1]/'tools/sonar_operator_pool_p0/confidence_followup.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record():
    return dict(image='train/images/a.jpg', target=dict(boxes=[[0,0,10,10]],labels=[1]),
                prediction=dict(boxes=[[20,20,30,30],[0,0,10,10],[0,0,10,10]],
                                scores=[.9,.4,.3],labels=[1,1,1]))


def test_duplicate_and_background_ordering():
    preds,gt=MODULE.analyze_record(record(),dict(split='train',seed=42,checkpoint='best'))
    assert sum(p['tp50'] for p in preds)==1
    assert preds[0]['background'] and not preds[0]['tp50']
    assert preds[1]['tp50'] and not preds[2]['tp50']
    assert gt[0]['background_outranking_correct']==1
    assert gt[0]['correct_score_below_05']
    assert not gt[0]['objectness_definitely_below_05']


def test_probability_product_bounds():
    for obj in np.linspace(0,1,11):
        for cls in np.linspace(.5,1,11):
            lo,hi=MODULE.probability_bounds(obj*cls)
            assert lo-1e-12<=obj<=hi+1e-12
    with pytest.raises(ValueError):MODULE.probability_bounds(1.1)


def test_low_score_does_not_imply_exact_component():
    r=record();r['prediction']['scores'][1]=.2;r['prediction']['scores'][2]=.1
    _,g=MODULE.analyze_record(r,dict(split='train'))
    assert g[0]['objectness_definitely_below_05']


def test_wrong_class_not_true_positive():
    r=record();r['prediction']['labels']=[0,0,0]
    p,g=MODULE.analyze_record(r,dict(split='train'))
    assert not any(x['tp50'] for x in p)
    assert not g[0]['correct_retained_exists']


def test_empty_predictions_and_invalid_multiple_gt():
    r=record();r['prediction']={k:[] for k in ('boxes','scores','labels')}
    p,g=MODULE.analyze_record(r,dict(split='train'))
    assert not p and not g[0]['correct_retained_exists']
    r['target']['boxes'].append([1,1,3,3]);r['target']['labels'].append(1)
    with pytest.raises(ValueError):MODULE.analyze_record(r,dict(split='train'))


def test_test_split_rejected():
    with pytest.raises(ValueError):MODULE.analyze_record(record(),dict(split='test'))
    r=record();r['image']='test/images/a.jpg'
    with pytest.raises(ValueError):MODULE.analyze_record(r,dict(split='train'))
