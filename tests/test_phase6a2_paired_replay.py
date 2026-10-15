"""Regression tests for replay information boundaries, matching and accounting."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/sonar_spec'))
import phase6a2_paired_replay as audit


@pytest.fixture(scope='module')
def data():
    return audit.prior.load_pool()


def test_frozen_table_binding_and_seed42_identity(data):
    audit.audit_data(data)


def test_encoding_matches_canonical_validator(data):
    sys.path.insert(0, str(ROOT/'src'))
    from hwnas_fpga.models.sonar_supernet import enumerate_architectures, architecture_id
    canonical = {architecture_id(a) for a in enumerate_architectures()}
    assert canonical == set(data['by_id']) and len(canonical) == 144
    assert audit.prior.encoding_audit(data)['minimal_model_bits']['n_unique_keys'] == 144


def test_budget40_features_do_not_read_future_metrics(data):
    from copy import deepcopy
    original = audit.make_features(data, 'mf40')
    changed = deepcopy(data)
    for table in ['stage2', 'final']:
        for row in changed[table].values():
            for key in ['mAP50_95','endpoint_mAP50_95','AP50','AP75','Precision','Recall']:
                row[key] = '999'
    changed['y'] = np.full_like(changed['y'], 999)
    changed['map60'] = np.full_like(changed['map60'], 999)
    actual = audit.make_features(changed, 'mf40')
    np.testing.assert_array_equal(original[0], actual[0])
    np.testing.assert_array_equal(original[1], actual[1])
    assert not any('60' in k or '80' in k for k in actual[2])


@pytest.mark.parametrize('family', ['ridge','et','pairwise'])
def test_predictor_uses_observed_labels_only_and_is_deterministic(data, family):
    X, base, _ = audit.make_features(data, 'mf40')
    tr = np.arange(20); remain=np.arange(20,72)
    a = audit.predict(family, X, base, tr, data['y'][tr], remain, 42)
    b = audit.predict(family, X, base, tr, data['y'][tr], remain, 42)
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError, match='overlap'):
        audit.predict(family,X,base,tr,data['y'][tr],np.array([0,25]),42)


def test_pairwise_is_invariant_to_training_row_order(data):
    X, base, _=audit.make_features(data,'arch')
    tr=np.arange(20);remain=np.arange(20,72)
    a=audit.predict('pairwise',X,base,tr,data['y'][tr],remain,42)
    b=audit.predict('pairwise',X,base,tr[::-1],data['y'][tr[::-1]],remain,42)
    np.testing.assert_allclose(a,b,atol=1e-9,rtol=1e-9)


def test_paired_initialization_tail_and_cost(data):
    seed=61000;init=audit.prior.stratified_indices(data['combos'],20,seed)
    features={m:audit.make_features(data,m) for m in ['arch','mf40','mf60']}
    traces=[]
    for method in ['random','direct40','ridge_arch']:
        t,s,b,p=audit.replay(data,method,seed,init,features)
        assert [r['evaluated'] for r in t] == [20,26,32,38,44,50,56]
        assert len({r['architecture_id'] for r in s})==36
        assert not ({data['pool_ids'][i] for i in init}&{r['architecture_id'] for r in s})
        assert b['n_holdout']==52
        assert all(t[i]['regret']>=t[i+1]['regret'] for i in range(6))
        expected=72*40+40*56 if method=='direct40' else 80*56
        assert t[-1]['candidate_epochs_total']==expected
        traces.append(t)
    assert len({t[0]['regret'] for t in traces})==1


def test_hypervolume_and_pareto():
    y=np.array([.02,.03,.01]);costs=np.array([2e9,4e9,5e9])
    assert set(audit.pareto_indices(y,costs))=={0,1}
    assert audit.hypervolume(y,costs)==pytest.approx(.02/.1*.8+.01/.1*.6)


def test_bootstrap_accepts_mean_improvement_with_zero_median():
    ci=audit.mean_ci([0]*18+[.001]*12)
    assert ci['median']==0 and ci['low']>0
