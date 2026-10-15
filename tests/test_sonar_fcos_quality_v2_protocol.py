"""Both endpoints and both coverage safeguards are preregistered gates."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.sonar_fcos_quality_v2.campaign import paired_decision


def rows():
    return [{'view':view,'variant':v,'seed':seed,'trained_epochs':30,
             'ap75':42. if v=='QFL_DFL' else 40.,'ap85':9.,'map':48.,
             'nms_first_loss':300 if v=='QFL_DFL' else 400,
             'raw_no_good_box':440,'final_no_good_box':700 if v=='QFL_DFL' else 828,
             'wrong_class_fp':577}
            for view in ('best','epoch30','mean26_30') for seed in (42,43,44) for v in ('AAA','QFL_DFL')]


def test_two_primary_views_must_pass_stability_is_supplementary():
    values=rows()
    assert paired_decision(values)['status']=='PASS'
    for r in values:
        if r['view']=='epoch30' and r['variant']=='QFL_DFL':r['ap75']=41.
    assert paired_decision(values)['status']=='NON_ROBUST'
    for r in values:
        if r['view']=='best' and r['variant']=='QFL_DFL':r['ap75']=41.
    assert paired_decision(values)['status']=='FAIL'
    values=rows()
    for r in values:
        if r['view']=='mean26_30' and r['variant']=='QFL_DFL':r['ap75']=30.
    result=paired_decision(values)
    assert result['status']=='PASS' and result['views']['mean26_30']['status']=='FAIL'


@pytest.mark.parametrize('field,value',[('final_no_good_box',828),('raw_no_good_box',441)])
def test_fewer_nms_losses_cannot_pass_when_coverage_fails(field,value):
    values=rows()
    for r in values:
        if r['variant']=='QFL_DFL':r[field]=value
    assert paired_decision(values)['status']=='FAIL'


def test_missing_seed_or_endpoint_and_zero_nms_do_not_pass():
    values=[r for r in rows() if not(r['view']=='epoch30' and r['seed']==44)]
    assert paired_decision(values)['status']=='PENDING'
    values=rows()
    for r in values:r['nms_first_loss']=0
    assert paired_decision(values)['status']=='FAIL'
    with pytest.raises(ValueError,match='Duplicate'):paired_decision(rows()+[rows()[0]])
