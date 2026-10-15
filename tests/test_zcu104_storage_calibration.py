import copy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hwnas_fpga.hardware.storage_calibration import fit_coupled_storage,predict_coupled_storage

def samples():
    result=[]
    for i,(hls,request) in enumerate([(900,600),(1000,640),(1100,680)]):
        raw=dict(lut=15000,lut_as_logic=10000,lut_as_memory=5000,ff=20000,dsp=100,bram36=220+10*i)
        extra=max(0,request-624)
        actual=dict(raw,bram36=min(312,request/2),lut_as_logic=10500+8*extra,
                    lut_as_memory=6000+256*extra,ff=20100+2*extra)
        actual['lut']=actual['lut_as_logic']+actual['lut_as_memory']
        result.append(dict(architecture_sha256=str(i),bram18_hls_sum=hls,requested_bram18=request,operator_sum=raw,actual=actual))
    return result

def test_bram_saturation_has_corresponding_lutram_cost():
    training=samples();model=fit_coupled_storage(training)
    predicted,storage=predict_coupled_storage(training[1]['operator_sum'],1050,model)
    assert storage['requested_bram18']==pytest.approx(660)
    assert predicted['bram36']==312
    assert predicted['lut_as_memory']==pytest.approx(6000+256*36)
    assert not storage['domain_unknown']

def test_unseen_overflow_is_unknown_instead_of_free_bram_clipping():
    training=samples();model=fit_coupled_storage(training)
    _,storage=predict_coupled_storage(training[0]['operator_sum'],1500,model)
    assert storage['domain_unknown']

def test_duplicate_architectures_cannot_inflate_training_sample_count():
    training=samples();training.append(copy.deepcopy(training[0]))
    with pytest.raises(ValueError,match='Duplicate training architecture'):
        fit_coupled_storage(training)

def test_heldout_excess_memory_remains_a_prediction_error():
    training=samples();model=fit_coupled_storage(training)
    original=copy.deepcopy(model)
    predicted,_=predict_coupled_storage(training[1]['operator_sum'],1050,model)
    heldout_actual_memory=40000
    assert predicted['lut_as_memory']-heldout_actual_memory<0
    assert model==original
