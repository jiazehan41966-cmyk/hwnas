"""Phase 2A regression tests; full real-batch validation lives in candidate_sanity."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

from hwnas_fpga.models.sonar_candidate import build_sonar_candidate, validate_sonar_architecture, SPEC_PATH
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss, decode_detection_predictions, _assign_targets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/sonar_spec'))
from candidate_sanity import architecture, protection
from candidate_diagnostics import topk_diagnostics

torch.set_num_threads(2)
BASELINE_PATH = ROOT/'configs/baseline_sonar_frozen.yaml'
BASELINE = yaml.safe_load(BASELINE_PATH.read_text(encoding='utf-8'))


@pytest.mark.parametrize('name', ['P3_P4_P5','P2_P3_P4','P2_P3_P4_P5'])
@pytest.mark.parametrize('mode', ['k3','k5','mixed'])
def test_physical_stages_kernels_and_derived_contract(name, mode):
    arch = architecture(name, mode)
    model = build_sonar_candidate(SPEC_PATH, arch)
    expected_stages = 6 if name=='P2_P3_P4' else 8
    assert len(model.backbone.stages)==expected_stages
    assert not any(k.startswith(('backbone.stages.6.', 'backbone.stages.7.')) for k in model.state_dict()) if expected_stages==6 else True
    assert len(model.levels)==len(model.head.towers)==len(model.anchor_sizes)
    for name, k in arch['kernels'].items():
        block = model.backbone.stages[int(name[5:])][0]
        assert block.dw_conv.kernel_size == (k,k)
        assert block.dw_conv.dilation == (1,1)
        assert block.dw_conv.groups == block.expand_conv.out_channels == 3*block.expand_conv.in_channels
    by_level = dict(zip(model.levels,model.anchor_sizes))
    for level in model.levels:
        baseline_index = {'P2':0,'P3':0,'P4':1,'P5':2}[level]
        assert by_level[level] == tuple(map(tuple, BASELINE['detection']['anchor_sizes'][baseline_index]))


def invalid_cases():
    cases=[]
    for field,value in [('output_combination','P2_P5'),('levels',['P4','P2','P2']),('number_of_heads',3),
                         ('anchor_sizes',[]),('expansion',6),('dilation',2),('inactive_stage',7),('operator','SonarOp')]:
        a=architecture('P2_P3_P4_P5'); a[field]=value; cases.append(a)
    for value in [7,1,3.0,True,{'kernel':3,'expansion':6}]:
        a=architecture('P2_P3_P4_P5'); a['kernels']['stage1']=value; cases.append(a)
    a=architecture('P2_P3_P4'); a['kernels']['stage6']=3; cases.append(a)
    a=architecture('P2_P3_P4'); a['kernels']['stage7']=5; cases.append(a)
    a=architecture('P2_P3_P4_P5'); del a['kernels']['stage7']; cases.append(a)
    a=architecture('P3_P4_P5'); a['kernels']['stage3']=3; cases.append(a)
    return cases


@pytest.mark.parametrize('arch', invalid_cases())
def test_invalid_rejected_before_construction(arch):
    with pytest.raises(ValueError): build_sonar_candidate(SPEC_PATH, arch)


@pytest.mark.parametrize('section,value', [('head',{'count':3}),('anchors',{'P2':[]}),('output_combinations',[
    {'id':'P2_P3_P4','levels':['P4','P2','P2'],'last_active_stage':5}])])
def test_invalid_derived_spec_rejected(section,value):
    spec=yaml.safe_load(SPEC_PATH.read_text()); spec['architecture_space'][section]=value
    with pytest.raises(ValueError,match='Specification differs'): validate_sonar_architecture(architecture('P3_P4_P5'),spec)


def test_baseline_equivalence_and_manifest():
    from detection_controlled_study import make_model
    protection(BASELINE_PATH)
    torch.manual_seed(42)
    baseline,_=make_model(BASELINE)
    candidate=build_sonar_candidate(SPEC_PATH,architecture('P3_P4_P5'))
    candidate.load_state_dict(baseline.state_dict(),strict=True)
    baseline.eval(); candidate.eval()
    with torch.no_grad():
        x=torch.randn(1,1,640,640)
        for key,left in baseline(x).items():
            for a,b in zip(left,candidate(x)[key]): torch.testing.assert_close(a,b,rtol=0,atol=0)
    protection(BASELINE_PATH)


def test_duplicate_anchor_best_match_tie_uses_first_level():
    model=build_sonar_candidate(SPEC_PATH,architecture('P2_P3_P4_P5'))
    cfg=model.loss_config(BASELINE)
    predictions=[torch.zeros(1,21,640//s,640//s) for s in cfg.strides]
    target=[dict(boxes=torch.tensor([[99.,99.,101.,101.]]),labels=torch.tensor([0]))]
    assigned=_assign_targets(predictions,target,cfg,device=torch.device('cpu'))
    assert (assigned[0]['objectness']>.5).any()
    assert all(not (a['objectness']>.5).any() for a in assigned[1:])


@pytest.mark.parametrize('name', ['P3_P4_P5','P2_P3_P4','P2_P3_P4_P5'])
def test_decode_known_coordinate_and_topk_parity(name):
    model=build_sonar_candidate(SPEC_PATH,architecture(name))
    cfg=model.loss_config(BASELINE)
    raw=[torch.zeros(1,21,640//s,640//s) for s in cfg.strides]
    # A unique highest-scoring anchor at cell(10,10), level0; zero box logits.
    raw[0][0,4,10,10]=10.; raw[0][0,5,10,10]=10.
    pred=decode_detection_predictions(raw,config=cfg,score_threshold=0.,pre_nms_topk=500,max_detections=100,nms_iou_threshold=.5)[0]
    aw,ah=cfg.anchor_sizes[0][0]; center=10.5*cfg.strides[0]
    torch.testing.assert_close(pred['boxes'][0],torch.tensor([center-aw/2,center-ah/2,center+aw/2,center+ah/2]))
    diag=topk_diagnostics(raw,cfg,model.levels,BASELINE['evaluation_protocol'])
    assert sum(r['global_top500'] for r in diag[0]['levels'])==500
    assert pred['boxes'].shape[0]<=100


@pytest.mark.parametrize('name,mode', [('P3_P4_P5','k3'),('P2_P3_P4','mixed'),('P2_P3_P4_P5','k5')])
def test_real_batch_forward_loss_backward(name,mode):
    import baseline as frozen
    frozen.seed_all(42)
    loader=frozen.make_loader(BASELINE,frozen.resolve(BASELINE['dataset']['data_dir']),'train',42,0,train=True)
    images,targets=next(iter(loader))
    model=build_sonar_candidate(SPEC_PATH,architecture(name,mode)).train()
    out=model(images)
    frozen.finite_outputs(out)
    for level,pred in zip(model.levels,out['predictions']):
        stride={'P2':4,'P3':8,'P4':16,'P5':32}[level]
        assert list(pred.shape)==[4,21,640//stride,640//stride]
    loss,parts=DenseAnchorDetectionLoss(model.loss_config(BASELINE))(out,targets)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(v)) for v in parts.values())
    loss.backward()
    frozen.gradient_summary(model)
