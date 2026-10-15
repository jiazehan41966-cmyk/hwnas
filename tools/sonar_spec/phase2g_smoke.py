"""Fail-closed Phase2G smoke gate and software deployment evidence generator."""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools/sonar_analysis'), str(ROOT/'scripts'), str(ROOT/'tools/sonar_spec')]
import baseline as frozen
from formal_protocol import sha, dump, verify_baseline
from hwnas_fpga.deploy.export import get_model_info
from hwnas_fpga.models.sonar_candidate import SPEC_PATH, build_sonar_candidate
from hwnas_fpga.models.sonar_op import (A1_ARCHITECTURE, A2_ARCHITECTURE,
    ReparamDualRFDepthwise, build_sonarop_candidate, fold_standard_stage7_depthwise)
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss

REPORT = ROOT/'reports/phase2g_sonarop'
BASELINE_PATH = ROOT/'configs/baseline_sonar_frozen.yaml'
EXPECTED = dict(
    search_space='ebb1708b394af20258c920eb29204d5cc9504f5908fdfdae62a62d78042eb7c4',
    baseline='23989d09cae29139be7bd74a92bf9c53f75605047b7d8b2d2e2601b98d2eb56e',
    phase2f='d3d31ab74490da0abbc6c18cf354a090cbe6f73c0c795b0c0e3ea41405af54a1')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def errors(a, b):
    delta=(a-b).abs(); denom=a.abs().clamp_min(1e-12)
    return dict(max_abs_error=float(delta.max()), mean_abs_error=float(delta.mean()),
                max_relative_error=float((delta/denom).max()))


class Predictions(nn.Module):
    def __init__(self, model): super().__init__(); self.model=model
    def forward(self, x): return self.model(x)['predictions']


def complexity(name, form, model):
    info=get_model_info(model, (1,1,640,640))
    op=model.sonarop() if hasattr(model,'sonarop') else model.backbone.stages[7][0].dw_conv
    op_params=sum(p.numel() for p in op.parameters())
    convs=[m for m in op.modules() if isinstance(m,nn.Conv2d)] if isinstance(op,nn.Module) else [op]
    op_macs=sum(m.kernel_size[0]*m.kernel_size[1]*(m.in_channels//m.groups)*20*20*m.out_channels for m in convs)
    return dict(model=name,form=form,Params=info['total_params'],MACs=info['estimated_macs'],
                FLOPs_2x_MACs=2*info['estimated_macs'],stage7_depthwise_params=op_params,
                stage7_depthwise_MACs=op_macs)


def main():
    REPORT.mkdir(parents=True,exist_ok=True); (REPORT/'figures').mkdir(exist_ok=True)
    actual=dict(search_space=sha(SPEC_PATH),baseline=sha(BASELINE_PATH),
                phase2f=sha(ROOT/'reports/phase2f_multiseed/phase2f_manifest.json'))
    if actual!=EXPECTED: raise ValueError(f'Frozen protection mismatch: {actual}')
    baseline=yaml.safe_load(BASELINE_PATH.read_text(encoding='utf-8')); verify_baseline(baseline)
    frozen.seed_all(42)

    fusion=[]
    for batch in (1,4):
        op=ReparamDualRFDepthwise(480).train()
        for _ in range(2): op(torch.randn(4,480,20,20))
        op.eval(); x=torch.randn(batch,480,20,20)
        k1,b1=op.get_equivalent_kernel_bias(); k2,b2=op.get_equivalent_kernel_bias()
        if not torch.equal(k1,k2) or not torch.equal(b1,b2): raise ValueError('Non-deterministic fusion')
        with torch.no_grad(): expected=op(x); deployed=deepcopy(op).switch_to_deploy(); actual_y=deployed(x)
        e=errors(expected,actual_y); fusion.append(dict(scope=f'operator_batch{batch}',**e))
        if e['max_abs_error']>1e-5: raise ValueError(f'Operator fusion error {e}')

    gradop=ReparamDualRFDepthwise(480).train(); y=gradop(torch.randn(4,480,20,20)).mean(); y.backward()
    required=[gradop.branch3[0].weight.grad,gradop.branch3[1].weight.grad,gradop.branch3[1].bias.grad,
              gradop.branch5[0].weight.grad,gradop.branch5[1].weight.grad,gradop.branch5[1].bias.grad]
    if any(g is None or not torch.isfinite(g).all() for g in required): raise ValueError('Branch/BN gradient failure')

    train_model=build_sonarop_candidate().eval(); deploy_model=deepcopy(train_model).switch_to_deploy().eval()
    fixed=torch.randn(1,1,640,640)
    with torch.no_grad(): left=train_model(fixed); right=deploy_model(fixed)
    all_left=torch.cat([x.flatten() for x in left['predictions']]); all_right=torch.cat([x.flatten() for x in right['predictions']])
    detector_error=errors(all_left,all_right); fusion.append(dict(scope='detector_predictions',**detector_error))
    if detector_error['max_abs_error']>1e-5: raise ValueError(f'Detector fusion error {detector_error}')
    write_csv(REPORT/'fusion_equivalence.csv',fusion)

    train_state=REPORT/'sonarop_train_form_state.pt'; deploy_state=REPORT/'sonarop_deploy_form_state.pt'
    torch.save(train_model.state_dict(),train_state); torch.save(deploy_model.state_dict(),deploy_state)
    train_reload=build_sonarop_candidate().eval(); train_reload.load_state_dict(torch.load(train_state,weights_only=True),strict=True)
    deploy_reload=build_sonarop_candidate(deploy=True).eval(); deploy_reload.load_state_dict(torch.load(deploy_state,weights_only=True),strict=True)
    with torch.no_grad():
        if errors(torch.cat([x.flatten() for x in train_model(fixed)['predictions']]),torch.cat([x.flatten() for x in train_reload(fixed)['predictions']]))['max_abs_error']!=0: raise ValueError('Train state roundtrip failed')
        if errors(torch.cat([x.flatten() for x in deploy_model(fixed)['predictions']]),torch.cat([x.flatten() for x in deploy_reload(fixed)['predictions']]))['max_abs_error']!=0: raise ValueError('Deploy state roundtrip failed')

    a1=build_sonar_candidate(SPEC_PATH,A1_ARCHITECTURE).eval(); a2=build_sonar_candidate(SPEC_PATH,A2_ARCHITECTURE).eval()
    rows=[complexity('A2_DW3','training',a2),complexity('A1_DW5','training',a1),complexity('A3_SonarOp','training',train_model),
          complexity('A2_DW3','deploy',fold_standard_stage7_depthwise(deepcopy(a2))),
          complexity('A1_DW5','deploy',fold_standard_stage7_depthwise(deepcopy(a1))),
          complexity('A3_SonarOp','deploy',deploy_model)]
    write_csv(REPORT/'operator_complexity.csv',rows)
    a1deploy=next(r for r in rows if r['model']=='A1_DW5' and r['form']=='deploy')
    a3deploy=next(r for r in rows if r['model']=='A3_SonarOp' and r['form']=='deploy')
    if (a1deploy['stage7_depthwise_params'],a1deploy['stage7_depthwise_MACs'])!=(a3deploy['stage7_depthwise_params'],a3deploy['stage7_depthwise_MACs']):
        raise ValueError('A3 deploy primitive complexity differs from standard DW5')

    loader=frozen.make_loader(baseline,frozen.resolve(baseline['dataset']['data_dir']),'train',42,0,train=True)
    images,targets=next(iter(loader)); real=build_sonarop_candidate().train(); out=real(images); frozen.finite_outputs(out)
    expected_shapes=[(4,21,80,80),(4,21,40,40),(4,21,20,20)]
    if [tuple(x.shape) for x in out['predictions']]!=expected_shapes: raise ValueError('Detector output shapes changed')
    criterion=DenseAnchorDetectionLoss(real.loss_config(baseline)); loss,parts=criterion(out,targets)
    if not torch.isfinite(loss) or not all(np.isfinite(v) for v in parts.values()): raise ValueError('Non-finite detector loss')
    loss.backward(); op=real.sonarop()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in op.parameters()): raise ValueError('Integrated SonarOp gradient failure')
    valid=frozen.make_loader(baseline,frozen.resolve(baseline['dataset']['data_dir']),'valid',42,0)
    valid_metrics,_=frozen.evaluate(deploy_model,valid,DenseAnchorDetectionLoss(deploy_model.loss_config(baseline)),baseline,'cpu',max_batches=1)

    export_model=Predictions(deploy_model).eval(); onnx_path=REPORT/'a3_deploy.onnx'
    output_names=['P3','P4','P5']; dynamic={'input':{0:'batch'},**{n:{0:'batch'} for n in output_names}}
    torch.onnx.export(export_model,fixed,str(onnx_path),opset_version=13,input_names=['input'],
                      output_names=output_names,dynamic_axes=dynamic,do_constant_folding=True)
    graph=onnx.load(str(onnx_path)); onnx.checker.check_model(graph)
    session=ort.InferenceSession(str(onnx_path),providers=['CPUExecutionProvider'])
    ort_outputs=session.run(output_names,{'input':fixed.numpy()})
    with torch.no_grad(): pt_outputs=export_model(fixed)
    export_rows=[]
    for name,pt,ort_value,stride in zip(output_names,pt_outputs,ort_outputs,(8,16,32)):
        e=errors(pt,torch.from_numpy(ort_value)); export_rows.append(dict(level=name,stride=stride,
            pytorch_shape=str(tuple(pt.shape)),onnx_shape=str(tuple(ort_value.shape)),**e))
        if e['max_abs_error']>1e-4: raise ValueError(f'ONNX Runtime parity failed {name}: {e}')
    write_csv(REPORT/'export_equivalence.csv',export_rows)

    sources=[ROOT/'src/hwnas_fpga/models/sonar_op/__init__.py',ROOT/'src/hwnas_fpga/models/sonar_op/sonar_dw_reparam.py',
             ROOT/'src/hwnas_fpga/models/sonar_op/candidate.py',ROOT/'tools/sonar_spec/phase2g_smoke.py',
             ROOT/'tools/sonar_spec/phase2g_runner.py',ROOT/'tools/sonar_spec/phase2g_analysis.py']
    manifest=dict(sonarop_version='v0.2-p0',status='SMOKE_PASS',deploy_primitive='EXISTING_DW5',
        source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sources},
        frozen_sha256=EXPECTED,train_state_sha256=sha(train_state),deploy_state_sha256=sha(deploy_state),
        onnx_sha256=sha(onnx_path),fusion_max_abs_error=max(r['max_abs_error'] for r in fusion),
        onnx_max_abs_error=max(r['max_abs_error'] for r in export_rows),valid_smoke_mAP50_95=valid_metrics['mAP50_95'],
        test_evaluated=False)
    dump(REPORT/'implementation_manifest.json',manifest)
    (REPORT/'smoke_validation.md').write_text(f'''# Phase 2G Smoke Validation

Status: **PASS**

- Shape and backward: PASS, including both DW branches and BN affine gradients.
- Train/deploy state_dict save/load: PASS.
- Operator fusion max absolute error: {max(r['max_abs_error'] for r in fusion if r['scope'].startswith('operator')):.9g}.
- Detector fusion max absolute error: {detector_error['max_abs_error']:.9g}.
- Real batch forward/loss/backward: PASS; loss={float(loss):.6f}.
- Validation evaluator compatibility: PASS (one valid batch; not a performance result).
- ONNX Runtime deploy parity max absolute error: {max(r['max_abs_error'] for r in export_rows):.9g}.
- Output order/shapes: P3/P4/P5 at strides 8/16/32, PASS.
- V0.1 and frozen baseline hashes: PASS.
- Deploy primitive: `EXISTING_DW5` structurally; integer/HLS parity remains untested.
- Test accessed: false.
''',encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
