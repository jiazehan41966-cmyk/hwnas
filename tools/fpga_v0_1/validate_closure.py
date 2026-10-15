"""Validate final artifact consistency and the frozen baseline guard."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/sonar_spec')]
from candidate_sanity import protection
from hwnas_fpga.models.sonar_candidate import build_sonar_candidate,SPEC_PATH
from export_contract import PredictionOutputs,fold_candidate_bn,export_candidate_predictions

OUT=ROOT/'reports/fpga_v0_1_closure'
ART=ROOT/'artifacts/fpga_v0_1_closure/20260914_v1'


def main():
    torch.set_num_threads(2)
    checks={'baseline_freeze':protection(ROOT/'configs/baseline_sonar_frozen.yaml')}
    spec=yaml.safe_load(SPEC_PATH.read_text())
    assert spec['architecture_space']['unique_architecture_count']==144
    assert spec['fpga_compatibility']['active_hardware_constraints'] is None
    manifest=json.loads((OUT/'export_manifest.json').read_text())
    import onnx
    for m in manifest:
        graph=onnx.load(str(ART/'export'/f"{m['case']}.onnx"));onnx.checker.check_model(graph)
        names=[o.name for o in graph.graph.output]
        assert names==[x['name'] for x in m['outputs']]
        if m['architecture']['output_combination']=='P2_P3_P4':assert 'P5' not in names
    checks['onnx_graphs']=len(manifest)
    m=next(x for x in manifest if x['case']=='P2_P3_P4_k3')
    candidate=build_sonar_candidate(SPEC_PATH,m['architecture'])
    candidate.load_state_dict(torch.load(ART/'export'/f"{m['case']}_synthetic_state.pt",weights_only=True))
    try:fold_candidate_bn(candidate)
    except ValueError:pass
    else:raise AssertionError('Training-state BN must not fold')
    x=torch.from_numpy(np.load(ART/'export/input.npy'))
    try:export_candidate_predictions(candidate,ART/'must_not_exist.onnx',x)
    except ValueError:pass
    else:raise AssertionError('Training-state export must reject')
    assert not (ART/'must_not_exist.onnx').exists()
    candidate.eval()
    with torch.no_grad():got=PredictionOutputs(candidate)(x)
    for info,a in zip(m['outputs'],got):
        b=np.load(ART/'export'/f"{m['case']}_{info['name']}_reference.npy")
        assert np.allclose(a.numpy(),b,atol=2e-5,rtol=1e-4)
    checks['relocated_export_interface']='PASS'
    checks['training_state_rejection']='PASS'
    with (OUT/'parity_results.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==52
    assert all(r['status']=='NUMERICAL_PARITY_VERIFIED' for r in rows)
    ints=[r for r in rows if r['domain']=='INT8_LEGACY_RAW_ARITHMETIC']
    assert len(ints)==12 and all(float(r['max_abs_error'])==0 and float(r['exact_match_ratio'])==1 for r in ints)
    checks['parity_rows']=52;checks['csim_cases']=12
    evidence=json.loads((OUT/'representative_cases/tool_evidence.json').read_text())
    assert sum(x.get('synthesis_status')=='HLS_SYNTH_VERIFIED' for x in evidence)==1
    assert len(evidence)==22 and all(x['compile_returncode']==0 for x in evidence)
    checks['vendor_compile_cases']=22;checks['synthesis_cases']=1
    with (OUT/'OPERATOR_COMPATIBILITY_MATRIX.csv').open(newline='') as f:matrix=list(csv.DictReader(f))
    assert len(matrix)==54 and all(r['calibrated_detector_numerical_status']=='BLOCKED' for r in matrix)
    with (OUT/'V0_1_SHAPE_CATALOG.csv').open(newline='') as f:catalog=list(csv.DictReader(f))
    mb=[r for r in catalog if r['operator']=='MBConv']
    assert len(mb)==12 and all(r['residual']=='False' for r in mb)
    checks['catalog_rows']=54;checks['mbconv_shapes']=12
    checks['spec_sha256']=hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
    checks['status']='PASS'
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
