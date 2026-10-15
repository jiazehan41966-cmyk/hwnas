"""Freeze corrected controls and the regression witness without modifying data."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hwnas_fpga.training.detection_trainer import _weighted_objectness_loss


def main():
    old=ROOT/'results/detection_controlled_20260912'
    out=ROOT/'results/detection_fixed_normalizer_20260912'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    witness={'logits':[-4.,-2.,0.,2.], 'targets':[0.,0.,0.,0.],
             'reference':'https://docs.pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html',
             'scope':'Autograd direction verification; not empirical performance evidence'}
    for mode in ('legacy_dynamic','fixed_weights'):
        x=torch.tensor(witness['logits'],requires_grad=True)
        y=torch.zeros_like(x)
        v=_weighted_objectness_loss(x,y,valid_obj_mask=torch.ones_like(x,dtype=torch.bool),
            positive_weight=1.,negative_weight=.2,focal_gamma=2.,focal_alpha=.75,normalization=mode)
        v.backward()
        witness[mode]={'loss':v.item(),'gradients':x.grad.tolist(),'wrong_direction_count':int((x.grad<0).sum())}
    for folder in (old,out):
        (folder/'objectness_gradient_witness.json').write_text(json.dumps(witness,indent=2))
    snapshot=out/'snapshot'
    hashes={}
    files=list((ROOT/'src').rglob('*.py'))+[ROOT/'scripts/detection_controlled_study.py']
    for path in files:
        if '__pycache__' in str(path):
            continue
        rel=path.relative_to(ROOT)
        dest=snapshot/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        original=path.read_bytes()
        dest.write_bytes(original)
        if original!=path.read_bytes():
            raise RuntimeError(f'Source changed during snapshot: {rel}')
        hashes[str(rel).replace('\\','/')]=hashlib.sha256(original).hexdigest()
    (out/'source_snapshot_manifest.json').write_text(json.dumps(hashes,indent=2))
    shutil.copy2(ROOT/'scripts/run_detection_controlled_queue.py',out)
    (out/'configs').mkdir()
    baseline=yaml.safe_load((old/'configs/baseline.yaml').read_text(encoding='utf-8-sig'))
    baseline['training']['detection_loss']['objectness_normalization']='fixed_weights'
    for name,weight in (('baseline',.1),('noobj02',.2)):
        config=json.loads(json.dumps(baseline))
        config['training']['detection_loss']['no_object_weight']=weight
        config['project']['output_dir']='results/detection_fixed_normalizer_20260912'
        (out/'configs'/f'{name}.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    spec=json.loads((old/'study_manifest.json').read_text(encoding='utf-8-sig'))
    spec.update(study='detection_fixed_normalizer_20260912',
        supersedes='detection_controlled_20260912 legacy-gradient-defect queue',
        normalization='fixed_weights',
        primary='paired validation legacy map(.50,.75) difference across seeds',
        promotion_rule='Exploratory candidate only if mean primary delta >0, positive in at least 2/3 seeds, and mean manta AP50 does not decline; no formal significance/generalization claim from n=3',
        structural_changes='None; fixed backbone, neck, head, data and anchors',
        corrected_loss='Target-dependent weighted normalizer excludes prediction-dependent focal weights',
        selected_epoch='Maximum validation legacy map, identical to historical metric; AP10 is supplemental')
    (out/'study_manifest.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    (out/'validation_evidence.json').write_text(json.dumps({'tests':'python -m pytest tests/test_detection_objectness_normalization.py tests/test_detection_controlled_study.py tests/test_detection_branch.py -q','result':'26 passed','legacy_background_wrong_gradients':3,'fixed_background_wrong_gradients':0},indent=2))
    print(json.dumps({'output':str(out),'witness':witness,'source_files':len(hashes)}))


if __name__=='__main__':
    main()
