"""Phase 2A: five fixed lightweight candidate checks, train/valid only."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools/sonar_analysis'), str(ROOT/'scripts'), str(Path(__file__).parent)]
import torch
import yaml
import baseline as frozen
from candidate_diagnostics import assignment_diagnostics, topk_diagnostics
from hwnas_fpga.models.sonar_candidate import build_sonar_candidate, load_spec
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss


def architecture(name, mode='k3'):
    stages = [1,2,4,5] + ([] if name=='P2_P3_P4' else [6,7])
    return dict(output_combination=name, kernels={f'stage{s}': (5 if mode=='k5' or mode=='mixed' and i%2 else 3) for i,s in enumerate(stages)})


def protection(baseline_path):
    manifest = json.loads((ROOT/'reports/baseline_sonar_frozen/freeze_manifest.json').read_text())
    assert frozen.source_hashes() == manifest['source_sha256'], 'Frozen source manifest changed'
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == manifest['config_sha256']
    return dict(source_files=len(manifest['source_sha256']), source_manifest_equal=True, config_equal=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--space', type=Path, default=ROOT/'configs/search_space_sonar_v0_1.yaml')
    p.add_argument('--baseline', type=Path, default=ROOT/'configs/baseline_sonar_frozen.yaml')
    p.add_argument('--output', type=Path, default=ROOT/'reports/phase2a_candidate')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    config = yaml.safe_load(args.baseline.read_text(encoding='utf-8'))
    spec = load_spec(args.space)
    before = protection(args.baseline)
    frozen.seed_all(42)
    train_loader = frozen.make_loader(config, frozen.resolve(config['dataset']['data_dir']), 'train',42,0,train=True)
    valid_loader = frozen.make_loader(config, frozen.resolve(config['dataset']['data_dir']), 'valid',42,0)
    images, targets = next(iter(train_loader))
    base, _ = frozen.make_model(config)
    candidate = build_sonar_candidate(spec, architecture('P3_P4_P5'))
    candidate.load_state_dict(base.state_dict(), strict=True)
    base.eval(); candidate.eval()
    fixed = torch.randn(1,1,640,640)
    with torch.no_grad():
        a, b = base(fixed)['predictions'], candidate(fixed)['predictions']
        errors = [dict(max_abs_error=float((u-v).abs().max()), mean_abs_error=float((u-v).abs().mean())) for u,v in zip(a,b)]
    assert all(e['max_abs_error']==0 for e in errors)
    report = dict(status='RUNNING', seed=42, device='cpu', batch_size=len(images),
        baseline_protection=before, equivalence=dict(state_dict_mapping='identity_strict_all_keys', levels=errors,
        max_abs_error=max(e['max_abs_error'] for e in errors), mean_abs_error=0.),
        train_paths=[t['path'] for t in targets], train_targets=[dict(boxes=t['boxes'].tolist(),labels=t['labels'].tolist()) for t in targets],
        protocol=config['evaluation_protocol'], cases=[], test_evaluated=False,
        diagnostic_scope='one identical augmented WeightedRandomSampler train batch; seed42 random initialization, before optimizer step; not performance evidence')
    del base, candidate, a, b
    expected = {'P3_P4_P5':1961715200,'P2_P3_P4':6081996800,'P2_P3_P4_P5':6319040000}
    cases = [(n,'k3') for n in expected] + [('P2_P3_P4_P5','k5'),('P2_P3_P4_P5','mixed')]
    for name, mode in cases:
        frozen.seed_all(42)
        model = build_sonar_candidate(spec, architecture(name, mode))
        info = frozen.get_model_info(model, (1,1,640,640))
        if mode=='k3': assert info['estimated_macs']==expected[name]
        executed=[]
        hooks=[stage[0].register_forward_hook(lambda m,i,o,s=s: executed.append(s)) for s,stage in enumerate(model.backbone.stages)]
        model.train()
        out = model(images)
        for h in hooks: h.remove()
        assert executed == list(range(6 if name=='P2_P3_P4' else 8))
        frozen.finite_outputs(out)
        for pred, stride in zip(out['predictions'], model.feature_strides):
            assert list(pred.shape)==[len(images),21,640//stride,640//stride]
        criterion = DenseAnchorDetectionLoss(model.loss_config(config))
        loss, parts = criterion(out, targets)
        assert all(math.isfinite(v) for v in parts.values())
        loss.backward()
        gradients = frozen.gradient_summary(model)
        detached = [t.detach() for t in out['predictions']]
        assignment = assignment_diagnostics(detached, targets, criterion.config, model.levels)
        topk = topk_diagnostics(detached, criterion.config, model.levels, config['evaluation_protocol'])
        torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay']).step()
        validation, _ = frozen.evaluate(model,valid_loader,criterion,config,torch.device('cpu'),max_batches=1)
        row = dict(name=name, mode=mode, architecture=model.candidate_architecture, executed_stages=executed,
            feature_shapes=[list(t.shape) for t in out['backbone_features']], head_shapes=[list(t.shape) for t in detached],
            anchors=model.anchor_sizes, params=info['total_params'], macs=info['estimated_macs'], flops_2macs=2*info['estimated_macs'],
            losses=parts, gradients=gradients, assignment=assignment, topk=topk, validation=validation)
        report['cases'].append(row)
        frozen.dump(args.output/'sanity_results.json', report)
        print(f'PASS {name} {mode}: Params={row["params"]}, MACs={row["macs"]}', flush=True)
        del model, out, loss, criterion, detached
        gc.collect()
    report['baseline_protection_after'] = protection(args.baseline)
    report['status']='PASS'
    report['source_sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [Path(__file__),ROOT/'tools/sonar_spec/candidate_diagnostics.py',ROOT/'src/hwnas_fpga/models/sonar_candidate/__init__.py',args.space.resolve()]}
    frozen.dump(args.output/'sanity_results.json', report)
    print('Phase 2A sanity PASS; test remains sealed.', flush=True)


if __name__=='__main__': main()
