"""Read-only checkpoint diagnosis: exposure, gradients and train-only BN budgets."""
import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from formal_protocol import (ROOT, load_protocol, verify_baseline, load_witness,
    source_binding, sha, dump, frozen, calibration_manifest)
from formal_runner import write_csv
from candidate_diagnostics import assignment_diagnostics
from hwnas_fpga.models.sonar_supernet import (build_sonar_supernet, BalancedArchitectureSampler,
    enumerate_architectures, architecture_id, calibrate_subnet_bn)
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss


def exposure(state, model, out):
    sampler=BalancedArchitectureSampler(42); ids=Counter(); modules=Counter(); total=state['global_step']
    for epoch in state['history']:
        macros=Counter(); kernels=Counter()
        for _ in range(epoch['batches']):
            a=sampler.sample(); ids[architecture_id(a)]+=1; macros[a['output_combination']]+=1
            for s,k in a['kernels'].items(): kernels[f'{s}_k{k}']+=1
        assert dict(macros)==epoch['combination_counts']
        assert dict(kernels)==epoch['kernel_counts']
    assert sampler.state_dict()==state['architecture_sampler_rng']
    assert dict(sampler.statistics)==state['architecture_sampling_statistics']
    assert sum(ids.values())==total
    stats=sampler.statistics
    levels={'P2':total-stats['P3_P4_P5'],'P3':total,'P4':total,'P5':total-stats['P2_P3_P4']}
    def expected(name):
        parts=name.split('.')
        if parts[0] in ('laterals','outputs','heads'):
            return levels[parts[1]],1. if parts[1] in ('P3','P4') else 2/3
        if name.startswith('backbone.stages.'):
            stage=int(parts[2]); prob=1. if stage<=5 else 2/3
            if '.branches.' in name:
                k=int(name.split('.branches.k')[1][0]); return stats[f'stage{stage}_k{k}'],prob/2
            return (total if stage<=5 else levels['P5']),prob
        return total,1.
    rows=[]
    for combo,n in [('P3_P4_P5',64),('P2_P3_P4',16),('P2_P3_P4_P5',64)]:
        rows.append(dict(kind='macro',name=combo,theoretical_probability=1/3,expected_iterations=total/3,
            actual_iterations=stats[combo],evidence='recorded_epoch_counts_and_checkpoint'))
    for a in enumerate_architectures():
        n=16 if a['output_combination']=='P2_P3_P4' else 64; aid=architecture_id(a)
        rows.append(dict(kind='architecture',name=aid,theoretical_probability=1/(3*n),expected_iterations=total/(3*n),
            actual_iterations=ids[aid],evidence='replayed_seed42_verified_each_epoch_and_final_RNG'))
    param_ids=state['optimizer_state_dict']['param_groups'][0]['params']
    assert len(param_ids)==len(list(model.named_parameters()))
    for (name,param),idx in zip(model.named_parameters(),param_ids):
        actual=int(state['optimizer_state_dict']['state'][idx]['step']);count,prob=expected(name)
        assert actual==count,(name,actual,count)
        rows.append(dict(kind='parameter',name=name,theoretical_probability=prob,expected_iterations=total*prob,
            actual_iterations=actual,evidence='AdamW_step_verified_against_replayed_active_path'))
    for name,module in model.named_modules():
        if isinstance(module,nn.BatchNorm2d):
            count,prob=expected(name);actual=int(module.num_batches_tracked)
            assert actual==count,(name,actual,count)
            rows.append(dict(kind='BN',name=name,theoretical_probability=prob,expected_iterations=total*prob,
                actual_iterations=actual,evidence='num_batches_tracked_verified_against_active_path'))
    write_csv(out/'SAMPLING_EXPOSURE_AUDIT.csv',rows)
    dump(out/'exposure_verification.json',dict(total_iterations=total,macros={k:stats[k] for k in ('P3_P4_P5','P2_P3_P4','P2_P3_P4_P5')},
        sampling_statistics=dict(stats),level_iterations=levels,unique_architectures=len(ids),
        min_architecture_exposure=min(ids.values()),max_architecture_exposure=max(ids.values()),
        replay_epoch_counts_equal=True,final_rng_equal=True,optimizer_steps_equal=True,bn_counters_equal=True))


def gradient_probe(model,state,loader,b,out,device):
    arches=list(enumerate_architectures()); selected=[]
    for combo in ('P3_P4_P5','P2_P3_P4','P2_P3_P4_P5'):
        group=[a for a in arches if a['output_combination']==combo]
        mix={'output_combination':combo,'kernels':{s:(3 if int(s[5:])%2 else 5) for s in group[0]['kernels']}}
        selected.extend([group[0],group[-1],mix])
    batches=list(itertools.islice(loader,2)); rows=[]; assignments=[]; cosines=[]
    original={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    for bi,(images,targets) in enumerate(batches):
        vectors={}
        for a in selected:
            model.load_state_dict(state['model_state_dict']);model.set_active_subnet(a);model.train();model.zero_grad(set_to_none=True)
            output=model(images.to(device));criterion=DenseAnchorDetectionLoss(model.loss_config(b)).to(device)
            loss,parts=criterion(output,targets);loss.backward()
            assert torch.isfinite(loss)
            aid=architecture_id(a); row=dict(batch=bi,architecture_id=aid,output_combination=a['output_combination'],**parts)
            diag=assignment_diagnostics(output['predictions'],targets,criterion.config,model.levels)
            for d in diag:assignments.append(dict(batch=bi,architecture_id=aid,**d))
            for name in ('positive','negative','ignore'): row[name+'_anchors']=sum(d[name] for d in diag)
            groups={'backbone':lambda n:n.startswith('backbone.'),'neck':lambda n:n.startswith(('laterals.','outputs.')),
                'head':lambda n:n.startswith('heads.'),'P2_specific':lambda n:'.P2.' in n,'P5_specific':lambda n:'.P5.' in n}
            for name,condition in groups.items():
                ps=[p for n,p in model.named_parameters() if condition(n) and p.grad is not None]
                assert all(torch.isfinite(p.grad).all() for p in ps)
                norm=math.sqrt(sum(float(p.grad.double().square().sum()) for p in ps));elements=sum(p.numel() for p in ps)
                row[name+'_gradient_L2']=norm;row[name+'_gradient_RMS']=norm/math.sqrt(elements) if elements else 0
            # Compare gradients on parameters shared by every macro/kernel path; excludes independent DW branches.
            vectors[aid]=torch.cat([p.grad.detach().cpu().flatten() for n,p in model.named_parameters()
                if n.startswith('backbone.') and '.branches.' not in n and not n.startswith(('backbone.stages.6.','backbone.stages.7.'))])
            rows.append(row)
        for a,c in itertools.combinations(selected,2):
            ai,ci=architecture_id(a),architecture_id(c)
            cosine=float(torch.nn.functional.cosine_similarity(vectors[ai],vectors[ci],dim=0))
            cosines.append(dict(batch=bi,architecture_a=ai,architecture_b=ci,same_macro=a['output_combination']==c['output_combination'],shared_backbone_gradient_cosine=cosine))
    assert all(torch.equal(p.detach().cpu(),original[n]) for n,p in model.named_parameters())
    model.load_state_dict(state['model_state_dict']);model.eval();model.zero_grad(set_to_none=True)
    write_csv(out/'GRADIENT_SCALE_DIAGNOSTICS.csv',rows);write_csv(out/'gradient_assignment_per_level.csv',assignments)
    write_csv(out/'shared_gradient_cosines.csv',cosines)
    dump(out/'gradient_protocol.json',dict(architectures=selected,batches=2,images=8,train_mode=True,augmentation=False,
        identical_batch_for_all_architectures=True,weights_unchanged=True,original_BN_restored=True,optimizer_steps=0,
        interpretation='Two fixed batches diagnose scale; they cannot establish persistent causal interference.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'results/supernet_seed42/latest.pt')
    p.add_argument('--phase2c',type=Path,default=ROOT/'reports/phase2c_supernet')
    p.add_argument('--output',type=Path,default=ROOT/'reports/phase2d_ranking_bias')
    p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--exposure-only',action='store_true',help='Recheck saved optimizer and sampling counts without forward/evaluation')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);frozen.seed_all(42)
    protocol=ROOT/'configs/supernet_sonar_v0_1_formal.yaml';config,b=load_protocol(protocol);protection=verify_baseline(b)
    witness=load_witness(args.phase2c/'ranking_witness_manifest.json',protocol)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert state['binding']['source_sha256']==source_binding(protocol)
    assert sha(args.checkpoint)==witness['supernet_checkpoint_sha256']
    model=build_sonar_supernet().to(args.device);model.load_state_dict(state['model_state_dict']);model.eval()
    exposure(state,model,out)
    if args.exposure_only:
        dump(out/'exposure_source.json',dict(source_sha256=sha(Path(__file__)),checkpoint_sha256=sha(args.checkpoint)))
        return
    loader,manifest=calibration_manifest(b,args.phase2c)
    assert loader.dataset.split=='train' and len(loader.dataset)==330 and loader.num_workers==0
    entries=[]
    for image in loader.dataset.samples:
        label=loader.dataset.labels_dir/(image.stem+'.txt')
        entries.append(dict(image=str(image.relative_to(loader.dataset.root)),image_sha256=sha(image),annotation_sha256=sha(label)))
    dump(out/'bn_prefix_manifest.json',dict(seed=42,split='train',augmentation=False,workers=0,batch_size=4,
        entries=entries,budgets=[8,16,32,64,len(loader)],formal_first64_sha256=manifest['samples_sha256']))
    gradient_probe(model,state,loader,b,out,args.device);print('Exposure and gradient diagnosis PASS',flush=True)
    valid=frozen.make_loader(b,ROOT/b['dataset']['data_dir'],'valid',42,0)
    rows=[];budgets=[8,16,32,64,len(loader)]
    for budget in budgets:
        for item in witness['witnesses']:
            start=time.perf_counter();candidate=calibrate_subnet_bn(model,item['architecture'],loader,budget)
            metrics,_=frozen.evaluate(candidate,valid,DenseAnchorDetectionLoss(candidate.loss_config(b)).to(args.device),b,args.device)
            rows.append(dict(batches=budget,images=min(budget*4,len(loader.dataset)),architecture_id=item['architecture_id'],
                **{k:metrics[k] for k in ('AP50','AP75','mAP50_95','Precision','Recall')},runtime_seconds=time.perf_counter()-start))
            write_csv(out/'BN_CALIBRATION_SENSITIVITY.csv',rows)
            print('BN',budget,item['architecture_id'],metrics['mAP50_95'],flush=True)
    checkpoints=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in args.checkpoint.parent.glob('*.pt')]
    dump(out/'diagnostic_provenance.json',dict(status='COMPLETE',supernet_checkpoint_sha256=sha(args.checkpoint),
        phase2c_witness_sha256=sha(args.phase2c/'ranking_witness_manifest.json'),source_sha256=sha(Path(__file__)),
        checkpoints_available=checkpoints,protection=protection,test_evaluated=False,optimizer_steps=0,
        budgets=budgets,interpretation='Diagnostic only. No formal protocol or witness selection changes.'))


if __name__=='__main__':main()
