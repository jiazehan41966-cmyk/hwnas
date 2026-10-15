"""Independent P0 confirmation, with original D checkpoint admission first.

Never builds a test loader. Screening artifacts and historical controls are read-only.
"""
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import torch

import run as p0

sys.path.insert(0, str(p0.ROOT / 'tools/sonar_spec'))
from formal_runner import atomic_save, rng_state, restore_rng, train_epoch

PAIRS = (('DW5', 7), ('DCN', 1), ('FrequencyLite', 1), ('DirectionalDW', 5))
REPORT = p0.ROOT / 'reports/sonar_operator_pool_p0_confirmation'
OUTPUT = p0.ROOT / 'results/sonar_operator_pool_p0_confirmation'
AUDIT = REPORT / 'BASELINE_COMPARABILITY.json'


def frozen_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Refusing to overwrite frozen evidence: {path}')
    else:
        p0.dump_json(path, value)


def audit(old, device):
    cfg = p0.load_frozen_config()
    manifest = json.loads((old / 'reports/baseline_sonar_frozen/freeze_manifest.json').read_text())
    assert p0.sha256(p0.BASELINE_CONFIG) == manifest['config_sha256']
    sources = {}
    for rel, expected in manifest['source_sha256'].items():
        rel = rel.replace('\\', '/')
        actual = p0.sha256(p0.ROOT / rel)
        if actual != expected:
            raise ValueError(f'Frozen baseline source changed: {rel}')
        sources[rel] = actual
    train, valid = p0.data_loaders(cfg, 0)
    bindings = {}
    for split, loader in [('train', train), ('valid', valid)]:
        actual = p0.bind_dataset(loader)
        normalize = lambda rows: [{k: v.replace('\\', '/') if k in ('image', 'annotation') else v
                                  for k, v in r.items()} for r in rows]
        assert normalize(actual['entries']) == normalize(manifest['datasets'][split]['entries']), split
        bindings[split] = actual
    original_spec = importlib.util.spec_from_file_location(
        '_historical_candidate', old / 'src/hwnas_fpga/models/sonar_candidate/__init__.py')
    original = importlib.util.module_from_spec(original_spec)
    original_spec.loader.exec_module(original)
    space = p0.ROOT / 'configs/search_space_sonar_v0_1.yaml'
    assert p0.sha256(space) == p0.sha256(old / 'configs/search_space_sonar_v0_1.yaml')
    p0.set_run_seed(42)
    current = p0.build_d_baseline().to(device).eval()
    historical = original.build_sonar_candidate(space, p0.D_ARCHITECTURE).to(device).eval()
    historical.load_state_dict(current.state_dict(), strict=True)
    assert asdict(current.loss_config(cfg)) == asdict(historical.loss_config(cfg))
    assert asdict(current.loss_config(cfg)) == asdict(p0.loss_for(cfg, device).config)
    with torch.no_grad():
        x = torch.randn(1, 1, 640, 640, device=device)
        a, b = current(x)['predictions'], historical(x)['predictions']
    error = max(float((u-v).abs().max()) for u,v in zip(a,b))
    assert error == 0
    rows = [r for r in p0.read_csv(old / 'reports/phase2f_multiseed/multiseed_results.csv') if r['label'] == 'D']
    controls = []
    for row in rows:
        seed = int(row['seed'])
        folder = old / ('results/phase2e_independent/' + p0.D_ARCHITECTURE_ID if seed == 42 else
                        'results/phase2f_multiseed/' + p0.D_ARCHITECTURE_ID + f'/seed_{seed}')
        latest = torch.load(folder/'latest.pt', map_location='cpu', weights_only=False)
        best = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        assert latest['epoch'] == 80 and latest['seed'] == seed
        assert latest['architecture_id'] == p0.D_ARCHITECTURE_ID
        assert [r['epoch'] for r in latest['history']] == list(range(1,81))
        assert p0.sha256(folder/'epoch_log.json') == row['history_sha256']
        # Phase2F's seed42 field inherited the Phase2E budget80 SHA, not best.pt.
        referenced = folder/('budget_80.pt' if seed == 42 else 'best.pt')
        assert p0.sha256(referenced) == row['best_checkpoint_sha256']
        assert best['epoch'] == int(row['best_epoch'])
        assert latest.get('test_evaluated') is False
        current.load_state_dict(best['model_state_dict'], strict=True)
        metrics, _ = p0.evaluate(current, valid, p0.loss_for(cfg, device), cfg, device)
        for k in ('mAP50_95','AP50','AP75','Precision','Recall'):
            assert abs(metrics[k]-float(row[k])) <= 1e-7, (seed,k,metrics[k],row[k])
        controls.append(dict(seed=seed, best_metrics=metrics, best_epoch=best['epoch'],
            endpoint_metrics=latest['history'][-1]['valid'], best_checkpoint=str(folder/'best.pt'),
            best_sha256=p0.sha256(folder/'best.pt'), latest_sha256=p0.sha256(folder/'latest.pt'),
            history_sha256=p0.sha256(folder/'epoch_log.json'), original_binding=latest['binding'],
            historical_sha_field_points_to=referenced.name))
        print('CONTROL VERIFIED',seed,metrics['mAP50_95'],flush=True)
    assert {r['seed'] for r in controls} == {42,43,44}
    frozen_json(AUDIT, dict(status='PASS', baseline_sha256=p0.sha256(p0.BASELINE_CONFIG),
        search_space_sha256=p0.sha256(space), source_sha256=sources, datasets=bindings,
        old_current_candidate_max_error=error, controls=controls, test_evaluated=False,
        caveats=['Seed42 historical best SHA field actually bound budget_80.pt; original best.pt independently verified.',
                 'P0 initialization and profiling consume RNG differently from historical controls; shared seed does not imply identical initial tensors or augmentation draws.',
                 'Matched effective optimizer/loss/sampler/augmentation/evaluator, not bitwise identical training trajectories.']))
    protocol = dict(pairs=[dict(operator=o,stage=s) for o,s in PAIRS], new_seeds=[43,44],
        epochs=80,from_scratch=True,baseline_audit_sha256=p0.sha256(AUDIT),
        source_sha256={str(p.relative_to(p0.ROOT)): p0.sha256(p) for p in
            [Path(__file__), Path(p0.__file__), p0.ROOT/'src/hwnas_fpga/models/sonar_operator_pool_p0.py',
             p0.ROOT/'tools/sonar_spec/formal_runner.py',p0.ROOT/'src/hwnas_fpga/models/sonar_candidate/__init__.py']},
        checkpoint_policy='earliest best valid within 80; endpoint also recorded',
        survival_rule='mean best delta > 0; at least 2/3 best deltas > 0; mean endpoint delta > 0; at least 2/3 endpoint deltas > 0',
        limitations='n=3; selected seed42 screening winners; no significance claim',test_evaluated=False)
    frozen_json(REPORT/'CONFIRMATION_MANIFEST.json',protocol)
    (REPORT/'BASELINE_COMPARABILITY.md').write_text(
        '# D baseline comparability\n\nPASS: three original controls verified against checkpoint histories, SHA256, and fresh full-valid evaluation. '
        'Frozen source files and train/valid content verified. Historical/current D forward is exact.\n\n'
        'Seed42 historical best_checkpoint_sha256 actually refers to budget_80.pt. Original best.pt is epoch64; '
        'metrics agree. Both hashes are preserved in BASELINE_COMPARABILITY.json. No historical CSV was overwritten.\n\n'
        'RNG consumption differs between runners; matched protocol does not imply identical initialization/augmentation draws. '
        'P0 MACs exclude interpolation, pooling and elementwise operations; not deployment latency.\n\nTest evaluated: false.\n')


def train(operator, stage, seed, device, smoke=False):
    if (operator,stage) not in PAIRS or seed not in (43,44):
        raise ValueError('Outside preregistered eight confirmation runs')
    if device != 'cuda' or not torch.cuda.is_available():
        raise ValueError('CUDA required')
    protocol_path = REPORT/'CONFIRMATION_MANIFEST.json'
    protocol = json.loads(protocol_path.read_text())
    admitted = json.loads(AUDIT.read_text())
    assert admitted['status']=='PASS' and p0.sha256(AUDIT)==protocol['baseline_audit_sha256']
    for rel,sha in {**admitted['source_sha256'],**protocol['source_sha256']}.items():
        assert p0.sha256(p0.ROOT/rel)==sha, rel
    assert p0.sha256(p0.BASELINE_CONFIG)==admitted['baseline_sha256']
    cfg=p0.load_frozen_config()
    p0.set_run_seed(seed)
    model=p0.build_p0_candidate(operator,stage).to(device)
    # Profiling must not alter train RNG or BN state.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        info=p0.model_info_with_p0_ops(model)
    train_loader,valid_loader=p0.data_loaders(cfg,0,seed)
    for split,loader in [('train',train_loader),('valid',valid_loader)]:
        assert p0.bind_dataset(loader)==admitted['datasets'][split]
    opt=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.0001)
    aid=p0.p0_architecture_id(operator,stage)
    folder=OUTPUT/('smoke' if smoke else 'formal')/aid/f'seed_{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    binding=dict(manifest_sha256=p0.sha256(protocol_path),architecture_id=aid,seed=seed,smoke=smoke)
    history=[]
    initial=p0.state_hash(model)
    if (folder/'latest.pt').exists():
        state=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
        assert state['binding']==binding
        model.load_state_dict(state['model_state_dict']);opt.load_state_dict(state['optimizer_state_dict'])
        restore_rng(state['rng'],train_loader)
        history=state['history'];initial=state['initial_state_sha256']
    limit=1 if smoke else 80
    for epoch in range(len(history),limit):
        start=time.perf_counter()
        batches=[next(iter(train_loader))] if smoke else train_loader
        row=train_epoch(model,opt,batches,cfg,device)
        metrics,_=p0.evaluate(model,valid_loader,p0.loss_for(cfg,device),cfg,device,max_batches=1 if smoke else None)
        row.update(epoch=epoch+1,valid=metrics,wall_seconds=time.perf_counter()-start)
        improved=metrics['mAP50_95']>max((r['valid']['mAP50_95'] for r in history),default=-1)
        history.append(row)
        state=dict(binding=binding,epoch=epoch+1,history=history,model_state_dict=model.state_dict(),
            optimizer_state_dict=opt.state_dict(),rng=rng_state(train_loader),initial_state_sha256=initial,
            initialization='independent_from_scratch',model_info=info,test_evaluated=False)
        if improved: atomic_save(folder/'best.pt',state)
        atomic_save(folder/'latest.pt',state)
        p0.dump_json(folder/'history.json',history)
        print(aid,seed,epoch+1,metrics['mAP50_95'],flush=True)
    best=max(history,key=lambda r:r['valid']['mAP50_95'])
    p0.dump_json(folder/'status.json',dict(status='SMOKE' if smoke else 'COMPLETE',binding=binding,
        epochs=len(history),best_epoch=best['epoch'],best_metrics=best['valid'],endpoint_metrics=history[-1]['valid'],
        model_info=info,history_sha256=p0.sha256(folder/'history.json'),best_sha256=p0.sha256(folder/'best.pt'),
        latest_sha256=p0.sha256(folder/'latest.pt'),test_evaluated=False))


def aggregate():
    audit_data=json.loads(AUDIT.read_text())
    controls={r['seed']:r for r in audit_data['controls']}
    rows=[]
    for operator,stage in PAIRS:
        aid=p0.p0_architecture_id(operator,stage)
        for seed in (42,43,44):
            path=p0.RESULTS_DIR/aid/'status.json' if seed==42 else OUTPUT/'formal'/aid/f'seed_{seed}'/'status.json'
            if not path.exists(): continue
            st=json.loads(path.read_text())
            if st['status']!='COMPLETE': continue
            best=st['best_valid'] if seed==42 else st['best_metrics']
            end=st['final_valid'] if seed==42 else st['endpoint_metrics']
            ref=controls[seed]
            rows.append(dict(operator=operator,stage=stage,seed=seed,best_epoch=st['best_epoch'],
                mAP50_95=best['mAP50_95'],endpoint=end['mAP50_95'],
                peak_endpoint_gap=best['mAP50_95']-end['mAP50_95'],
                delta=best['mAP50_95']-ref['best_metrics']['mAP50_95'],
                endpoint_delta=end['mAP50_95']-ref['endpoint_metrics']['mAP50_95'],test_evaluated=False))
    p0.write_csv(REPORT/'MULTISEED_RESULTS.csv',rows)
    summary=[]
    for operator,stage in PAIRS:
        group=[r for r in rows if r['operator']==operator]
        if len(group)!=3:continue
        deltas=[r['delta'] for r in group];ends=[r['endpoint_delta'] for r in group]
        survives=statistics.mean(deltas)>0 and sum(d>0 for d in deltas)>=2 and statistics.mean(ends)>0 and sum(d>0 for d in ends)>=2
        summary.append(dict(operator=operator,stage=stage,mean_delta=statistics.mean(deltas),
            sample_std_delta=statistics.stdev(deltas),positive_seeds=sum(d>0 for d in deltas),
            endpoint_mean_delta=statistics.mean(ends),survives_for_transfer_review=survives,n=3))
    p0.write_csv(REPORT/'STABILITY_SUMMARY.csv',summary)
    print('completed candidate seed rows',len(rows),'/12; new runs',sum(r['seed']!=42 for r in rows),'/8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['audit','train','aggregate'])
    parser.add_argument('--historical-root',type=Path)
    parser.add_argument('--operator');parser.add_argument('--stage',type=int);parser.add_argument('--seed',type=int)
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda']);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    if args.mode=='audit':audit(args.historical_root,args.device)
    elif args.mode=='train':train(args.operator,args.stage,args.seed,args.device,args.smoke)
    else:aggregate()
