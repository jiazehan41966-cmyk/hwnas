"""Unified bounded V0.1 independent search -> PTQ -> exact cost -> Pareto.

plan is read-only except its isolated report; freeze binds sources/data/costs;
run executes real workers. No dummy scores and no automatic unknown-cost fill.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/fpga_h4'))
from candidate_graphs import architectures, candidate_graph
from cost_api import CostDatabase, atomic_usage
from joint_search import run_joint_search
from quant_contracts import PROFILES
from pareto import pareto_max_min, pareto_accuracy_resources

CONFIG=ROOT/'configs/search/sonar_joint_v1.yaml'
SPACE=ROOT/'configs/search_space_sonar_v0_1.yaml'
PLATFORM=dict(part='xczu7ev-ffvc1156-2-e',tool_version='2022.2',target_clock_ns=5,
    board_reference='ZCU104',latency_fidelity='HLS_CSYNTH_SERIAL_ESTIMATE',
    board_latency='NOT_CLAIMED',deployment='NOT_CLAIMED',int8_substitution=False)
COST_FILE_PREFIXES=('artifacts/fpga_h3/','artifacts/sonar_joint_cost_v1/','reports/fpga_h3_cost_db/',
    'artifacts/fpga_h2/','reports/fpga_h2_contract/')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def read(path):return json.loads(Path(path).read_text())
def subset(files, prefixes):
    return {k:files[k] for k in files if k.startswith(prefixes)}

def load_config(path):
    cfg=yaml.safe_load(path.read_text())
    expected=yaml.safe_load(CONFIG.read_text())
    if cfg!=expected:raise ValueError('This version accepts only its frozen protocol; version changes require a new config/runner')
    if cfg['test_policy']!='sealed':raise ValueError('Test split is sealed')
    return cfg

def data_binding(cfg):
    baseline=yaml.safe_load((ROOT/cfg['baseline']).read_text())
    data=ROOT/baseline['dataset']['data_dir'];rows={}
    # Read train/valid only. Do not scan test, even for hashing.
    for split in ('train','valid'):
        images=sorted(p for p in (data/split/'images').iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff'))
        if not images:raise ValueError('Empty data split '+split)
        entries=[]
        for image in images:
            label=data/split/'labels'/(image.stem+'.txt')
            entries.append(dict(image=image.relative_to(data).as_posix(),image_sha256=sha(image),
                annotation=label.relative_to(data).as_posix(),annotation_sha256=sha(label)))
        rows[split]=entries
    frozen=read(ROOT/'reports/baseline_sonar_frozen/freeze_manifest.json')
    def normalized(entries):return sorted((e['image'].replace('\\','/'),e['image_sha256'],e['annotation'].replace('\\','/'),e['annotation_sha256']) for e in entries)
    for split in rows:
        if normalized(rows[split])!=normalized(frozen['datasets'][split]['entries']):raise ValueError('Frozen dataset mismatch '+split)
    return rows

def inputs(config_path):
    paths={config_path,ROOT/'configs/baseline_sonar_frozen.yaml',ROOT/'configs/search_space_sonar_v0_1.yaml',
        ROOT/'reports/baseline_sonar_frozen/freeze_manifest.json'}
    for directory in ('tools/sonar_search','tools/fpga_h4','tools/sonar_spec','tools/sonar_analysis','tools/phase5a_quantization','src/hwnas_fpga'):
        paths.update((ROOT/directory).rglob('*.py'))
    paths.add(ROOT/'scripts/detection_controlled_study.py')
    for rel in ('artifacts/fpga_h3/20260918_batch01/clean.json','artifacts/fpga_h3/20260918_batch01/network_usage.json',
                'artifacts/sonar_joint_cost_v1/clean.json','reports/fpga_h3_cost_db/H3_DATABASE_MANIFEST.json'):
        p=ROOT/rel
        if p.is_file():paths.add(p)
    for directory in ('reports/fpga_h2_contract','artifacts/fpga_h2/20260916_v1'):
        paths.update((ROOT/directory).glob('*.json'))
    for letter in ('A','B','C'):paths.add(ROOT/f'artifacts/fpga_h2/20260916_v1/graph_{letter}/graph.json')
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}

def coverage():
    db=CostDatabase();rows=[]
    for aid in architectures():
        for q in PROFILES:
            r=db.estimate_cost(aid,q,'reference_min_latency')
            rows.append(dict(architecture_id=aid,profile=q,status=r['status'],missing=r.get('missing',[]),
                cost={k:r.get(k) for k in ('Latency','DSP','LUT','BRAM','measurement_ids')},
                fidelity='HLS_CSYNTH_SERIAL_ESTIMATE',source='CSYNTH',
                validity='VALID' if r['status']=='LOOKED_UP' and r.get('measurement_valid') else r['status'],
                measurement_valid=r.get('measurement_valid'),timing_eligible=r.get('timing_eligible'),
                deployment=r.get('deployment','NOT_CLAIMED'),board_measured=False))
    return dict(architectures=144,triples=len(rows),looked_up=sum(r['status']=='LOOKED_UP' for r in rows),
        missing=sum(r['status']!='LOOKED_UP' for r in rows),rows=rows,deployment='NOT_CLAIMED',
        latency_fidelity='HLS_CSYNTH_SERIAL_ESTIMATE',board_latency='NOT_CLAIMED')

def cost_snapshot(cov):
    rows=[]
    for r in cov['rows']:
        ids=r['cost'].get('measurement_ids') or ''
        profile=r['profile']
        if r['status']!='LOOKED_UP' or r.get('board_measured'):
            raise ValueError('Snapshot contains non-looked-up or board-claimed row '+r['architecture_id']+' '+profile)
        if profile=='HEAD_W8A16' and '_w8a16_' not in ids:
            raise ValueError('HEAD_W8A16 missing measured mixed-precision ids: '+r['architecture_id'])
        if profile=='HEAD_W16A16' and '_w16a16_' not in ids:
            raise ValueError('HEAD_W16A16 missing measured mixed-precision ids: '+r['architecture_id'])
        if profile=='W8A8' and ('_w8a16_' in ids or '_w16a16_' in ids):
            raise ValueError('W8A8 row mixed with mixed-precision measurement ids: '+r['architecture_id'])
        rows.append(dict(architecture_id=r['architecture_id'],precision=profile,mapping='reference_min_latency',
            status=r['status'],Latency=r['cost']['Latency'],DSP=r['cost']['DSP'],LUT=r['cost']['LUT'],BRAM=r['cost']['BRAM'],
            measurement_ids=ids,fidelity='HLS_CSYNTH_SERIAL_ESTIMATE',source='CSYNTH',validity=r.get('validity','VALID'),
            measurement_valid=r.get('measurement_valid'),timing_eligible=r.get('timing_eligible'),
            board_measured=False,deployment='NOT_CLAIMED',int8_substitution=False))
    return dict(architectures=cov['architectures'],triples=cov['triples'],looked_up=cov['looked_up'],missing=cov['missing'],
        profiles=list(PROFILES),platform=PLATFORM,fidelity='HLS_CSYNTH_SERIAL_ESTIMATE',
        board_latency='NOT_CLAIMED',deployment='NOT_CLAIMED',int8_substitution=False,rows=rows)

def named_hashes(cfg,files,dataset,cov,snapshot):
    space_rel=SPACE.relative_to(ROOT).as_posix()
    return dict(
        config_hash=digest(cfg),
        code_hash=digest(subset(files,('tools/','src/','scripts/'))),
        data_hash=digest(dataset),
        cost_db_hash=digest(dict(files=subset(files,COST_FILE_PREFIXES),
            coverage={k:cov[k] for k in ('architectures','triples','looked_up','missing')},
            snapshot_digest=digest(snapshot),platform=PLATFORM)),
        candidate_space_hash=digest(dict(space_file=files[space_rel],architecture_ids=sorted(architectures()),
            count=len(architectures()))))

def freeze(config_path,out):
    cfg=load_config(config_path)
    if (out/'freeze.json').exists():raise FileExistsError('Freeze exists; use run --resume')
    if any(out.joinpath(name).exists() for name in ('training','quantization','tasks','state.json')):
        raise FileExistsError('Training artifacts already exist; refusing freeze overwrite')
    cov=coverage();save(out/'coverage.json',cov)
    if cov['architectures']!=144 or cov['triples']!=432 or cov['looked_up']!=432 or cov['missing']:
        raise ValueError('Hardware coverage incomplete; measurements must pass before freezing search')
    snapshot=cost_snapshot(cov);save(out/'cost_snapshot.json',snapshot)
    files=inputs(config_path);dataset=data_binding(cfg)
    hashes=named_hashes(cfg,files,dataset,cov,snapshot)
    if any(not hashes[k] for k in ('config_hash','code_hash','data_hash','cost_db_hash','candidate_space_hash')):
        raise ValueError('Named freeze hash missing')
    binding=dict(protocol=cfg,files=files,dataset=dataset,test_evaluated=False,training_started=False,
        hashes=hashes,platform=PLATFORM,coverage={k:cov[k] for k in ('architectures','triples','looked_up','missing')},
        mixed_precision={p:'available' for p in PROFILES},pending_measurements=0,
        cost_snapshot='cost_snapshot.json',cost_snapshot_sha256=sha(out/'cost_snapshot.json'))
    save(out/'freeze.json',binding)
    with zipfile.ZipFile(out/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for rel in binding['files']:archive.write(ROOT/rel,rel)
    save(out/'snapshot.json',dict(sha256=sha(out/'source_snapshot.zip')))
    print('Frozen',out,flush=True)

def verify(config_path,out):
    binding=read(out/'freeze.json')
    cfg=load_config(config_path);files=inputs(config_path);dataset=data_binding(cfg)
    if binding['protocol']!=cfg or binding['files']!=files:raise ValueError('Source/config/cost binding drift; refusing resume')
    if dataset!=binding['dataset']:raise ValueError('Dataset drift')
    if sha(out/'source_snapshot.zip')!=read(out/'snapshot.json')['sha256']:raise ValueError('Source snapshot drift')
    if sha(out/'cost_snapshot.json')!=binding.get('cost_snapshot_sha256'):raise ValueError('Cost snapshot drift')
    cov=read(out/'coverage.json');snapshot=read(out/'cost_snapshot.json')
    expected=named_hashes(cfg,files,dataset,cov,snapshot)
    if binding.get('hashes')!=expected:raise ValueError('Named hash drift; refusing resume')
    live=coverage()
    if {k:live[k] for k in ('architectures','triples','looked_up','missing')}!={k:cov[k] for k in ('architectures','triples','looked_up','missing')}:
        raise ValueError('Live cost coverage drift; refusing resume')
    CostDatabase()  # Reverify reports and measurement source hashes.
    return binding

@contextmanager
def run_lock(out):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a+b') as handle:
        handle.seek(0);handle.write(b'0');handle.flush();handle.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            handle.seek(0)
            if os.name=='nt':msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle,fcntl.LOCK_UN)

def task(out,mode,aid,seed,budget,device):
    key=f'{mode}_{aid}_{seed}_{budget}'
    receipt=out/'tasks'/(key+'.json')
    binding=sha(out/'freeze.json')
    if receipt.exists():
        r=read(receipt)
        if r['freeze_sha256']!=binding or any(sha(out/p)!=h for p,h in r['outputs'].items()):raise ValueError('Task artifact drift '+key)
        return r['result']
    log=out/'logs'/(key+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        subprocess.run([sys.executable,str(Path(__file__).with_name('worker.py')),mode,'--output',str(out),
            '--architecture',aid,'--seed',str(seed),'--budget',str(budget),'--device',device],cwd=ROOT,
            stdout=stream,stderr=subprocess.STDOUT,check=True)
    r=read(receipt)
    if r['freeze_sha256']!=binding or any(sha(out/p)!=h for p,h in r['outputs'].items()):raise ValueError('Worker receipt invalid')
    return r['result']

def rank(rows):
    if len({r['architecture_id'] for r in rows})!=len(rows) or any(not math.isfinite(r['mAP50_95']) for r in rows):raise ValueError('Invalid ranking')
    return sorted(rows,key=lambda r:(-r['mAP50_95'],r['architecture_id']))

def admit(fp32,uniform,values,profile,gate):
    if len(fp32)!=3 or len(uniform)!=3 or len(values)!=3:raise ValueError('Three paired seeds required')
    if any(not math.isfinite(v) for v in fp32+uniform+values):raise ValueError('Nonfinite metric')
    f=sum(fp32)/3;q=sum(values)/3;drop=f-q
    ok=drop<=gate['absolute_drop'] and drop<=gate['relative_drop']*f
    if profile!='W8A8':ok=ok and q>sum(uniform)/3 and sum(a>b for a,b in zip(values,uniform))>=gate['improved_seeds']
    return bool(ok)

def joint_points(evals,db,cfg):
    points=[];rejected=[]
    for aid,runs in sorted(evals.items()):
        if sorted(int(s) for s in runs)!=cfg['confirmation_seeds']:raise ValueError('Seed set mismatch')
        rr=[runs[str(seed)] for seed in cfg['confirmation_seeds']]
        for r in rr:
            if r['architecture_id']!=aid or r['test_evaluated'] or r['split']!='valid':raise ValueError('Evaluation binding mismatch')
        fp=[r['fp32']['mAP50_95'] for r in rr];uniform=[r['profiles']['W8A8']['mAP50_95'] for r in rr]
        for q in cfg['quant_profiles']:
            values=[r['profiles'][q]['mAP50_95'] for r in rr]
            if not admit(fp,uniform,values,q,cfg['admission']):
                rejected.append(dict(architecture_id=aid,profile=q,reason='accuracy_gate',values=values));continue
            for policy in cfg['mapping_policies']:
                cost=db.estimate_cost(aid,q,policy)
                if cost.get('status')!='LOOKED_UP' or cost.get('hardware_status')!='MEASURED':
                    rejected.append(dict(architecture_id=aid,profile=q,mapping=policy,
                        reason=cost.get('hardware_status') or cost.get('status') or 'NEEDS_MEASUREMENT'))
                    continue
                # Campaign PTQ is the accuracy for this candidate. Do not keep a Phase2F FP32 score.
                cost.update(Accuracy=sum(values)/3,mean_mAP50_95=sum(values)/3,
                    std_mAP50_95=(sum((x-sum(values)/3)**2 for x in values)/2)**.5,
                    quantized_accuracy_bound=True,accuracy_precision=q,accuracy_status='MEASURED',
                    hardware_status='MEASURED',eligible=True,
                    checkpoint_sha256=[r['checkpoint_sha256'] for r in rr],
                    quant_profile_sha256=[r['profiles'][q]['profile_sha256'] for r in rr],
                    calibration_sha256=[r['calibration_sha256'] for r in rr],
                    n_seeds=3,test_evaluated=False,accuracy_split='valid')
                points.append(cost)
    return dict(points=points,rejected=rejected,accuracy_latency=pareto_max_min(points,'Accuracy','Latency'),
        accuracy_resources=pareto_accuracy_resources(points),deployment='NOT_CLAIMED',
        scope='validation_quantized_accuracy_vs_CSYNTH_operator_cost')

def execute(config_path,out,device,resume):
    if (out/'state.json').exists() and not resume:raise FileExistsError('Run exists; use --resume')
    binding=verify(config_path,out);cfg=binding['protocol'];selected=sorted(architectures());audit=[]
    try:
        # The full frozen population, not only B1/C/D, is a hardware-agent candidate set.
        agent=run_joint_search(architecture_ids=selected,profiles=cfg['quant_profiles'],policies=cfg['mapping_policies'])
        save(out/'hardware_agent_pass.json',agent['summary'])
        save(out/'missing_measurements.json',agent['missing_measurements'])
        save(out/'missing_accuracy.json',agent['missing_accuracy'])
        for stage,(budget,count) in enumerate(zip(cfg['budgets'],cfg['populations'])):
            if len(selected)!=count:raise ValueError('Promotion population mismatch')
            rows=[]
            for aid in selected:
                save(out/'state.json',dict(stage='architecture',budget=budget,architecture_id=aid,completed=len(rows),total=count))
                rows.append(task(out,'train',aid,42,budget,device))
            rows=rank(rows);save(out/f'stage_{budget}.json',rows)
            if stage<2:
                n=cfg['populations'][stage+1];removed=rows[n:]
                for combo in ('P3_P4_P5','P2_P3_P4','P2_P3_P4_P5'):
                    group=[r['architecture_id'] for r in removed if r['architecture_id'].split('_s')[0]==combo]
                    if group:audit.append(min(group,key=lambda a:digest([42,stage,a])))
                selected=[r['architecture_id'] for r in rows[:n]]
        audit_rows=[task(out,'train',aid,42,80,device) for aid in audit]
        save(out/'rejection_audit.json',audit_rows)
        if any(r['mAP50_95']>rows[4]['mAP50_95'] for r in audit_rows):raise ValueError('Rejection audit exceeded promoted top5; stop selection')
        shortlist=[r['architecture_id'] for r in rows[:cfg['shortlist']]]
        save(out/'shortlist.json',dict(architectures=shortlist,scope='promoted_population_not_global_optimum'))
        evaluations={}
        for aid in shortlist:
            evaluations[aid]={}
            for seed in cfg['confirmation_seeds']:
                task(out,'train',aid,seed,80,device)
                save(out/'state.json',dict(stage='quantization',architecture_id=aid,seed=seed))
                evaluations[aid][str(seed)]=task(out,'quant',aid,seed,80,device)
        report=joint_points(evaluations,CostDatabase(),cfg)
        save(out/'joint_pareto.json',report)
        save(out/'state.json',dict(status='COMPLETE' if report['points'] else 'COMPLETE_NO_ADMITTED_POINTS',test_evaluated=False,deployment='NOT_CLAIMED'))
    except Exception as error:
        save(out/'state.json',dict(status='FAILED_RESUMABLE',error=str(error)));raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['plan','freeze','run'])
    p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--output',type=Path,default=ROOT/'results/sonar_joint_v1')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--resume',action='store_true');a=p.parse_args()
    out=a.output.resolve();cfg=load_config(a.config)
    if a.mode=='plan':
        report=coverage();save(out/'coverage.json',report)
        save(out/'plan.json',dict(protocol=cfg,max_training_epochs=144*40+108*20+72*20+6*40+12*2*80,
            max_quant_evaluations=12*3*3,missing_costs=report['missing'],training_started=False))
        print(json.dumps({k:report[k] for k in ('architectures','triples','looked_up','missing')}));return
    with run_lock(out):
        if a.mode=='freeze':freeze(a.config,out)
        else:execute(a.config,out,a.device,a.resume)

if __name__=='__main__':main()
