"""Phase2C protocol binding, train-only calibration and deterministic witnesses."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import csv
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/sonar_analysis'),str(ROOT/'scripts')]
import baseline as frozen
from hwnas_fpga.models.sonar_candidate import load_spec
from hwnas_fpga.models.sonar_supernet import enumerate_architectures,architecture_id


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,data): frozen.dump(path,data)


def load_protocol(path):
    p=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if sha(ROOT/p['space'])!=p['space_sha256']: raise ValueError('Search space hash drift')
    load_spec(ROOT/p['space'])
    if sha(ROOT/p['baseline'])!=p['baseline_sha256']: raise ValueError('Baseline config hash drift')
    b=yaml.safe_load((ROOT/p['baseline']).read_text(encoding='utf-8'))
    if sha(ROOT/b['baseline_freeze']['manifest'])!=p['baseline_manifest_sha256']:raise ValueError('Baseline freeze manifest hash drift')
    required=dict(epochs=80,input=640,batch_size=4,seed=42,optimizer='AdamW',lr=.001,weight_decay=.0001,scheduler='none',amp=False,workers=0)
    for k,v in required.items():
        if p['training'][k]!=v: raise ValueError(f'Formal protocol fixed value mismatch: {k}')
    c=p['calibration']
    if c!=dict(split='train',augmentation=False,shuffle=False,workers=0,seed=42,batch_size=4,batches=16,images=64,
               order='frozen_loader_sorted_first_64',statistics='cumulative_batch_BN_mean_variance_in_independent_export'):
        raise ValueError('Formal calibration protocol must be fixed16 train-only batches')
    return p,b


def verify_baseline(b):
    """Normalize OS path spelling only; preserve and verify every frozen byte hash."""
    manifest=json.loads((ROOT/b['baseline_freeze']['manifest']).read_text())
    normalize=lambda d:{k.replace('\\','/'):v for k,v in d.items()}
    if normalize(frozen.source_hashes())!=normalize(manifest['source_sha256']): raise ValueError('Frozen42 source drift')
    root=ROOT/b['dataset']['data_dir']
    for split in ('train','valid'):
        loader=frozen.make_loader(b,root,split,42,0)
        actual=frozen.bind_dataset(loader)['entries'];expected=manifest['datasets'][split]['entries']
        clean=lambda rows:[{k:(v.replace('\\','/') if k in ('image','annotation') else v) for k,v in row.items()} for row in rows]
        if clean(actual)!=clean(expected): raise ValueError(f'Frozen data drift: {split}')
    return dict(source_files=len(manifest['source_sha256']),manifest_sha256=sha(ROOT/b['baseline_freeze']['manifest']),data_verified=['train','valid'])


def initialize_frozen_baseline(model,p,b):
    """Transport the original untrained seed42 state across OS floating-point differences."""
    import torch
    reference=p['baseline_initialization_reference'];path=ROOT/reference['path']
    if sha(path)!=reference['sha256']:raise ValueError('Frozen initialization artifact drift')
    initial=torch.load(path,map_location='cpu',weights_only=True)
    model.load_state_dict(initial['model_state_dict'],strict=True)
    manifest=json.loads((ROOT/b['baseline_freeze']['manifest']).read_text())
    if frozen.state_hash(model)!=manifest['initial_state_sha256']:raise ValueError('Frozen initialization tensor hash mismatch')
    torch.set_rng_state(initial['torch_rng'])
    return reference['sha256']


def source_binding(protocol_path):
    paths=[Path(protocol_path).resolve(),ROOT/'tools/sonar_spec/formal_protocol.py',ROOT/'tools/sonar_spec/formal_runner.py',
           ROOT/'tools/sonar_spec/ranking_analysis.py',ROOT/'reports/baseline_sonar_frozen/freeze_manifest.json']
    for package in ('sonar_candidate','sonar_supernet'):
        paths.extend(sorted((ROOT/'src/hwnas_fpga/models'/package).rglob('*.py')))
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}


def require_gate(p,protocol_path):
    audit=json.loads((ROOT/p['gate']['original_audit']).read_text())
    unresolved=[f['title'] for f in audit if f['severity'] in ('CRITICAL','MAJOR')]
    closure=ROOT/p['gate']['closure']
    if not closure.exists(): raise RuntimeError(f'Independent gate not closed: {unresolved}; missing {closure.name}')
    c=json.loads(closure.read_text())
    if c.get('status')!='PASS' or c.get('reviewer_role')!='independent' or c.get('open_critical_major')!=[]:
        raise RuntimeError('Independent review has unresolved blockers')
    if not set(unresolved)<=set(c.get('closed_findings',[])): raise RuntimeError('Audit findings not independently closed')
    if c.get('source_sha256')!=source_binding(protocol_path): raise RuntimeError('Independent review source binding stale')
    if c.get('original_audit_sha256')!=sha(ROOT/p['gate']['original_audit']): raise RuntimeError('Independent audit binding stale')
    return sha(closure)


def calibration_manifest(b,out):
    loader=frozen.make_loader(b,ROOT/b['dataset']['data_dir'],'train',42,0)
    ds=loader.dataset
    if len(ds)<64: raise ValueError('Need64 train calibration images')
    entries=[]
    for image in ds.samples[:64]:
        label=ds.labels_dir/(image.stem+'.txt')
        entries.append(dict(image=str(image.relative_to(ds.root)).replace('\\','/'),image_sha256=sha(image),annotation_sha256=sha(label)))
    payload=dict(split='train',batches=16,batch_size=4,seed=42,augmentation=False,shuffle=False,workers=0,entries=entries)
    payload['samples_sha256']=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()
    path=out/'calibration_manifest.json'
    if path.exists() and json.loads(path.read_text())!=payload: raise ValueError('Calibration sample manifest drift')
    dump(path,payload)
    return loader,payload


def select_witness(rows):
    """Fixed9 stratified +3 random; kernel coverage reported, never cherry-picked."""
    all_arch={architecture_id(a):a for a in enumerate_architectures()}
    ids=[r['architecture_id'] for r in rows]
    if len(ids)!=144 or set(ids)!=set(all_arch): raise ValueError('Witness selection requires144 unique legal evaluations')
    selected=[]
    for combo in ('P3_P4_P5','P2_P3_P4','P2_P3_P4_P5'):
        group=sorted([r for r in rows if r['output_combination']==combo],key=lambda r:(-float(r['mAP50_95']),r['architecture_id']))
        for index in (0,(len(group)-1)//2,len(group)-1): selected.append(group[index]['architecture_id'])
    remaining=sorted(set(ids)-set(selected))
    selected+=random.Random(42).sample(remaining,3)
    assert len(set(selected))==12
    return [dict(architecture_id=i,architecture=all_arch[i]) for i in selected]


def load_witness(path,protocol_path):
    path=Path(path);manifest=json.loads(path.read_text())
    if manifest['protocol_sha256']!=sha(protocol_path):raise ValueError('Witness protocol drift')
    ranking=path.parent/'subnet_inherited_ranking.csv'
    if sha(ranking)!=manifest['ranking_sha256']:raise ValueError('Witness ranking binding drift')
    with ranking.open(newline='') as f:rows=list(csv.DictReader(f))
    if manifest['witnesses']!=select_witness(rows):raise ValueError('Witness list differs from deterministic frozen rule')
    calibration=json.loads((path.parent/'calibration_manifest.json').read_text())
    if manifest['calibration_sha256']!=calibration['samples_sha256']:raise ValueError('Witness calibration binding drift')
    return manifest
