from copy import deepcopy
from pathlib import Path
import random
import sys
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/sonar_spec'))
from formal_protocol import load_protocol,calibration_manifest,select_witness,require_gate,verify_baseline,source_binding
from formal_runner import train_epoch,rng_state,restore_rng
from ranking_analysis import correlations
from hwnas_fpga.models.sonar_supernet import enumerate_architectures,architecture_id,build_sonar_supernet,BalancedArchitectureSampler
import baseline as frozen

PROTOCOL=ROOT/'configs/supernet_sonar_v0_1_formal.yaml'
torch.set_num_threads(2)


def test_calibration_manifest_is_fixed_train_only(tmp_path):
    p,b=load_protocol(PROTOCOL);loader,a=calibration_manifest(b,tmp_path);_,again=calibration_manifest(b,tmp_path)
    assert a==again and len(a['entries'])==64 and a['batches']==16
    assert all(e['image'].startswith('train/images/') for e in a['entries'])
    assert loader.num_workers==0 and loader.batch_size==4
    assert verify_baseline(b)['source_files']==42


def test_missing_independent_gate_rejected():
    p,b=load_protocol(PROTOCOL);p['gate']['closure']='reports/phase2c_supernet/nonexistent_closure.json'
    with pytest.raises(RuntimeError,match='Independent gate not closed'):require_gate(p,PROTOCOL)


def test_candidate_package_is_in_formal_binding():
    binding=source_binding(PROTOCOL)
    for package in ('sonar_candidate','sonar_supernet'):
        for source in (ROOT/'src/hwnas_fpga/models'/package).rglob('*.py'):
            assert str(source.relative_to(ROOT)).replace('\\','/') in binding


def test_witness_is_deterministic_and_tie_safe():
    rows=[dict(architecture_id=architecture_id(a),output_combination=a['output_combination'],mAP50_95=0.) for a in enumerate_architectures()]
    selected=select_witness(rows)
    assert selected==select_witness(list(reversed(rows)))
    assert len({w['architecture_id'] for w in selected})==12
    for c in ('P3_P4_P5','P2_P3_P4','P2_P3_P4_P5'):
        assert sum(w['architecture']['output_combination']==c for w in selected)>=3
    with pytest.raises(ValueError):select_witness(rows[:-1])


def test_correlation_paired_bootstrap():
    x=np.arange(12,dtype=float)
    positive=correlations(x,x,bootstrap_samples=100)
    negative=correlations(x,-x,bootstrap_samples=100)
    assert positive['spearman']['value']==pytest.approx(1)
    assert negative['kendall_tau_b']['value']==pytest.approx(-1)
    assert positive['spearman']['percentile_ci95']==pytest.approx([1,1])
    undefined=correlations(np.ones(12),x,bootstrap_samples=10)
    assert undefined['spearman']['value'] is None


class EpochDataset(Dataset):
    def __init__(self,image,target):self.image=image;self.target=target
    def __len__(self):return 4
    def __getitem__(self,i):
        noise=(random.random()+np.random.random()+float(torch.rand(())))*.001
        return self.image+noise,self.target


def test_epoch_boundary_exact_resume_with_data_and_architecture_rng():
    p,b=load_protocol(PROTOCOL);frozen.seed_all(42)
    original=frozen.make_loader(b,ROOT/b['dataset']['data_dir'],'train',42,0)
    images,targets=next(iter(original));ds=EpochDataset(images[0],targets[0])
    def loader():
        sampler=WeightedRandomSampler(torch.ones(4),4,replacement=True,generator=torch.Generator().manual_seed(42))
        return DataLoader(ds,batch_size=4,sampler=sampler,num_workers=0,generator=torch.Generator().manual_seed(77),collate_fn=lambda rows:(torch.stack([r[0] for r in rows]),[r[1] for r in rows]))
    train=loader();model=build_sonar_supernet();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);sampler=BalancedArchitectureSampler(42)
    train_epoch(model,opt,train,b,'cpu',sampler)
    saved=dict(model=deepcopy(model.state_dict()),opt=deepcopy(opt.state_dict()),rng=rng_state(train),sampler=sampler.state_dict())
    expected=train_epoch(model,opt,train,b,'cpu',sampler)
    restored=build_sonar_supernet();optimizer=torch.optim.AdamW(restored.parameters(),lr=.001,weight_decay=.0001);sample2=BalancedArchitectureSampler();train2=loader()
    restored.load_state_dict(saved['model']);optimizer.load_state_dict(saved['opt']);sample2.load_state_dict(saved['sampler']);restore_rng(saved['rng'],train2)
    actual=train_epoch(restored,optimizer,train2,b,'cpu',sample2)
    assert expected['losses']==actual['losses']
    assert expected['combination_counts']==actual['combination_counts'] and expected['kernel_counts']==actual['kernel_counts']
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())


def test_analysis_pipeline_on_explicitly_synthetic_fixture(tmp_path,monkeypatch):
    """No real architecture evaluation: exercise plotting/provenance on mock scalar scores."""
    import json,hashlib,pandas as pd
    import ranking_analysis as analysis
    from formal_protocol import sha
    out=tmp_path/'synthetic_only';out.mkdir();runs=tmp_path/'runs';runs.mkdir()
    rows=[]
    for i,a in enumerate(enumerate_architectures()):
        rows.append(dict(architecture_id=architecture_id(a),output_combination=a['output_combination'],
            **{f'stage{s}':a['kernels'].get(f'stage{s}') for s in (1,2,4,5,6,7)},mAP50_95=(i+1)/200,MACs=1e9+i*1000))
    ranking=out/'subnet_inherited_ranking.csv';pd.DataFrame(rows).to_csv(ranking,index=False)
    witness=dict(protocol_sha256='synthetic_protocol',ranking_sha256=sha(ranking),witnesses=select_witness(rows))
    witness_path=out/'ranking_witness_manifest.json';witness_path.write_text(json.dumps(witness))
    (out/'calibration_manifest.json').write_text('{}')
    scores={r['architecture_id']:r['mAP50_95'] for r in rows}
    shared=[]
    for i,item in enumerate(witness['witnesses']):
        score=scores[item['architecture_id']];shared.append(dict(architecture_id=item['architecture_id'],mAP50_95=1-score))
        run=runs/f'witness_{i:02d}';run.mkdir()
        (run/'status.json').write_text(json.dumps(dict(status='COMPLETE',epochs=80,binding=dict(mode='standalone',witness_index=i,protocol_sha256='synthetic_protocol',witness_sha256=sha(witness_path)))))
        (run/'best_valid_metrics.json').write_text(json.dumps(dict(mAP50_95=score,AP50=score,AP75=score,Precision=score,Recall=score)))
    pd.DataFrame(shared).to_csv(out/'witness_shared_bn.csv',index=False)
    (out/'ranking_provenance.json').write_text(json.dumps(dict(status='COMPLETE',ranking_sha256=sha(ranking),witness_manifest_sha256=sha(witness_path),
        shared_bn_sha256=sha(out/'witness_shared_bn.csv'),calibration_manifest_sha256=sha(out/'calibration_manifest.json'))))
    real=analysis.correlations
    monkeypatch.setattr(analysis,'correlations',lambda x,y:real(x,y,bootstrap_samples=100))
    monkeypatch.setattr(sys,'argv',['ranking_analysis','--directory',str(out),'--standalone-root',str(runs)])
    analysis.main()
    result=json.loads((out/'reliability_statistics.json').read_text())
    assert result['calibrated_bn']['spearman']['value']==pytest.approx(1)
    assert result['shared_bn']['spearman']['value']==pytest.approx(-1)
    assert result['overlaps']['top3']['count']==3
    assert len(list((out/'figures').glob('*.png')))==6
    assert len(list((out/'figures').glob('*.csv')))==6
