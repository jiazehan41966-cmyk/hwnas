"""Completed checkpoint software contract witness; no board or precision claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

import torch
from tools.detector_v0_2 import remote, train
from tools.detector_v0_2.ptq_export import export_ptq_package,profile_digest,sha256
from hwnas_fpga.deploy.sonar_v0_2_ptq import prepare_v0_2_ptq
from hwnas_fpga.models.sonar_candidate_v0_2 import build_sonar_candidate_v0_2
from tools.sonar_deploy.ptq_export import IntegerGraphModel


def witness(prototype,output):
    if prototype not in train.PROTOTYPES:
        raise ValueError('unregistered prototype')
    snapshot=ROOT/'results/detector_v0_2_execution_20261002/remote_snapshot/raw_bytes'
    source=snapshot/'formal'/prototype/'seed_42'
    result=json.loads((source/'result.json').read_bytes())
    status=json.loads((source/'status.json').read_bytes())
    if status['status']!='COMPLETE':
        raise ValueError('completed training required')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    folder=output/'training_run';folder.mkdir()
    for name in ('status.json','result.json','epoch_log.json','run_binding.json','model_info.json'):
        shutil.copyfile(source/name,folder/name)
    shutil.copyfile(snapshot/'execution_manifest.json',output/'execution_manifest.json')
    origin=remote.REMOTE+'/'+remote.RESULT+'/formal/'+prototype+'/seed_42'
    for name,key in (('best.pt','best_checkpoint_sha256'),('latest.pt','latest_checkpoint_sha256')):
        subprocess.run(remote.SCP+[remote.HOST+':'+origin+'/'+name,str(folder/name)],env=remote.ENV,check=True)
        if sha256((folder/name).read_bytes())!=result[key]:
            raise ValueError('downloaded checkpoint SHA mismatch')
    state=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    model=build_sonar_candidate_v0_2(architecture=prototype).eval()
    model.load_state_dict(state['model_state_dict'],strict=True)
    cfg,baseline=train.load_config()
    loader=train.frozen.make_loader(baseline,ROOT/baseline['dataset']['data_dir'],'train',42,0,train=False)
    images=[]
    for batch,_targets in loader:
        images.append(batch)
        if len(images)==4:break
    if sum(batch.shape[0] for batch in images)!=16:
        raise ValueError('requires sixteen deterministic train calibration images')
    provenance=dict(purpose='trained_checkpoint',checkpoint_path=str(folder/'best.pt'),
        checkpoint_sha256=result['best_checkpoint_sha256'],execution_manifest_path=str(output/'execution_manifest.json'),
        training_run_dir=str(folder),parameter_preparation_device='cpu',test_evaluated=False,
        precision_gate='NOT_ADMITTED_SINGLE_SEED_FP32_SCREENING',scope='trained software integer contract witness only')
    # Validate before any calibration/export work; this helper reads source,
    # train/valid input hashes and complete selected-best training artifacts.
    from tools.detector_v0_2.provenance import verify_completed_training
    binding=verify_completed_training(state,folder/'best.pt',provenance)
    with torch.no_grad():
        _,canonical,extrema,profile=prepare_v0_2_ptq(model,images,device='cpu')
    entries=json.loads((output/'execution_manifest.json').read_bytes())['input_verification']['datasets']['train']['entries'][:16]
    profile['calibration_sources']=entries
    profile_path=output/'profile.json';profile_path.write_text(json.dumps(profile,indent=2,allow_nan=False),encoding='utf-8')
    provenance['profile_sha256']=profile_digest(profile_path)
    export_ptq_package(model,profile_path,output/'package',provenance)
    graph=IntegerGraphModel(output/'package',model).eval()
    with torch.no_grad():
        actual=graph(images[0][:1])['predictions']
        expected=canonical(images[0][:1])['predictions']
    differences=[dict(level=level,total_elements=left.numel(),different_elements=int((left!=right).sum()),
        max_abs_difference=float((left-right).abs().max())) for level,left,right in zip(model.levels,expected,actual)]
    report=dict(prototype=prototype,seed=42,best_epoch=state['epoch'],FP32_valid_mAP50_95=result['mAP50_95'],
        real_weights_bound=True,calibration_split='train',calibration_images=16,
        integer_vs_canonical=differences,integer_vs_canonical_equal=all(x['different_elements']==0 for x in differences),
        software_numeric_scope='one real train image; no full valid PTQ accuracy or hardware parity',
        hardware_parity='NOT_RUN',board_latency='NOT_MEASURED',test_evaluated=False,**binding)
    (output/'WITNESS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))
    if not report['integer_vs_canonical_equal']:
        raise ValueError('trained integer contract mismatch')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prototype',choices=train.PROTOTYPES,default='R1')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    witness(args.prototype,args.output)
