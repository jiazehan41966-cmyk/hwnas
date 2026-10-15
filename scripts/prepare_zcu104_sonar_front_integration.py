"""Prepare four measured-kernel whole networks for the 720-point successor."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')

def main():
    old=Path('/home/jiajia/experiments/zcu104_width_e_pilot_20260912_v1')
    extension=Path('/home/jiajia/experiments/zcu104_width_e_extension_20260913_v1')
    front=Path('/home/jiajia/experiments/zcu104_sonar_front_pilot_20260913_v1')
    old_bundle=Path('/dev/shm/hwnas_width_e_20260912_v1/bundle')
    out=Path('/home/jiajia/experiments/zcu104_sonar_front_integration_20260913_v1')
    bundle=Path('/dev/shm/hwnas_sonar_front_reference_20260913_v1/bundle')
    if (out/'manifest.json').exists() or bundle.exists(): raise FileExistsError('Preserve the existing front-integration experiment')
    out.mkdir(parents=True,exist_ok=True)
    (out/'bundle').symlink_to(extension/'bundle',target_is_directory=True)
    manifest=json.loads((old_bundle/'pilot_manifest.json').read_text())
    sources={r['case_name']:(r,old/'output') for r in manifest['operators']}
    for source in [extension,front]:
        for r in json.loads((source/'manifest.json').read_text())['operators']:
            sources[r['case_name']]=(r,source/'output')
    def select(prefix):
        found=[r for r,_ in sources.values() if r['operator_name']==prefix]
        if len(found)!=1: raise ValueError('Ambiguous exact operator '+prefix)
        return found[0]
    s3=select('mbconv_h56_i24_o32_s2_k3_e3')
    dmdc=select('dmdc_conv_sonar_h28_i32_o32_s1_k3_e1')
    cfem=select('cfem_sonar_h28_i32_o32_s1_k3_e1')
    pilots=[]
    for width,slot in [('c','skip'),('c','dmdc'),('c','cfem'),('b','dmdc')]:
        pilot=copy.deepcopy(next(p for p in manifest['pilots'] if p['pilot_id']==f'pilot_{width}_tail_e1'))
        pilot['pilot_id']=f'pilot_{width}_front_e3_{slot}'
        pilot['architecture']['stages'][2]['blocks']=[dict(op='mbconv',kernel_size=3,expand_ratio=3,stride=2)]
        pilot['case_names'][3]=s3['case_name'];pilot['operator_names'][3]=s3['operator_name']
        if slot!='skip':
            operator=dmdc if slot=='dmdc' else cfem
            pilot['architecture']['stages'][3]['blocks']=[dict(op=slot+'_conv_sonar' if slot=='dmdc' else 'cfem_sonar',kernel_size=3,expand_ratio=1,stride=1)]
            pilot['case_names'][4]=operator['case_name'];pilot['operator_names'][4]=operator['operator_name']
        pilots.append(pilot)
    manifest.update(pilots=pilots,pilot_count=len(pilots),operators=[r for r,_ in sources.values()],
                    purpose='Core feasible seed, sonar resource calibration, architecture-disjoint width/DMDC holdout',nas_resume_allowed=False)
    shutil.copytree(old_bundle,bundle,ignore=shutil.ignore_patterns('_vivado_builds','.Xil','__pycache__','*.log','*.jou'))
    write(bundle/'pilot_manifest.json',manifest)
    write(out/'manifest.json',manifest)
    output=out/'output/operators';output.mkdir(parents=True)
    used={n for p in pilots for n in p['case_names']}
    for name,(record,source) in sources.items():
        path=source/'operators'/f'{name}.json'
        if name in used:
            measured=json.loads(path.read_text())
            if measured.get('status')!='PASS_OPERATOR_REPORTS' or not measured.get('timing_met'):
                raise ValueError('Unready operator '+name)
            if measured['source_sha256']!=record['source_sha256_lf']: raise ValueError('Source mismatch '+name)
        (output/path.name).symlink_to(path)
    shutil.copytree(extension/'output/gap_witness',out/'output/gap_witness')
    write(out/'preparation.json',dict(reference_bundle=str(bundle),pilots=[p['pilot_id'] for p in pilots],
        manifest_sha256=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),required_operator_cases=sorted(used),
        architecture_changes=['S3 k5/e6 -> k3/e3','S4 sonar slot','C/B tail width profile'],
        hardware_flow='same cached cores, synchronous distributed FIFO d64, 200MHz, post-route AggressiveExplore + MoreGlobalIterations'))
    print(json.dumps(dict(output=str(out),reference_bundle=str(bundle),pilots=[p['pilot_id'] for p in pilots])))

if __name__=='__main__': main()
