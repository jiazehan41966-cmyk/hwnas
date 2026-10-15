"""One added S3 k3/e3 measurement to create latency room for sonar slots."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'hls_lut_builder/scripts'))
from common import build_cases,load_config,render_template

def main():
    original=ROOT/'results/zcu104_width_e_extension_20260913_v1'
    out=ROOT/'results/zcu104_sonar_front_pilot_20260913_v1'
    if out.exists(): raise FileExistsError(out)
    shutil.copytree(original/'bundle',out/'bundle')
    config=yaml.safe_load((original/'bundle/hls_lut_builder/configs/width_e_extension.yaml').read_text())
    name='mbconv_h56_i24_o32_s2_k3_e3'
    params=dict(feature_h=56,feature_w=56,in_channels=24,out_channels=32,stride=2,kernel_size=3,padding=1,expand_ratio=3)
    config['representative_shapes']={name:params}
    config['operators']={name:dict(enabled=True,template='../templates/mbconv.cpp.tmpl',top_function='mbconv_e3_k3_kernel',
        op_type='mbconv',shape_refs=[name],implementation_refs=['balanced_pi4_po4_u1'],clock_profile_refs=['main_5ns'],
        parameters=params,op_spec_defaults=dict(op='mbconv',kernel_size=3,expand_ratio=3))}
    path=out/'bundle/hls_lut_builder/configs/sonar_front_pilot.yaml'
    path.write_text(yaml.safe_dump(config,sort_keys=False))
    case=build_cases(load_config(path))[0]
    source=render_template(case.template_path.read_text(),dict(top_function=case.top_function,**case.parameters))
    record=dict(operator_name=case.operator,case_name=case.case_name,top_function=case.top_function,
                source_sha256_lf=hashlib.sha256(source.replace('\r\n','\n').encode()).hexdigest(),parameters=case.parameters,legacy_row=None)
    manifest=dict(schema_version=1,pilots=[],operators=[record],nas_resume_allowed=False,
        motivation='All DMDC/CFEM candidates in the 360-point space fail the 120ms sum-of-HLS-cycle limit. Its fixed S3 k5/e6 costs 40.93145ms.',
        controlled_change='Measure S3 24->32 at 56x56 stride2 k3/e3; no budget or mapping relaxation.',
        proposed_space='Keep the frozen 360-point v1; add this S3 choice in a separate 720-point successor if measured.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copytree(original/'output/e1_witness',out/'output/e1_witness')
    (out/'source_freeze.json').write_text(json.dumps({str(p.relative_to(out/'bundle')):hashlib.sha256(p.read_bytes()).hexdigest()
                                                  for p in (out/'bundle').rglob('*') if p.is_file()},indent=2)+'\n')
    print(json.dumps(dict(output=str(out),case_name=case.case_name)))

if __name__=='__main__': main()
