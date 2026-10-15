"""Prepare exact missing width transitions, sonar slots and a serial GAP control."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'hls_lut_builder/scripts'))
from common import build_cases, load_config, render_template

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot-root',type=Path,default=ROOT/'results/zcu104_width_expansion_pilot_20260912_v1')
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists(): raise FileExistsError(out)
    bundle=out/'bundle'
    original=args.pilot_root/'bundle'
    for directory in ['hls_lut_builder/scripts','hls_lut_builder/include','hls_lut_builder/templates']:
        shutil.copytree(original/directory,bundle/directory)
    for filename in ['global_avg_pool_serial','dmdc_conv_sonar','cfem_sonar']:
        shutil.copy2(ROOT/f'hls_lut_builder/templates/{filename}.cpp.tmpl',bundle/f'hls_lut_builder/templates/{filename}.cpp.tmpl')
    (bundle/'scripts').mkdir()
    for filename in ['run_zcu104_width_expansion_pilot.py','check_zcu104_gap_integer_semantics.py',
                     'build_zcu104_mobilenetA_full_formal_lut_from_reports.py']:
        shutil.copy2(ROOT/'scripts'/filename,bundle/'scripts'/filename)
    base=yaml.safe_load((original/'hls_lut_builder/configs/width_expansion_pilot.yaml').read_text())
    cfg=copy.deepcopy(base)
    cfg['operators']={}
    cfg['representative_shapes']={}
    def add(kind,cin,cout,h,stride=1,kernel=3,expand=1,template=None,**extra):
        name=f'{kind}_h{h}_i{cin}_o{cout}_s{stride}_k{kernel}_e{expand}'
        params=dict(feature_h=h,feature_w=h,in_channels=cin,out_channels=cout,stride=stride,
                    kernel_size=kernel,padding=kernel//2,expand_ratio=expand,**extra)
        chosen=template or ('mbconv_e1_noexpand' if kind=='mbconv' and expand==1 else kind)
        cfg['representative_shapes'][name]=params
        cfg['operators'][name]=dict(enabled=True,template=f'../templates/{chosen}.cpp.tmpl',
            top_function=f'mbconv_e{expand}_k{kernel}_kernel' if kind=='mbconv' else f'{kind}_kernel',
            op_type=kind,shape_refs=[name],implementation_refs=['balanced_pi4_po4_u1'],
            clock_profile_refs=['main_5ns'],parameters=params,op_spec_defaults={'op':kind,'kernel_size':kernel,'expand_ratio':expand})
    # First make the timing-repair control available to whole-network integration.
    add('global_avg_pool',1280,1280,7,kernel=1,template='global_avg_pool_serial')
    add('dmdc_conv_sonar',32,32,28,dilation_rates='1_3_5')
    add('cfem_sonar',32,32,28,dilation_rates='2_3_5')
    for k in [3,5]: add('mbconv',32,32,28,kernel=k,expand=3)
    for cin,cout,h,stride in [(96,112,14,2),(112,256,7,1),(128,192,7,1)]:
        for e in [1,2,3]: add('mbconv',cin,cout,h,stride=stride,expand=e)
    path=bundle/'hls_lut_builder/configs/width_e_extension.yaml'
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    cases=build_cases(load_config(path))
    records=[]
    for case in cases:
        source=render_template(case.template_path.read_text(),{'top_function':case.top_function,**case.parameters})
        records.append(dict(operator_name=case.operator,case_name=case.case_name,top_function=case.top_function,
                            source_sha256_lf=hashlib.sha256(source.replace('\r\n','\n').encode()).hexdigest(),
                            parameters=case.parameters,legacy_row=None))
    manifest=dict(schema_version=1,pilots=[],operators=records,unique_operator_count=len(cases),nas_resume_allowed=False,
                  intended_space=dict(stage_channels=[[16],[24],[32],[32],[64],[80,96],[112,128],[192,256]],
                                      tail_expansions=[1,2,3],sonar_slot=['skip','mbconv_k3_e3','mbconv_k5_e3','dmdc_conv_sonar','cfem_sonar'],
                                      legal_architectures=360),
                  mapping_change='serial GAP control plus synchronous distributed FIFO d64; fixed 200MHz')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'source_freeze.json').write_text(json.dumps({str(p.relative_to(bundle)):hashlib.sha256(p.read_bytes()).hexdigest()
                                                   for p in bundle.rglob('*') if p.is_file()},indent=2)+'\n')
    shutil.copytree(args.pilot_root/'remote_output/e1_witness',out/'output/e1_witness')
    print(json.dumps(dict(output=str(out),operators=len(cases))))

if __name__=='__main__': main()
