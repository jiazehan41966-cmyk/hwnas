"""Prepare frozen-candidate coverage and a finite two-mapping pilot batch."""
import sys,re,json,shutil,importlib.util
from pathlib import Path
from common import *
sys.path[:0]=[str(ROOT/'tools/fpga_h2'),str(ROOT/'src')]
import numpy as np
import torch
import build as h2
from integer_reference import PackageBuilder,forward
from numeric import quantize_real

def description(nodes,kind):
    a,b=nodes[0],nodes[-1]
    if kind=='up':
        ci,h,w=GRAPH['edges'][a['inputs'][0]]['shape'];co=ci;k=1;s=1;g=1;act='aligned_add'
    else:
        ci,h,w=a['Cin'],a['H'],a['W'];co=b['Cout'];n=nodes[1] if kind=='mb' else a
        k,s,g=n['kernel'],n['stride'],n['groups'];act='relu6_relu6_linear' if kind=='mb' else a['activation']
    op={'up':'upsample_add','dw':'DW','mb':'MBConv','conv':'Conv'}[kind]
    return dict(operator_type=op,Cin=ci,Cout=co,H=h,W=w,kernel=k,stride=s,groups=g,activation=act,expanded_channels=a['Cout'] if kind=='mb' else 0)

def main():
    global GRAPH
    OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    hashes=contracts();torch.set_num_threads(4)
    sys.path.insert(0,str(ROOT/'tools/sonar_spec'))
    from candidate_sanity import protection
    frozen=json.loads((ROOT/'reports/baseline_sonar_frozen/freeze_manifest.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in frozen['source_sha256'].items())
    assert sha(ROOT/'configs/baseline_sonar_frozen.yaml')==frozen['config_sha256']
    guard=dict(frozen_files_unchanged=True,config_unchanged=True,glob_extra_unrelated='src/hwnas_fpga/data/evidence_audit.py')
    candidates=json.loads((ROOT/'reports/phase2f_multiseed/network_candidate_set.json').read_text())['architectures']
    allk3='P3_P4_P5_'+ '_'.join(f's{s}k3' for s in [1,2,4,5,6,7])
    candidates=[dict(label='baseline',architecture_id=allk3)]+candidates
    sources={};usage={};packages={};selected={};coverage=[]
    h2.OUT=ART
    for candidate in candidates:
        label=candidate['label'];aid=candidate['architecture_id']
        arch=dict(output_combination=aid.split('_s')[0],kernels={f'stage{s}':int(k) for s,k in re.findall(r's(\d)k(\d)',aid)})
        torch.manual_seed(42);model=h2.build_sonar_candidate(h2.SPEC_PATH,arch).eval()
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m,torch.nn.BatchNorm2d):
                    m.running_mean.uniform_(-.1,.1);m.running_var.uniform_(.6,1.4);m.weight.uniform_(.8,1.2);m.bias.uniform_(-.05,.05)
        folded,_=h2.fold_candidate_bn(model);folder=ART/('graph_'+label)
        GRAPH=PackageBuilder(folded,folder,hashes).build();packages[label]=str(folder.relative_to(ROOT))
        image=quantize_real(np.random.default_rng(42).normal(0,1,(1,640,640)),-5)
        outputs,values=forward(folder,image);np.save(folder/'input.npy',image)
        nodes=GRAPH['nodes'];byid={n['id']:n for n in nodes};usage[label]=[]
        entries=[([n],'up' if n['op']=='upsample_add' else 'dw' if n.get('groups',1)>1 else 'conv') for n in nodes]
        entries += [([byid[f'backbone.stages.{s}.0.{t}_conv'] for t in ['expand','dw','project']],'mb') for s in [1,2,4,5,6,7]]
        for ns,kind in entries:
            desc=description(ns,kind);key=shape_key(desc)
            sources.setdefault(key,set()).add(label)
            coverage.append(dict(shape_id=key,source=label,node=ns[0]['id'],role='composite_alternative' if kind=='mb' else 'atomic',**desc))
            if kind!='mb':usage[label].append(dict(shape_id=key,node=ns[0]['id']))
            # All FPN levels plus early DW/MB shapes; bounded batch, not exhaustive.
            want=ns[0]['id'].startswith(('neck.','head.')) or kind in ['dw','mb'] and ns[0]['id'].startswith(('backbone.stages.1.','backbone.stages.2.'))
            if want and key not in selected:
                name='shape_'+key
                c=h2.build_case(name,kind,ns,folder,GRAPH,values)
                selected[key]=dict(case=c,desc=desc)
        print('Prepared graph',label,len(nodes),'nodes',flush=True)
    save(ART/'network_usage.json',dict(packages=packages,usage=usage,candidates=candidates,baseline_guard=guard))
    # Each selected shape gets exactly two points. No fictitious tiling dimension.
    grid=[]
    for key,item in selected.items():
        c=item['case'];desc=item['desc'];base=ART/'representative_cases'/('shape_'+key)
        for index,p in enumerate([2,4]):
            pe=(1 if index==0 else 3) if desc['Cout']==21 else p
            simd=1 if desc['Cin']==1 else p
            name=f'{key}_p{pe}_s{simd}';d=ART/'cases'/name;shutil.copytree(base,d)
            src=(d/'kernel.cpp').read_text()
            # Replace only template constants; the V1 arithmetic is unchanged.
            for constant,value in [('IN_TILE',simd),('OUT_TILE',pe),('CH_TILE',pe),('EXP_TILE',simd)]:
                src=re.sub(r'(static constexpr int '+constant+r' = )[^;]+;',lambda m:m[1]+str(value)+';',src)
            (d/'kernel.cpp').write_text(src)
            tcl=(d/'run.tcl').read_text().replace('exit\n','csynth_design\nexit\n');(d/'run.tcl').write_text(tcl)
            c2=dict(c,name=name);save(d/'case.json',c2)
            sf={p.name:sha(p) for p in d.iterdir() if p.suffix in ['.h','.cpp','.tcl']}
            weight_bytes=sum(int(np.prod(n.get('weight_shape',[0]))) for n in c['nodes'])
            linebytes=(c['ec'] if c['kind']=='mb' else c['ci'])*c['k']*(c['w']+2*(c['k']//2))
            r=dict(point_id=name,shape_id=key,**desc,**hashes,input_bitwidth=8,weight_bitwidth=8,accumulator_bitwidth=32,output_bitwidth=8,
                PE=pe,SIMD=simd,input_tile=simd,output_tile=pe,spatial_tile=f"{c['h']}x{c['w']}",buffer_profile='linebuffer_fifo4_ddr_low' if c['kind']=='up' else 'linebuffer_fifo4',
                part=PART,tool_version='2022.2',target_clock_ns=5,source_hash=hashobj(sf),source_files=sf,
                input_stream_bits=c['input_pack']*8,output_stream_bits=c['output_pack']*8,wide_adapter_contract=True,linebuffer_bytes=linebytes,
                weight_buffer_bytes=weight_bytes,ddr_scratch_bytes=c['ci']*c['low_h']**2 if c['kind']=='up' else 0,
                sources='|'.join(sorted(sources[key])),split='validation' if int(key[:4],16)%4==0 else 'fit',case_dir=str(d.relative_to(ROOT)),kind=c['kind'])
            reasons=legality(r,hashes);r['LEGAL_MAPPING_CHECK']='PASS' if not reasons else 'REJECT';r['rejection_reason']='|'.join(reasons)
            save(d/'measurement_key.json',r);grid.append(r)
    for r in coverage:r['batch_selected']=r['shape_id'] in selected
    csvwrite(OUT/'H3_SHAPE_COVERAGE_PLAN.csv',coverage)
    csvwrite(OUT/'H3_MAPPING_GRID.csv',[{k:v for k,v in r.items() if k!='source_files'} for r in grid]);save(ART/'grid.json',grid)
    save(ART/'preflight.json',dict(**hashes,baseline_guard=guard,search_space_hash=sha(ROOT/'configs/search_space_sonar_v0_1.yaml'),candidate_hash=sha(ROOT/'reports/phase2f_multiseed/network_candidate_set.json'),seed=42,total_points=len(grid)))
    print('FINITE BATCH',len(grid),'points;',len(sources),'unique shapes',flush=True)
if __name__=='__main__':main()
