"""Deduplicated executed-model shapes and explicit static storage assumptions."""
import csv
import json
import math
import sys
from pathlib import Path
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from hwnas_fpga.models.sonar_candidate import build_sonar_candidate,load_spec,SPEC_PATH
from hwnas_fpga.models.builder import MBConvBlock
OUT=ROOT/'reports/fpga_v0_1_closure'


def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    spec=load_spec(SPEC_PATH)['architecture_space'];unique={};buffers=[]
    for combo in spec['output_combinations']:
        for kernel in [3,5]:
            arch=dict(output_combination=combo['id'],kernels={f"stage{s['index']}":kernel for s in spec['stages'] if s['op']=='mbconv' and s['index']<=combo['last_active_stage']})
            model=build_sonar_candidate(SPEC_PATH,arch).eval().to('meta')
            def add(op,ci,ec,co,h,w,oh,ow,stride,k,pad,residual,role,activation):
                key=(op,ci,ec,co,h,w,oh,ow,stride,k,pad,residual,activation)
                if key not in unique:
                    unique[key]=dict(operator=op,Cin=ci,expanded_C=ec,Cout=co,H=h,W=w,out_H=oh,out_W=ow,stride=stride,kernel=k,padding=pad,residual=residual,activation=activation,roles=set(),combinations=set(),software_shape_evidence='six_candidate_meta_forwards_and_FP32_exports')
                unique[key]['roles'].add(role);unique[key]['combinations'].add(combo['id'])
            def hook(name,m):
                def collect(_,inputs,output):
                    x=inputs[0];ci,h,w=map(int,x.shape[1:]);co,oh,ow=map(int,output.shape[1:])
                    if isinstance(m,MBConvBlock):
                        add('MBConv',ci,ci*3,co,h,w,oh,ow,m.dw_conv.stride[0],m.dw_conv.kernel_size[0],m.dw_conv.padding[0],m.use_residual,name,'ReLU6/ReLU6/linear')
                    elif isinstance(m,nn.Conv2d):
                        op='DWConv' if m.groups==ci and ci>1 else 'Conv'
                        act='linear' if ('project_conv' in name or (name.startswith('head.towers.') and name.endswith('.3'))) else ('ReLU6' if ('expand_conv' in name or 'dw_conv' in name) else 'ReLU')
                        add(op,ci,ci if op=='DWConv' else '',co,h,w,oh,ow,m.stride[0],m.kernel_size[0],m.padding[0],False,name,act)
                    elif name=='backbone.stages.3.0':
                        add('identity',ci,'',co,h,w,oh,ow,1,1,0,False,name,'linear')
                return collect
            handles=[m.register_forward_hook(hook(name,m)) for name,m in model.named_modules() if isinstance(m,(nn.Conv2d,MBConvBlock)) or name=='backbone.stages.3.0']
            with torch.no_grad(): model(torch.empty(1,1,640,640,device='meta'))
            for h in handles:h.remove()
            for low,high in zip(combo['levels'][1:],combo['levels'][:-1]):
                hi=spec['features'][high]['spatial'];lo=spec['features'][low]['spatial']
                add('upsample_add',96,'',96,lo,lo,hi,hi,1,1,0,False,f'{low}_to_{high}','linear_saturating_add')
            # Uniform eager inference storage upper bound; no autograd, no tiling assumption.
            dims=[spec['features'][p]['spatial'] for p in combo['levels']]
            raw=sum(spec['features'][p]['channels']*spec['features'][p]['spatial']**2 for p in combo['levels'])
            lateral=sum(96*d*d for d in dims)
            extra_fused=sum(96*d*d for d in dims[:-1])
            pred=sum(21*d*d for d in dims)
            largest_nearest=96*dims[0]**2
            # Forward keeps backbone/lateral/fused and output tuple references; deliberately conservative overlap.
            neck_bound=raw+lateral+extra_fused+lateral+largest_nearest+pred
            mb_bound=0
            for s in spec['stages']:
                if s['op']!='mbconv' or s['index']>combo['last_active_stage']:continue
                ci=spec['stages'][s['index']-1]['channels'];co=s['channels']
                spatial=320
                for previous in spec['stages'][:s['index']]:spatial//=previous['stride']
                oh=spatial//s['stride'];ec=ci*3
                mb_bound=max(mb_bound,ci*spatial**2+ec*spatial**2+ec*oh**2+co*oh**2)
            weight=sum(m.weight.numel() for m in model.modules() if isinstance(m,nn.Conv2d))
            bias=sum(m.out_channels*4 for m in model.modules() if isinstance(m,nn.Conv2d))
            up_max=96*dims[1]**2
            buffers.append(dict(combination=combo['id'],all_active_kernel=kernel,evidence='static estimate',activation_bytes_per_element=1,
                retained_backbone_bytes=raw,lateral_feature_bytes=lateral,fused_extra_bytes=extra_fused,
                neck_output_bytes=lateral,prediction_bytes=pred,largest_upsample_buffer_bytes=up_max,
                largest_upsample_ideal_BRAM18=math.ceil(up_max/2304),
                eager_lifetime_upper_bound_bytes=max(raw+mb_bound,neck_bound),
                weights_bytes_INT8=weight,folded_bias_bytes_INT32=bias,
                lifetime_assumption='batch1_no_autograd_full_tensors_conservative_overlap_no_reuse; external storage allowed'))
    rows=[]
    for i,row in enumerate(unique.values(),1):
        row=dict(id=f'S{i:03d}',**row)
        row['roles']=';'.join(sorted(row['roles']));row['combinations']=';'.join(sorted(row['combinations']))
        row['feature_input_bytes_INT8']=row['Cin']*row['H']*row['W']
        row['feature_output_bytes_INT8']=row['Cout']*row['out_H']*row['out_W']
        if row['operator']=='MBConv':
            row['line_buffer_bytes_INT8']=row['expanded_C']*row['kernel']*(row['W']+2*row['padding'])
            row['weight_buffer_bytes_INT8']=row['Cin']*row['expanded_C']+row['expanded_C']*row['kernel']**2+row['expanded_C']*row['Cout']
        else:
            row['line_buffer_bytes_INT8']=row['Cin']*row['kernel']*(row['W']+2*row['padding']) if row['operator'] in ['Conv','DWConv'] else 0
            row['weight_buffer_bytes_INT8']=(row['Cin'] if row['operator']=='Conv' else 1)*row['Cout']*row['kernel']**2 if row['operator'] in ['Conv','DWConv'] else 0
        row['upsample_buffer_bytes_INT8']=row['feature_input_bytes_INT8'] if row['operator']=='upsample_add' else 0
        row['storage_evidence']='static estimate; aggregate_MBConv_and_suboperators_must_not_be_summed'
        rows.append(row)
    csv_write(OUT/'V0_1_SHAPE_CATALOG.csv',rows)
    csv_write(OUT/'BUFFER_STATIC_ESTIMATES.csv',buffers)
    print(len(rows),'unique operator/shape/activation contracts; macro static rows',len(buffers))


if __name__=='__main__':main()
