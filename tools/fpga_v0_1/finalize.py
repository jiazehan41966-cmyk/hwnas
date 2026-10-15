"""Reconcile raw artifacts, classify evidence, and write reviewable closure reports."""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/fpga_v0_1_closure'
ART=ROOT/'artifacts/fpga_v0_1_closure/20260914_v1'
RAW=ART/'remote_results'
CASES=OUT/'representative_cases'


def loadcsv(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def writecsv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def dump(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen=sha(ROOT/'configs/search_space_sonar_v0_1.yaml')
    assert frozen=='ebb1708b394af20258c920eb29204d5cc9504f5908fdfdae62a62d78042eb7c4'
    raw=json.loads((RAW/'case_status.json').read_text());cases={}
    for p in CASES.glob('*/case.json'):
        c=json.loads(p.read_text());cases[c['name']]=c
    supplement=json.loads((RAW/'supplement_status.json').read_text())
    states=[];parity=loadcsv(OUT/'software_parity.csv')+loadcsv(RAW/'ort_parity.csv')
    for r in parity:
        r['scope']='software_only_FP32';r['hardware_detector_parity']='not_verified'
    for state in raw:
        name=state['case'];c=cases[name]
        if name==supplement['case']:state.update(supplement)
        logpath=RAW/(name+'_hls.log');log=logpath.read_text(errors='replace') if logpath.exists() else ''
        state['status']='HLS_TEMPLATE_AVAILABLE' if state['compile_returncode']==0 else 'BLOCKED'
        state.pop('blocker',None)
        # Parse the real Vitis 2022.2 completion marker. Preserve original transport status JSON.
        if 'CSim done with 0 errors.' in log and state.get('returncode')==0:
            a=np.fromfile(RAW/(name+'_hardware_output.bin'),dtype=np.int8).astype(np.int32)
            b=np.fromfile(CASES/name/'reference.bin',dtype=np.int8).astype(np.int32)
            assert a.shape==b.shape and len(a)>0
            d=np.abs(a-b)
            m=re.search(r'PARITY samples=(\d+) mismatches=(\d+) invalid_nonzero=(\d+) framing_errors=(\d+)',log)
            assert m and tuple(map(int,m.groups()))==(len(a),0,0,0)
            assert np.array_equal(a,b)
            state.update(status='HLS_CSIM_VERIFIED',csim_status='HLS_CSIM_VERIFIED',
                         parity=dict(max_abs_error=int(d.max()),mean_abs_error=float(d.mean()),exact_match_ratio=float((a==b).mean()),samples=len(a)))
            parity.append(dict(case=name,operator=c['kind'],domain='INT8_LEGACY_RAW_ARITHMETIC',status='NUMERICAL_PARITY_VERIFIED',**state['parity'],scope='unmodified_template_vs_independent_bounded_integer_reference',hardware_detector_parity='BLOCKER_FOR_NUMERICAL_PARITY'))
        elif c['run']:
            state.update(status='BLOCKED',blocker='CSIM_DID_NOT_PASS')
        xml=RAW/(name+'_csynth.xml')
        state['measured']={}
        if xml.exists() and state.get('returncode')==0:
            root=ET.parse(xml).getroot()
            paths={'latency_cycles':'.//SummaryOfOverallLatency/Worst-caseLatency','top_interval_cycles':'.//SummaryOfOverallLatency/Interval-max','DSP':'.//AreaEstimates/Resources/DSP','LUT':'.//AreaEstimates/Resources/LUT','FF':'.//AreaEstimates/Resources/FF','BRAM_18K':'.//AreaEstimates/Resources/BRAM_18K','URAM':'.//AreaEstimates/Resources/URAM','estimated_clock_ns':'.//SummaryOfTimingAnalysis/EstimatedClockPeriod'}
            state['measured']={k:root.findtext(v) for k,v in paths.items()}
            state.update(status='HLS_SYNTH_VERIFIED',synthesis_status='HLS_SYNTH_VERIFIED')
        state.update(evidence_source=str(logpath.relative_to(ROOT)) if logpath.exists() else str((RAW/(name+'_compile.log')).relative_to(ROOT)),
                     measured_scope='actual_HLS_report_estimates_only_no_route_or_board',source_sha256=sha(CASES/name/'kernel.cpp'))
        states.append(state)
    assert len(states)==22 and sum(s.get('csim_status')=='HLS_CSIM_VERIFIED' for s in states)==12
    dump(CASES/'tool_evidence.json',states)
    writecsv(OUT/'parity_results.csv',parity)
    measured=[]
    for s in states:
        measured.append(dict(case=s['case'],status=s['status'],tool=s['tool'],clock_target_ns=5,command=' '.join(s.get('command',s['compile_command'])),
            latency_cycles=s['measured'].get('latency_cycles',''),II_top_interval_cycles=s['measured'].get('top_interval_cycles',''),
            DSP=s['measured'].get('DSP',''),LUT=s['measured'].get('LUT',''),FF=s['measured'].get('FF',''),BRAM_18K=s['measured'].get('BRAM_18K',''),URAM=s['measured'].get('URAM',''),
            source=s['evidence_source'],resource_evidence='actual tool report: HLS estimates' if s['measured'] else 'not measured'))
    writecsv(CASES/'case_results.csv',measured)
    matrix=[]
    for row in loadcsv(OUT/'V0_1_SHAPE_CATALOG.csv'):
        matches=[]
        for s in states:
            c=cases[s['case']]
            if row['operator']!={'conv':'Conv','dw':'DWConv','mb':'MBConv','up':'upsample_add'}[c['kind']]:continue
            rh=c['low'] if c['kind']=='up' else c['h']
            if [int(row[k]) for k in ['Cin','Cout','H','kernel','stride']]==[c['ci'],c['co'],rh,c['k'],c['stride']]:matches.append(s)
        rank={'BLOCKED':-1,'HLS_TEMPLATE_AVAILABLE':0,'HLS_CSIM_VERIFIED':1,'HLS_SYNTH_VERIFIED':2}
        best=max(matches,key=lambda s:rank[s['status']]) if matches else None
        matrix.append(dict(id=row['id'],operator=row['operator'],roles=row['roles'],Cin=row['Cin'],expanded_C=row['expanded_C'],Cout=row['Cout'],H=row['H'],W=row['W'],kernel=row['kernel'],stride=row['stride'],
            software_support=True,export_status='EXPORT_VERIFIED',template_available=True,
            implementation_status=best['status'] if best else 'HLS_TEMPLATE_AVAILABLE',
            instantiation_detail='exact_shape_vendor_header_compile' if best else 'SUPPORTED_BY_TEMPLATE_ONLY',
            raw_integer_numerical_status='NUMERICAL_PARITY_VERIFIED' if any(s.get('csim_status') for s in matches) else 'unverified',
            calibrated_detector_numerical_status='BLOCKED',
            blockers='BLOCKER_FOR_NUMERICAL_PARITY; scale_and_interface_contract_unbound',
            cases=';'.join(s['case'] for s in matches),
            evidence='representative_cases/tool_evidence.json' if matches else 'repository_template_only'))
    writecsv(OUT/'OPERATOR_COMPATIBILITY_MATRIX.csv',matrix)
    dump(OUT/'quantization_interface.json',dict(schema_version=1,status='BLOCKER_FOR_NUMERICAL_PARITY',search_variables_added=False,
        current_hls=dict(weight_bits=8,activation_bits=8,accumulator_bits=32,signed=True,output_range=[-128,127],scale_conversion='absent',relu6_integer_upper=6),
        software_reference=dict(source='src/hwnas_fpga/deploy/fixed_point.py',zero_point=0,range=[-127,127],rounding='nearest_even',scale_representation='positive real scales converted to deterministic integer numerator/denominator'),
        required_per_edge=['tensor_id','producer','consumer','shape','layout','scale','zero_point','signed_range','bit_width','lifetime','output_level'],
        required_per_conv=['folded_weight_sha256','folded_bias_sha256','input_scale','weight_scale','output_scale','bias_scale','requant_numerator','requant_denominator','intermediate_width','rounding','saturation','activation_kind','activation_integer_bounds'],
        required_per_add=['left_scale','right_scale','output_scale','left_multiplier','right_multiplier','round_once_after_wide_sum','accumulator_width','saturation'],
        missing_implementations=['HLS scale conversion','ReLU versus scaled ReLU6 selector','overflow policy agreement','detector multi-output integer interpreter','DMA/tensor metadata binding']))
    write_documents(states,parity,frozen)
    inputs=[ROOT/'configs/search_space_sonar_v0_1.yaml',ROOT/'src/hwnas_fpga/models/sonar_candidate/__init__.py',ROOT/'src/hwnas_fpga/models/detection.py',ROOT/'src/hwnas_fpga/models/builder.py',
            ROOT/'tools/fpga_v0_1/export_contract.py',ROOT/'src/hwnas_fpga/deploy/ptq_eval.py',ROOT/'src/hwnas_fpga/deploy/fixed_point.py',ROOT/'src/hwnas_fpga/deploy/int8_reference.py',
            ROOT/'hardware/zcu104_dma_shell_v1/contracts/platform_contract.json',ROOT/'hardware/zcu104_dma_shell_v1/rtl/operator_benchmark_wrapper.v',ROOT/'hls_lut_builder/board_harness/configs/board_zcu104.yaml',
            ROOT/'hls_lut_builder/configs/candidate_kernels_detection_multiscale_640.yaml',ROOT/'hls_lut_builder/results/detection_multiscale_640_zcu104/hls.csv',ROOT/'hls_lut_builder/results/detection_multiscale_640_zcu104/vivado_downstream.csv']
    inputs+=list((ROOT/'tools/fpga_v0_1').glob('*.py'))+list((ROOT/'tools/fpga_v0_1').glob('*.cpp'))
    inputs += [p for p in ART.rglob('*') if p.is_file() and 'remote_results/results/' not in p.as_posix()]
    inputs += [p for p in OUT.rglob('*') if p.is_file() and p.name!='provenance_manifest.json']
    dump(OUT/'provenance_manifest.json',dict(spec_sha256=frozen,dataset='closure witnesses: synthetic seed42; existing regression suite additionally reads three train minibatches for forward/backward only, no optimizer or dataset mutation',
         artifacts={str(p.relative_to(ROOT)):sha(p) for p in inputs},
         parser_correction='Initial remote collector looked for a nonexistent completion string and conservatively emitted BLOCKED. Canonical tool_evidence reparses actual CSim done with 0 errors plus exit code, packet checks and binary equality; original raw collector JSON retained.',
         duplicate_transport_snapshot='remote_results/results is an earlier SCP snapshot; not used for canonical results'))
    print('Finalized:',len(matrix),'shape rows;',len(parity),'parity rows; 12 C-sims; 1 synthesis')


def write_documents(states,parity,frozen):
    (OUT/'ACTIVATION_CONTRACT.md').write_text('''# V0.1 Activation Contract

状态：**BLOCKER_FOR_NUMERICAL_PARITY**。本合同描述当前源码及待实现接口，不改变网络。

| 位置 | 软件（eval） | 当前 HLS | 结论 |
| --- | --- | --- | --- |
| stem、stage0 Conv | Conv→BN→ReLU | standard_conv_packed: APPLY_RELU=1→relu6_quant | 不等价；ReLU 无 6 上界 |
| MBConv expand PW | Conv→BN→ReLU6 | 累加器直接整数 clip[0,6] | 只有特定单位尺度才吻合；正常尺度需先 requant |
| MBConv DW3/DW5 | Conv→BN→ReLU6 | 同上 | 同上 |
| MBConv projection | Conv→BN→linear | INT32→clip[-128,127] | 缺输出尺度转换 |
| MBConv residual | projection+identity；无后置激活 | projection accumulator直接加INT8 residual再饱和 | 两项通常不同单位；V0.1六个active MBConv均无残差 |
| stage3 identity | identity | skip原样转发 | 必须保持尺度、布局和帧边界 |
| FPN lateral/output、head tower | Conv→BN→ReLU | APPLY_RELU=1→整数clip[0,6] | 不等价，不能忽略 |
| FPN upsample+Add | nearest复制后相加；无激活 | 相加后clip[-128,127] | 需共同尺度；量化饱和不是软件ReLU |
| prediction head | Conv+bias，linear原始logits | APPLY_RELU=0→clip[-128,127] | 不应加ReLU/ReLU6；缺requant |

来源：`src/hwnas_fpga/models/builder.py`、`models/detection.py`、
`hls_lut_builder/templates/{standard_conv_packed,mbconv,dw_conv,upsample_add_packed}.cpp.tmpl`、`include/common_types.h`。

未来每个Conv输出需显式 `activation_kind={linear,relu,relu6}`、输出scale/zero_point、整数上下界。
zero_point=0时 ReLU 下界0；ReLU6上界应由6/output_scale及确定的舍入/饱和规则导出，不能固定为整数6。
本轮仅定义该接口，没有修改共享HLS模板的数学语义。

真实vendor-header C-model反例见 `artifacts/fpga_v0_1_closure/20260914_v1/remote_results/packet_numeric_witness.log`：
整数16经relu6_quant得到6，而output_scale=0.5时实数ReLU6的量化上界应为12。
代表层C-sim零误差只验证现有整数裁剪语义；不能升级为软件FP32/校准INT8检测器等价。
''',encoding='utf-8')
    (OUT/'QUANTIZATION_CONTRACT.md').write_text('''# V0.1 Quantization Contract

状态：**BLOCKER_FOR_NUMERICAL_PARITY**。Q(x)尚未实例化；没有量化搜索或新位宽支持主张。
机器可读接口：`quantization_interface.json`。所有scale尚未由检测器校准数据绑定。

| 项目 | 当前硬件已实现 | 现有软件参考 / 缺口 |
| --- | --- | --- |
| activation / weight | signed ap_int<8> | 仅INT8已证实；4/6bit、mixed precision未证实 |
| accumulator / bias | ap_int<32> | bias应按sx·sw量化；需绑定folded参数与hash |
| zero point | 没有zero-point输入或减法 | 只有零中心算术；软件symmetric策略z=0 |
| 输出范围 | [-128,127] | canonical软件量化预留-128，用[-127,127]，冲突未消除 |
| scale表示 | 当前这些模板无scale参数 | fixed_point.py用正实scale→确定性整数分子/分母；分母最多2^30 |
| requantization | 缺失，直接累加→clip | 应对acc乘sx·sw/sy，再舍入、加zy、饱和、激活 |
| rounding | 不发生scale conversion，不能称已实现rounding | 软件nearest-even；负数/半整数行为需硬件落实 |
| overflow | ap_int<32>赋值/累加截位，溢出回绕 | conv2d_int_reference先INT64累加再INT32饱和；两者不同 |
| residual/FPN Add | 原整数相加后clip | 输入单位必须相同；不同scale需显式转换到输出尺度 |
| upsample | 整数倍nearest复制 | 复制保持scale；随后的Add仍需对齐 |

所依据的实现：`src/hwnas_fpga/deploy/{fixed_point,int8_reference,quantization,ptq_eval}.py`、
`hls_lut_builder/include/common_types.h`及上述模板。软件支持函数并不意味着HLS已接入。

推荐接口表达式（待硬件实现、非已完成配置）：

```
a = sum((qx-zx)*(qw-zw)) + qb
qb = round_even(b_folded/(sx*sw))
qy = clamp(round_even(a*(sx*sw/sy)) + zy, qmin, qmax)
qadd = clamp(round_even((qleft-zleft)*sleft/sout
                       +(qright-zright)*sright/sout) + zout, qmin, qmax)
```

Add须确定中间位宽及“宽累加后统一舍入”，不得隐式分别饱和再相加。
若约束左右scale和输出scale相同，原整数Add才是可用特例；校准必须显式验证此条件。
MBConv residual模板把projection accumulator和identity直接相加，单位尤其需要核对；冻结V0.1全部MBConv residual=false，因此该分支不是本轮实际网络的活动路径。

软件整图INT8解释器仍偏向单tensor输出，`run_integer_reference`要求(tensor,scale)，
当前function dispatch没有F.interpolate的nearest分支；candidate shape guard也不能直接当FX动态控制流处理。
需要为检测器建立支持Resize/Add及多个输出的显式图接口，不能复用分类器成功记录宣称完成。

## 本轮数值证据的边界

1. FP32：六个随机初始化候选，三种macro×全k3/全k5，固定seed42、固定非平凡BN统计量；不读数据、不校准。
   BN折叠与ONNXRuntime分别计算FP32 max/mean abs error，容差atol=2e-5、rtol=1e-4。
2. INT8：真实Vitis HLS C-sim执行未改数学语义的仓库模板。独立参考使用PyTorch CPU float64卷积计算有界整数和，
   输入[-3,3]、权重[-2,2]、bias[-10,10]；最大绝对整数和远小于2^31及2^53，因此该测试域内和可精确表示。
   在每个原语/MBConv子层执行模板规定的clip。upsample测试覆盖[-128,127]输入及加法饱和。
   这是明确的`INT8_LEGACY_RAW_ARITHMETIC`，等价于该测试的无scale转换单位算术，不是已校准的detector。
   软件参考和C-model输出均保存，使用INT32差值避免INT8溢出；exact-match ratio独立记录。
3. vendor-header helper反例：acc=16、ratio=1/2，当前HLS输出16而应为8；ReLU6(scale=.5)输出6而应为12；负饱和-128与软件-127不同。
   因此不能以C-sim通过掩盖完整数值合同的缺失。

## BN folding

复用统一接口 `fold_candidate_bn(candidate.eval())` → `(copied_model,pairs)`，内部调用既有
`ptq_eval.fold_batch_norms_inplace` / `torch.nn.utils.fusion.fuse_conv_bn_eval`；不改训练模型。
仅eval模式、frozen running mean/variance、Conv后紧邻BN且无分支共享该Conv输出时成立。

```
alpha[o] = gamma[o] / sqrt(running_var[o] + eps)
W_fold[o,...] = alpha[o] * W[o,...]
b_fold[o] = beta[o] + alpha[o] * (b[o] - running_mean[o])
```

无bias时b=0。groups/depthwise按输出channel缩放。覆盖stem/stage0、MBConv expand/DW/project、FPN lateral/output、head tower；prediction无BN。
V0.1这些位置均为串行Conv→BN，可fold。168条逐层FP32检查和20条预测输出检查详见`bn_layer_parity.csv`、`software_parity.csv`。
没有保留未折叠BN；训练态输入会被接口拒绝。尚未验证训练checkpoint经fold和quant后在HLS上的等价性。
''',encoding='utf-8')
    (OUT/'INTERFACE_CONTRACT.md').write_text('''# V0.1 Tensor / Export / Packet Interface

## Export

入口：`tools/fpga_v0_1/export_contract.py::export_candidate_predictions(candidate.eval(),path,example)`。
仅batch1、输入`image:[1,1,640,640]`、FP32/NCHW，opset13，导出prediction-only tuple；不将backbone/neck中间tuple误当输出。
ONNX graph metadata和同名JSON均保存output names、stride、shape、anchors和architecture。

| macro | ordered output names | strides | shapes |
| --- | --- | --- | --- |
| P3/P4/P5 | P3,P4,P5 | 8,16,32 | [1,21,80,80]; [1,21,40,40]; [1,21,20,20] |
| P2/P3/P4 | P2,P3,P4 | 4,8,16 | [1,21,160,160]; [1,21,80,80]; [1,21,40,40] |
| P2/P3/P4/P5 | P2,P3,P4,P5 | 4,8,16,32 | 上述四个尺度 |

三种macro各验证全k3和全k5，20个输出均通过ONNX checker/ONNX Runtime FP32 parity。
这覆盖全部unique kernel/shape实例，但未穷举144个组合；没有声称任意混合实例已逐图导出。
P2/P3/P4模型在stage5截断；导出没有P5伪输出。固定shape tracer warning符合明确的静态接口，不表示支持动态分辨率。
输出为raw logits；decode/NMS不在当前FPGA primitive合同内，保持现有软件后处理。

## Tensor ordering and parameter memory

软件输入/输出是NCHW；HLS按batch1的H、W、channel/segment递增，channel0放最低8bit。
标准Conv weights是连续`[Cout][Cin][Kh][Kw]`，与PyTorch OIHW相同；bias是`[Cout]`的INT32。
DW是`[C][Kh][Kw]`，从PyTorch`[C,1,Kh,Kw]`去掉单例维度。
MBConv weights按expand[EC,CI]、DW[EC,K,K]、projection[CO,EC]拼接；bias依次[EC]、[EC]、[CO]。
偏置单位必须是对应子层sx·sw；当前模板没有该尺度元数据。
head的21通道是3 anchors×7，anchor-major，单anchor为tx,ty,tw,th,obj,class0,class1，参照detection_trainer._reshape_prediction。
每个输出level单独保存，未来DMA descriptor需绑定level/stride/shape/layout/scale/offset/byte_length，不能靠buffer位置猜测level。

## 21-channel boundary

实际C-sim：96→21、20²及160²；OUT_TILE=8，最后计算tile仅5个有效channel，源码用`oc<OUT_CH`保护weights、bias与输出写入。
native输出PACK_CH=min(21,32)=21，即168-bit，每像素一个packet，21个字节均有效，**没有需要mask的额外native lane**。
HLS stream TLAST仅在整幅最后像素的最后packet为1；它不是每行TLAST的视频协议。输入TLAST/keep由当前primitive忽略，合法帧长度由固定shape决定。

额外的真实shared-helper C-model测试采用16-byte segment：21通道分两包，第二包有效5字节、11字节填零；roundtrip通过。
但`make_axis_segment_packet`仍设置TKEEP=TSTRB=0xffff，而逻辑有效掩码应是0x001f。补零不等于有效lane mask。
固定shape consumer可以忽略无效lane；generic DMA/byte-stream adapter不能据此推断逻辑长度。
本轮没有修改公共packing helper，防止未验证地改变既有consumer合同。

## 256-bit DMA integration gap

历史shell为256bit；native 168→256须用跨pixel byte repacking或显式padded-pixel布局，二者必须择一绑定并验证。
当前要求紧凑HWC时：

| level | valid output bytes | ceil DMA beats | last valid bytes |
| --- | --- | --- | --- |
| P2 160² | 537600 | 16800 | 32 |
| P3 80² | 134400 | 4200 | 32 |
| P4 40² | 33600 | 1050 | 32 |
| P5 20² | 8400 | 263 | 16 |

P5最后包需要TKEEP=0x0000ffff，或者显式传输8416 bytes并由descriptor剔除16 padding bytes。
历史`operator_benchmark_wrapper.v`强制FULL_BYTE_MASK并对非满mask报错；不能直接宣称它兼容紧凑8400-byte P5输出。
192/256等旧分类adapter实例不证明168/256和多head调度已验证。
剩余工作：选定字节布局、末包策略、256bit桥接RTL及backpressure/TLAST/TKEEP测试、PS buffer descriptor和cache flush/invalidate。
这属于未来Q/M接口准入条件，不是当前已完成board部署。

## Upsample and stream scheduling

20→40、40→80、80→160均为factor2；三个真实完整shape的96通道C-sim均验证HWC/segment顺序、输出数量、TLAST及整数饱和Add。
low_buffer分别38400、153600、614400 bytes；先读完整low帧，再消费lateral。
当前模板用整数除法OUT/LOW，泛化时必须断言LOW>0且OUT%LOW==0，本轮生成case增加此编译期断言。
不声称任意resize/bilinear支持。C-sim的unbounded stream不证明RTL有限FIFO、背压或整图调度无死锁。
''',encoding='utf-8')
    syn=next(s for s in states if s.get('synthesis_status'))
    bn=loadcsv(OUT/'bn_layer_parity.csv');ort=[r for r in parity if r['domain']=='FP32_ONNX_RUNTIME']
    hlsrows='\n'.join('| '+s['case']+' | '+s['status']+' | '+str(s.get('parity',{}).get('samples','—'))+' | '+('0 / 1.0' if s.get('csim_status') else '未测')+' |' for s in states)
    buf=loadcsv(OUT/'BUFFER_STATIC_ESTIMATES.csv')
    bufrows='\n'.join(f"| {r['combination']} | {r['all_active_kernel']} | {r['weights_bytes_INT8']} | {r['folded_bias_bytes_INT32']} | {r['largest_upsample_buffer_bytes']} | {r['eager_lifetime_upper_bound_bytes']} |" for r in buf)
    (OUT/'FPGA_V0_1_COMPATIBILITY_CLOSURE.md').write_text(f'''# FPGA V0.1 Compatibility Closure

整理日期：2026-09-15；本轮工具报告时间：2026-09-14（服务器本地时间）。
**Overall Status: BLOCKED**，具体为`BLOCKER_FOR_NUMERICAL_PARITY`与未闭环DMA合同。
本轮完成证据归档；software/export/BN及代表原语取得新证据，不代表完整detector FPGA-ready。

## Scope / reproducibility

- 冻结空间144 architectures；YAML SHA256 `{frozen}`，与现有adapter固定hash一致。本轮未修改该文件。
- 沿用640输入、channels、anchors、三种output combinations、k3/k5；Q/M仍非search变量；active_hardware_constraints=null。
- 没有训练实验、校准、NAS、144结构综合、整网P&R、board运行或功耗测量。closure数值测试为seed42合成数据；既有回归套件另读取三个train minibatch做forward/backward，不执行optimizer、不修改数据。
- 软件新入口：`tools/fpga_v0_1/export_contract.py`；复用现有candidate builder、BN folding和HLS render_template。入口独立于冻结baseline源码清单，保持清单不变。
- 原始输入/ONNX/参考/C-model输出：`artifacts/fpga_v0_1_closure/20260914_v1/`。
- 本轮服务器独立目录：`/home/jiajia/experiments/fpga_v0_1_closure_20260914_v1`。凭据只通过本地Credential Manager helper使用。
- `provenance_manifest.json`绑定配置、源码、生成case、二进制、日志、报告hash；历史dirty worktree未清理。

## 1. Platform evidence

| 字段 | repository实际证据 | 分类 / 结论 |
| --- | --- | --- |
| board | board_zcu104.yaml与hardware/zcu104_dma_shell_v1/contracts/platform_contract.json：ZCU104 | historical_reference；本轮用户指定同板，用于兼容性case |
| FPGA part | 同上xczu7ev-ffvc1156-2-e；本轮D_head21_P5_csynth报告再次确认 | 本轮单算子target已验证，不是整网deployment target闭环 |
| tool | 旧detection YAML指向2022.2；本轮vitis_hls_version.log和csynth.rpt为2022.2 build3670227 | HLS live verified；未启动Vivado/P&R |
| target clock | historical shell200MHz，当前run.tcl与报告5ns | target clock；实际route Fmax未测 |
| DMA width | historical platform contract及build_bd.tcl：256bit MM2S/S2MM | historical_reference；168bit head接入未验证 |
| device capacity | 本轮csynth available LUT230400/FF460800/DSP1728/BRAM18 624/URAM96 | part capacity；BRAM36=312；不是应用可用预算 |
| shell reservation | board YAML为PLATFORM_CONTRACT_PENDING_BUILD；所查contract未绑定V0.1 shell资源预留 | UNKNOWN；不得默认0 |
| deployment interface | 根目录deploy/不存在；实际部署代码在src/hwnas_fpga/deploy/ | 单tensor分类export/int8路径不能直接替代多head检测合同 |

旧detection `hls_lut_builder/results/detection_multiscale_640_zcu104/{{hls.csv,vivado_downstream.csv}}`
及`artifacts/detection_multiscale_640`仍是P3/P4/P5、24-channel head的historical_reference。
本轮不把旧表里的latency/resource搬作当前21/P2结果；也不把分类shell成功材料升级为检测器部署目标。

## 2. Unique shape catalog and MBConv

`V0_1_SHAPE_CATALOG.csv`含54条unique operator/shape/activation合同，组合/角色合并；包含12个MBConv整体实例及其不重复Conv/DW子层。
整体MBConv与子层是两个观察层，资源汇总时不能重复相加。六个软件图覆盖全部独特kernel/shape，但未枚举144个整图。

| stage | Cin→expanded→Cout | H/W→out | stride | k/padding | residual | 本轮HLS证据 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 16→48→24 | 320→160 | 2 | 3/1，5/2 | no | k3/k5完整shape C-sim、raw integer parity |
| 2 | 24→72→32 | 160→80 | 2 | 3/1，5/2 | no | exact-shape vendor-header compile；HLS_TEMPLATE_AVAILABLE |
| 4 | 32→96→64 | 80→40 | 2 | 3/1，5/2 | no | 同上 |
| 5 | 64→192→96 | 40→40 | 1 | 3/1，5/2 | no | 同上 |
| 6 | 96→288→160 | 40→20 | 2 | 3/1，5/2 | no | 同上；P2/P3/P4不含此stage |
| 7 | 160→480→320 | 20→20 | 1 | 3/1，5/2 | no | 同上；P2/P3/P4不含此stage |

所有12个MBConv实例vendor-header C++语法编译通过。固定EXP_TILE/OUT_TILE=8，本空间expanded/Cout都整除8；仍保留ec/oc边界保护。
内部分段PACK最多32channel，stage2 expanded72会有内部packet尾lane且按固定总channels解包。
MBConv外部仍是Cin×8/Cout×8整pixel端口，max_axis_bits仅限制内部expanded stream；stage7输出2560bit，不能当256bit DMA直连。
line buffer为EC×K×(W+2padding)，合法padding、stride边界由完整stage1 DW3/DW5及MBConv C-sim实测。
stage2/4/5/6/7仍归`SUPPORTED_BY_TEMPLATE_ONLY`（另有exact-shape C++编译证据），不写成HLS_CSIM/HLS_SYNTH verified。

## 3. P2 closure

- stage1给出24×160×160，lateral24→96、3×3 neck/output与head tower96→96、head96→21、80→160+Add全部有软件和ONNX证据。
- B_P2_lateral、C_P2_tower、D_head21_P2、F_upsample80_160全部完成完整shape真实C-sim，INT8原语参考零误差。
- neck/output与head tower为同shape/同激活原语，C case代表该等价类；不是两层串联的整图RTL验证。
- P2所有这些层尚无本轮综合资源/route/board结果；FP32模型等价被ReLU与requant合同阻塞。

## 4. Export / BN / numeric

`INTERFACE_CONTRACT.md`固定3/4-head名称、顺序、shape、stride、anchor、layout和21-channel memory/packet合同。
六个ONNX图均通过checker及CPU ONNXRuntime，20个输出逐一对照，最大FP32误差 {max(float(r['max_abs_error']) for r in ort):.9g}。
BN folding复用既有工具，168条逐层检查最大FP32误差 {max(float(r['max_abs_error']) for r in bn):.9g}，20个预测输出也通过。
软件测试使用非平凡running mean/var、gamma/beta，完整参数和输出可追溯。

`ACTIVATION_CONTRACT.md`逐位置区分ReLU/ReLU6/linear；`QUANTIZATION_CONTRACT.md`明确现有INT8/INT32事实、软件尺度规则和硬件缺口。
`parity_results.csv`分FP32_BN_FOLD、FP32_ONNX_RUNTIME、INT8_LEGACY_RAW_ARITHMETIC三个域，避免混用误差。
HLS的12个原语case全部exact-match=1.0只证明模板现有整数数学，不证明folded/校准detector numerical parity。

## 5. Representative tool evidence

全部case目标part为XCZU7EV、clock5ns。以下status仅描述实际工具层，不替代矩阵中的calibrated_detector_numerical_status。

| case | implementation status | output samples | INT8 max error / exact ratio |
| --- | --- | --- | --- |
{hlsrows}

唯一新HLS synthesis：D_head21_P5，96→21、20²、1×1、linear、PI8/PO8、最大AXIS256。
真实报告：latency142324 cycles（5ns下0.71162ms）、top invocation interval142325 cycles、DSP8、LUT5939、FF1233、BRAM18=16、URAM0；estimated clock3.490ns。
这些是**真实HLS工具报告中的估计值**，不是route实测或board测量；top interval不是内部loop II=1。
其余case latency/II/DSP/LUT/FF/BRAM/URAM留空，未用C-sim wall time、MACs或历史值填充。
完整command/exit code/source hash及原始日志见`representative_cases/tool_evidence.json`、`case_results.csv`。
12次C-sim完成，22个exact-shape编译完成，1次单算子综合；没有整网P&R。

原始remote collector完成字符串写错，保守误标若干BLOCKED。最终解析器读取实际`CSim done with 0 errors.`、exit0、packet检查和独立二进制相等后重新分类。
原始JSON仍保留，规范结果是`representative_cases/tool_evidence.json`；未重跑或伪造报告消除状态差异。

## 6. Static buffer / BRAM analysis

所有以下数值均为**static estimate**。INT8 feature/weight按1byte、folded bias按4bytes。
完整模型不是全部装片上；这里不选择mapping、不将理论容量换算冒充BRAM综合结果。

统一公式：feature=C×H×W；dense/DW line=Cin×K×(W+2P)；MBConv line=EC×K×(W+2P)；
Conv weights=Cout×Cin×K²；DW weights=C×K²；MBConv weights=EC×Cin+EC×K²+Cout×EC；upsample低帧=C×Hlow×Wlow。
stage-level line buffer见catalog；本轮固定profile不搜索分区。片上权重端口/分区的额外复制与BRAM碎片未计。

| macro | all-active k | INT8 weights bytes | INT32 biases bytes | largest upsample buffer bytes | conservative eager activation lifetime bound bytes |
| --- | --- | --- | --- | --- | --- |
{bufrows}

activation lifetime模型：batch1、无autograd、完整tensor、不做tile/reuse；保留所选backbone输出，
逐MBConv计input+expanded+DWoutput+projection的保守重叠；FPN计raw+lateral+新增fused+neck outputs+最大nearest/tower临时张量+prediction outputs，取两阶段上界。
这是对明确执行假设的保守存储界，不是实际peak profiler结果，更不是未来streaming mapping最小容量。
权重数由实际model Conv2d参数计数，k3/k5给出两个端点；组合中inactive尾部已剔除。

20→40低帧38400bytes→理想ceil(bytes/2304)=17 BRAM18；40→80为153600→67；80→160为614400→267。
BRAM36理想容量以4608bytes换算；BRAM18和BRAM36不得混写。P2的96×160²完整单feature即2457600bytes，理想1067 BRAM18超过624器件容量，
说明“全帧都放BRAM”假设不可直接沿用，需要后续DDR/reuse/streaming合同；它并不证明所有mapping不可行。
partition、FIFO、权重驻留、端口约束、shell及route开销未计入静态换算。

## 7. Status matrix and budgets

`OPERATOR_COMPATIBILITY_MATRIX.csv`按每条unique合同分别保留software_support、EXPORT_VERIFIED、implementation_status、raw_integer_numerical_status与calibrated_detector_numerical_status。
不以EXPORT_VERIFIED顶替HLS_CSIM，不以C-sim顶替synthesis，也不以原始整数parity顶替量化网络parity。
状态词仅用于其所属证据层。未运行过的模板实例明确为HLS_TEMPLATE_AVAILABLE / SUPPORTED_BY_TEMPLATE_ONLY。

旧latency120ms、energy375mJ、power15W、DSP1209、BRAM700、LUT161280仍为`legacy_unverified_budget`。
本轮虽确认part容量及代表case clock，没有确认V0.1完整shell预留和应用约束；不得成为NAS active constraints。

## 8. Remaining FPGA blockers

1. HLS缺完整requant、ReLU/scaled-ReLU6区分、symmetric范围与overflow策略一致性；需要用folded权重和显式scale验证数值。
2. FPN多输入Add需共同scale/输出scale合同；detector integer reference需支持Resize及多个有名称输出。
3. native168bit head、MBConv多种宽AXIS与256bit DMA桥接未验证；21-channel P5末包只有16有效bytes，与历史FULL_BYTE_MASK shell冲突。
4. 缺绑定当前V0.1的weight/bias装载、tensor descriptor、multihead调度、有限FIFO/背压/activation lifetime及shell资源预留合同。
5. 本轮只综合1个原语；其余shape的资源/时序测量缺失。改变数值/接口实现后需重新绑定source/profile并测量，旧原语成本不可直接作为正式V0.1模型。

## 9. Next hardware step

**Can we start building a measured latency/resource model for V0.1? CONDITIONAL.**
可以立即建立测量框架与按source hash、shape、numeric contract、interface、固定profile、tool/part/clock索引的case数据库；本轮唯一新综合值只进入legacy-raw-arithmetic profile。
先实现并验证固定INT8数值合同与168/256bit及multioutput接口，再对代表shape补HLS synthesis和必要的独立route资源证据。
只有绑定新合同的成本模型才能服务后续Q(x)/M(x,q)及hardware-aware search；当前不启动正式联合搜索。

## Commands

最终验证：`tests/test_sonar_candidate_shapes.py` 37 passed，包含冻结baseline源码/config清单检查。
`validate_closure.py`额外核对6个ONNX图、54条shape、52条parity、22个vendor-header编译、12个C-sim、1个synthesis及训练态拒绝行为；结果见`validation.json`。
新入口已放在独立tools目录；baseline的42个源码文件及其manifest保持不变。

本地cwd为repository root，Python使用`.venv_cuda/Scripts/python.exe`：
```
tools/fpga_v0_1/software_witness.py
tools/fpga_v0_1/build_cases.py
tools/fpga_v0_1/catalog.py
tools/fpga_v0_1/remote_bridge.py launch
tools/fpga_v0_1/supplement.py
tools/fpga_v0_1/remote_bridge.py fetch
tools/fpga_v0_1/finalize.py
tools/fpga_v0_1/validate_closure.py
```
launch会要求新的远程目录，避免覆盖历史。本轮原始bundle及supplement bundle均保留；复现需使用新run目录，而不是覆盖本轮ONNX证据。
packet helper命令：`g++ -O2 -std=c++14 -Icases/D_head21_P5 -I/opt/Xilinx/Vitis_HLS/2022.2/include packet_witness.cpp -o packet_witness`，随后运行binary；只是vendor-header host C-model，非额外HLS synthesis。
''',encoding='utf-8')


if __name__=='__main__':main()
