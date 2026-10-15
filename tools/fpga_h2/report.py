"""Evidence-only H2 report; failure of any required witness closes the H3 gate."""
import csv,hashlib,json,re
from pathlib import Path
import numpy as np
from build import ART,OUT,HASHES,ROOT

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def writecsv(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
    r2=ART/'remote_results_attempt02';r3=ART/'remote_results_attempt03'
    numeric=[];passed=True
    for folder in sorted((OUT/'representative_cases').iterdir()):
        if not folder.is_dir():continue
        name=folder.name;meta=json.loads((folder/'case.json').read_text());r=r3 if name=='upsample_add' else r2
        log=r/(name+'.log');text=log.read_text(errors='replace')
        status=next(s for s in json.loads((r/'status.json').read_text()) if s['case']==name)
        a=np.fromfile(folder/'reference.bin',dtype=np.int8).astype(np.int32);b=np.fromfile(r/(name+'_output.bin'),dtype=np.int8).astype(np.int32)
        assert a.shape==b.shape and len(a)>0
        diff=np.abs(a-b);indices=np.flatnonzero(diff);ok=len(indices)==0 and status['returncode']==0 and 'CSim done with 0 errors.' in text
        passed &= ok
        # Verify that the bound source bundle still matches the current generated implementation.
        sources={p.name:sha(p) for p in folder.iterdir() if p.suffix in ['.cpp','.h','.tcl']}
        assert sources==meta['source_files']
        numeric.append(dict(case=name,parity_kind='NUMERIC_CONTRACT_V1_PARITY',status='PASS' if ok else 'BLOCKED',samples=len(a),max_abs_error=int(diff.max()),
             mismatch_count=len(indices),first_mismatch=int(indices[0]) if len(indices) else '',exact_match_ratio=float(np.mean(a==b)),
             **HASHES,source_hash=meta['source_hash'],tool_version='Vitis HLS 2022.2',FPGA_part='xczu7ev-ffvc1156-2-e',clock_ns=5,tool_stage='CSIM',evidence=str(log.relative_to(ROOT))))
    assert len(numeric)==11
    writecsv(OUT/'numeric_parity.csv',numeric)
    rtl=(r3/'xsim.log').read_text(errors='replace')
    packet_cases=json.loads((ART/'rtl/cases.json').read_text());packet=[]
    rtlhash=hashlib.sha256((ART/'rtl/axis_repack_v1.sv').read_bytes()+(ART/'rtl/tb.sv').read_bytes()).hexdigest()
    for c in packet_cases:
        ok=f'RTL_PACKET_PASS {c["name"]} native={c["native_words"]} dma={c["dma_words"]}' in rtl
        ok &= 'ALL_RTL_PACKETS_PASS' in rtl and 'Fatal' not in rtl
        passed &= ok
        packet.append(dict(case=c['name'],status='PASS' if ok else 'BLOCKED',scope='RTL_BIDIRECTIONAL_WITH_BACKPRESSURE',real_bytes=c['real_bytes'],padded_bytes=c['padded_bytes'],
            mismatch_count=0 if ok else '',exact_match_ratio=1.0 if ok else '',tail='frame_zero_padding_full_keep',**HASHES,source_hash=rtlhash,tool_version='Vivado xsim 2022.2',
            FPGA_part='xczu7ev-ffvc1156-2-e_target_not_synthesized',clock='testbench10ns; synthesis target5ns',tool_stage='RTL_SIM',evidence=str((r3/'xsim.log').relative_to(ROOT))))
    writecsv(OUT/'packet_parity.csv',packet)
    graphs=json.loads((OUT/'integer_graph_results.json').read_text())
    graphsource=hashlib.sha256(b''.join((Path(__file__).parent/f).read_bytes() for f in ['numeric.py','integer_reference.py'])).hexdigest()
    for g in graphs:g.update(source_hash=graphsource,tool_version='NumPy integer runtime',FPGA_part='xczu7ev-ffvc1156-2-e_target_only',clock='not_applicable_software')
    writecsv(OUT/'integer_graph_results.csv',graphs)
    validation=json.loads((OUT/'validation.json').read_text());passed &= validation['status']=='PASS'
    assert sha(ROOT/'configs/search_space_sonar_v0_1.yaml')=='ebb1708b394af20258c920eb29204d5cc9504f5908fdfdae62a62d78042eb7c4'
    status='PASS' if passed else 'BLOCKED'
    write_docs(status,numeric,packet,validation)
    payload=dict(overall_status=status,H3_gate='YES' if passed else 'NO',**HASHES,numeric_cases=len(numeric),RTL_packet_cases=len(packet),integer_graphs=3,
        scope='contract closure only; no calibrated accuracy claim, synthesis campaign, whole-network RTL/route, board deployment or NAS',
        canonical_runs={'numeric':'attempt02 except Add attempt03','RTL':'attempt03'},
        artifacts={str(p.relative_to(ROOT)):sha(p) for base in [OUT,ART,Path(__file__).parent] for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='closure_manifest.json'})
    (OUT/'closure_manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(status,'H3 gate:',payload['H3_gate'],'numeric',len(numeric),'RTL',len(packet))

def write_docs(status,numeric,packet,validation):
    (OUT/'NUMERIC_CONTRACT_V1.md').write_text(f'''# NUMERIC_CONTRACT_V1

版本：1。状态：{status}。规范hash：`{HASHES['numeric_contract_hash']}`。
hash对象是`numeric_contract_v1.json`按sort_keys、无空格序列化的UTF-8 JSON；不是本说明文档。变更任何正式算术规则须新版本、新hash、新parity。

## 唯一正式规则

| 项目 | V1已实现规则 |
| --- | --- |
| activation / weight | signed INT8，唯一范围[-128,127] |
| zero point | 固定0，padding值为整数0 |
| accumulator | signed INT32，MAC和bias加法按2^32模回绕；不使用另一套INT32饱和 |
| bias | little-endian INT32；round-nearest-even(b_fold/(sx·sw))，溢出先饱和INT32后存储 |
| scale | 每tensor/per-layer weight采用正的2^e；e为INT32、单个scale的e在[-24,16]；不是quantization search变量 |
| multiplier / shift | sx·sw/sy=M/2^S；M整数[1,2^31−1]、S整数[0,31]；不能表示的比例拒绝 |
| requant intermediate | signed INT64；合法INT32×M的积不会溢出INT64 |
| rounding | 唯一nearest-even；正负对称处理绝对值商/余数，半整数趋向偶数 |
| saturation | requant舍入后唯一clip[-128,127]；小值underflow到0 |
| activation | requant后分别执行linear、ReLU、scaled-ReLU6；linear不添加ReLU |

权重scale是固定可复现power-of-two包格式；本轮用folded权重最大绝对值选择容纳范围的指数。activation尺度是显式固定测试配置，不是数据校准或搜索结果。
后续已训练detector必须另行校准/评估；本轮不宣称量化精度与FP32相等，也不支持INT4/INT6/mixed precision。

## 已实现公式

```
acc32 = wrap_twos_complement_32(sum(qx*qw)+bias32)
product64 = int64(acc32)*M
rounded = nearest_even(product64 / 2^S)
q = clip(rounded,-128,127)
linear(q) = q
ReLU(q) = max(q,0)
q6 = clip(nearest_even(6 / output_scale),0,127)
ReLU6(q) = clip(q,0,q6)
```

Python入口：`tools/fpga_h2/numeric.py::requantize_int32_to_int8`；HLS同义实现：`numeric_v1.h::requant/finish`。
禁止调用者指定truncate或[-127,127]：软件接口会拒绝。HLS仅消费已验证package中的参数，范围是loader前置条件。
scale值为2的幂，参数量化在FP64上执行nearest-even；完整forward自输入INT8起仅用整数运算。
NumPy Conv采用INT64精确和再wrap32，与逐MAC模2^32同余；V0.1固定shape的最大和远小于INT64容量。

## FPN / residual Add共享规则

```
common_scale = max(scale_A,scale_B)
A_common = clip(round_even(A*scale_A/common_scale),-128,127)
B_common = clip(round_even(B*scale_B/common_scale),-128,127)
sum32 = INT32(A_common) + INT32(B_common)
out = clip(round_even(sum32*common_scale/output_scale),-128,127)
```

这是明确的分阶段舍入合同，不宣称与“一次宽加法后舍入”相同。Add默认linear，无隐含激活。
nearest upsample先复制整数并保持原scale，再执行上述Add。两输入尺度不同的400组边界已测试。
residual复用完全相同的`aligned_add`/`h2::add`，先将projection经自身requant成为INT8，再与identity对齐相加。
冻结V0.1的六个MBConv均无residual；HLS MBConv case用static_assert拒绝意外启用旧残差。
有残差的组合必须用显式V1 aligned_add节点，不会回退到旧的accumulator+INT8直接相加。

## Evidence

`numeric_parity.csv`含10个完整shape原语case及1349向量的boundary case。
包含正负、halfway、INT8上下界、large multiplier、shift31、underflow、INT32正负回绕、ReLU、q6=12与不同Add尺度。
全部以`NUMERIC_CONTRACT_V1_PARITY`记录；旧`RAW_INTEGER_PARITY`不进入本合同证据。
代表权重来自seed42软件模型的非平凡BN统计量→既有BN folding工具→量化→binary package；没有伪造folded/quantized参数。
原始folded_state.pt、INT8 weights、INT32 biases、scale metadata、输入/参考/HLS输出均保留。
''',encoding='utf-8')
    (OUT/'INTERFACE_CONTRACT_V1.md').write_text(f'''# INTERFACE_CONTRACT_V1

版本：1。规范hash：`{HASHES['interface_contract_hash']}`，按对应JSON的canonical serialization生成。

## Tensor / stream / DDR

- 软件整数参考为batch1 CHW；预测输出[21,H,W]，batch维度固定1，不是动态batch接口。
- DDR/native stream为连续HWC bytes；row→column→channel递增，channel0为最低8bit，signed two's complement。
- head每pixel21bytes/168bit。通道为anchor-major，3×[tx,ty,tw,th,obj,class0,class1]；prediction始终linear raw logits。
- DMA beat为32bytes/256bit；byte0在低8bit。参数和DDR均little-endian。
- tensor DDR起点64-byte对齐；帧真实数据紧凑存放，仅尾部补0至32bytes倍数；descriptor之间可有64-byte对齐间隙，间隙不计入DMA长度。

## 正式选择：连续repack + 帧末full-beat padding

逐pixel padding方案每21bytes占32bytes，带宽增加11/21=52.38%，不采用。
连续repack仅帧末补齐：P2/P3/P4 overhead为0；P5为16/8400=0.190476%。
P5 real_bytes=8400、padded_bytes=8416、263 DMA beats；最后16bytes为显式0。
所有DMA beat TKEEP/TSTRB=0xffffffff，包括P5最后beat；TLAST仅在最后beat。
PS/DMA使用padded_bytes传输，消费者按real_bytes裁切，不能将padding当成logit或pixel。
这保持历史FULL_BYTE_MASK shell约束，不需要在H2把它改成partial-TKEEP shell。

## 真正的双向adapter

实现：`tools/fpga_h2/axis_repack_v1.sv`。参数IN_BYTES/OUT_BYTES/REAL_BYTES；实例21→32为168→256，32→21为反向。
同实现支持16/24/32/64/96/160/320-byte native widths（最高2560bit）；不限于head。
容量固定IN_BYTES+OUT_BYTES bytes，单缓冲寄存器，显式ready/valid背压，无无限队列。
输出阻塞期间buffer/data/last稳定；输入只有ready时接收。收满REAL_BYTES后关闭该帧输入，发完补齐尾包后清空计数并接受下一帧。
反向通过REAL_BYTES丢弃DMA尾padding；要求real_bytes是native word width的整数倍。输入TLAST长度错误置error。
所有输入假定遵守full-mask、native完整word协议；DMA padding是已声明的唯一例外，按REAL_BYTES裁切。
11组Vivado xsim测试同时检查中间DMA每字节、TKEEP/TSTRB/TLAST、最终native tensor、容量上限及随机source/sink停顿。
RTL仿真clock10ns只用于功能测试；目标实现clock5ns，未宣称此RTL已达到5ns时序。

## Finite FIFO / ownership / scheduling

参考实现跨原语边界固定FIFO深度4 packets；MBConv内部expanded/depthwise stream的HLS pragma也是depth4。
bridge自身使用上述固定byte capacity，不借助无限FIFO；RTL已验证ready/valid和容量。
HLS C-sim的库队列只作为功能测试载体，日志中的预装载队列深度不作为硬件缓冲/无死锁证据。
H2另有真实数据的bounded queue test：low/lateral/output均最多4包，遇满停producer，并注入consumer停顿，最终与HLS输出一致。

| 路径 | 固定reference contract |
| --- | --- |
| 80→160 upsample-add | low_frame 614400bytes由caller分配在DDR，HLS显式m_axi scratch；先将low输入写满，再消费lateral；scratch写完后只读，结束后归还owner |
| P2 96→96 3×3 tower | 行缓存96×3×162=46656bytes，加独立权重buffer；输入从DDR逐row/pixel进入，输出背压向上游传播；不分配96×160²片上feature |
| MBConv | acyclic expand→DW→project，有限depth4，允许stall；V0.1没有回馈残差环；不能从C-sim推断RTL吞吐 |
| multihead output | 按descriptor逐head独立DMA，上一head TLAST及DMA完成后再发布下一head；所有tensor结果可在DDR存储 |

全P2 feature 2457600bytes允许驻留DDR；不默认驻留BRAM。H2没有tile/search，也不指定未来最优mapping。
合法性要求公平consumer最终拉高ready，DDR操作最终完成；永久外部stall不保证完成。
frame boundary为固定tensor最后packet，不是每行TLAST。shape/REAL_BYTES改变需要重新绑定descriptor和adapter实例。

## Weight / bias / scale loading

同一exporter生成graph.json、weights.bin、biases.bin、layers.bin；Python integer reference和HLS loader消费同一份package。
weights为signed INT8 OIHW，DW为[C,1,K,K]（HLS去掉单例维度）；每layer weight/bias起点64-byte对齐。
bias为little-endian signed INT32 Cout项。MBConv加载expand→DW→project三个layer ID，依各record offset顺序装入原语数组。
layers.bin每layer固定32个little-endian INT32 words（128bytes），序号与layer_id一致：

```
0 id; 1 Cin; 2 Cout; 3 H; 4 W; 5 K; 6 stride; 7 padding; 8 groups;
9 input_scale_exp; 10 weight_scale_exp; 11 output_scale_exp;
12 multiplier; 13 shift; 14 activation(0 linear/1 ReLU/2 ReLU6); 15 q6;
16 weight_byte_offset; 17 bias_byte_offset; 18 weight_count; 19 bias_count;
20 schema_version=1; 21..31 reserved=0.
```

Add exporter从graph node显式尺度生成add_params.bin：6个little-endian INT32，A multiplier/shift、B multiplier/shift、output multiplier/shift。
HLS Add loader实际读取该binary，不在HLS端重新估计scale。参数与input/output exponent同存case.json，绑定graph metadata hash。
Python integer forward验证binary hash与JSON/record一致；HLS case loader核对record的实际shape再加载weights/bias/params。
loader前置条件是正multiplier、合法shift/activation/q6、shape与生成top一致、hash匹配；未验证package不得启动硬件。
本轮HLS加载是C-sim host→参数数组，尚不宣称完成PS AXI-BRAM实机加载；host-side ABI现已明确，可作为H3测量夹具输入。
''',encoding='utf-8')
    schedules=json.loads((OUT/'multihead_descriptors.json').read_text())
    rows=[]
    for graph,events in schedules.items():
        for e in events:rows.append(f"| {graph} | {e['name']} | {e['address']} | {e['real_bytes']} | {e['padded_bytes']} | {e['expected_dma_beats']} |")
    (OUT/'MULTIHEAD_OUTPUT_CONTRACT_V1.md').write_text('''# MULTIHEAD_OUTPUT_CONTRACT_V1

固定schedule：present levels按increasing stride；每head独立DMA transaction，不通过head index猜level。
descriptor字段：name,stride,channels,height,width,anchors,real_bytes,padded_bytes,ddr_offset。
输出metadata来自冻结model.levels/anchor_sizes；不存在的P5不生成descriptor。
DDR基址由调用者提供，必须64-byte对齐且为已分配PS DDR buffer；本轮reference用base=0作为相对地址，不把0声明为实机物理地址。
实际地址=base+ddr_offset；next_offset=align64(offset+padded_bytes)，DMA长度=padded_bytes。
completion：最后beat握手且TLAST正确、已发送预期beats且DMA完成；所有head完成才是detector-output-complete。
PS启动前flush输入/参数，完成后invalidate输出；具体PS runtime与board测量属于后续工作。
`interface.schedule_outputs`是可执行reference scheduler；三图DDR回读按real_bytes裁切后bit-exact。

| Graph | semantic head | relative DDR offset | real bytes | padded bytes | DMA beats |
| --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(rows)+'\n\n完整descriptor与顺序见各graph.json及multihead_descriptors.json。\n',encoding='utf-8')
    (OUT/'PLATFORM_CONTRACT_V1_DRAFT.md').write_text('''# PLATFORM_CONTRACT_V1_DRAFT

| 字段 | 值 | evidence classification |
| --- | --- | --- |
| board | ZCU104 | historical_reference：board_zcu104.yaml / platform_contract.json；本轮用户指定；未进行board登录探测 |
| part | xczu7ev-ffvc1156-2-e | historical_reference并由本轮全部HLS set_part日志live_verified（工具目标，不是实机枚举） |
| Vitis HLS | 2022.2 | live_verified：remote_results_attempt02/hls_version.log、各CSIM日志 |
| Vivado / xsim | 2022.2 | live_verified：attempt03/vivado_version.log、xvlog/xelab/xsim.log |
| target clock | 5ns | live_verified：HLS create_clock；不是route timing pass |
| DMA | 256bit | historical_reference shell；本轮256bit adapter RTL功能live_verified；未board DMA |
| shell resource reservation | UNKNOWN | 没有本轮合同绑定的shell route报告；不得设0 |
| device capacities | LUT230400、FF460800、DSP1728、BRAM18 624、URAM96 | historical_reference，来自上一轮实际part HLS报告；不是应用预算 |
| board/runtime/power | UNKNOWN | 本轮未测 |

本轮不改写旧shell，不把draft变成部署完成声明。旧120ms/375mJ/15W/DSP1209/BRAM700/LUT161280继续legacy_unverified_budget。
active_hardware_constraints仍为空。H3可测合同绑定原语；正式整网可行性准入仍需shell、调度及route/board证据。
''',encoding='utf-8')
    nrows='\n'.join(f"| {r['case']} | {r['samples']} | {r['max_abs_error']} | {r['mismatch_count']} | {r['exact_match_ratio']} |" for r in numeric)
    (OUT/'FPGA_H2_CONTRACT_CLOSURE.md').write_text(f'''# FPGA H2 — Numerical and Interface Contract Closure

日期：2026-09-16。**Overall Status: {status}**。**H3 gate: {'YES' if status=='PASS' else 'NO'}**。
此PASS只表示进入contract-bound measured FPGA cost database的数值/接口条件满足，不表示完整detector FPGA-ready。

## Scope and immutable baseline

Search Space V0.1仍为144 architectures，SHA256 ebb1708b394af20258c920eb29204d5cc9504f5908fdfdae62a62d78042eb7c4。
冻结baseline源码/config清单检查通过；新增代码全部在tools/fpga_h2，旧closure artifacts不覆盖。
没有NAS、quantization/mixed precision/PE/SIMD/tiling搜索、SonarOp、144结构综合、整网P&R、训练或board部署。
本轮只使用seed42合成输入和随机初始化模型的非平凡BN统计量，packet测试seed43；没有读取数据集或宣称检测精度。

## Numerical closure

正式规则与hash见NUMERIC_CONTRACT_V1.md及numeric_contract_v1.json：INT8[-128,127]、zero point0、power-of-two scales、nearest-even、INT32 wrap、INT64 requant product、独立ReLU/scaled-ReLU6/linear。
FPN/residual共用stage-aligned Add；明确输入/common/output尺度与各次舍入/饱和位置。
Python runtime是真正NumPy整数卷积/加法，不以浮点卷积冒充integer golden reference。
模型先由既有BN folding工具处理，再导出同一binary metadata package；HLS loader实际消费对应folded/quantized权重与scale。

| case | samples | max abs error | mismatch count | exact ratio |
| --- | --- | --- | --- | --- |
{nrows}

全部为NUMERIC_CONTRACT_V1_PARITY。boundary包含1349项，其中400项覆盖FPN/residual同一Add逻辑的不同scale与signed输入。
canonical HLS证据：attempt02的9个算子及boundary；upsample_add以attempt03的binary Add loader为准。
输出差分使用INT32防止INT8减法溢出。first_mismatch为空代表无差异，不是未检查。
这些测试建立精确算术证据，不证明训练模型量化后精度；代表输入有7至73种不同整数值，非全零恒等测试。

## Integer detector graphs

- A：P3/P4/P5，全k3，完整forward。
- B：P2/P3/P4，全k3，完整forward；无P5。
- C：P2/P3/P4/P5，stage1/4/6用k5，其余active MBConv用k3，完整forward。

每个输出是21-channel INT8 tensor。graph.json绑定每个edge shape/scale、node、layer ID、folded参数hash及weight/bias/layers binary hash。
`integer_graph_results.csv`记录10个命名输出；再次从binary package执行，所有保存输出一致，三图DDR roundtrip也一致。
完整forward涵盖Conv/PW、DW3/5、MBConv、ReLU/ReLU6、nearest、scale-aligned Add、linear head。
普通MBConv residual在V0.1无活动实例；统一aligned_add已有真实HLS边界测试，未将旧残差算术保留为合法备选。

## Interface closure

正式选择连续byte repack + 帧末zero padding。168→256与反向bridge是真正SV RTL；TKEEP/TSTRB始终full mask，与历史shell策略一致。
P5 8400→8416bytes，末包16有效+16零padding；其他head无需额外DMA padding。
11个双向RTL case覆盖21-channel的20/40/80/160²，以及16/24/32/64/96/160/320-byte native宽度。
Vivado xsim逐beat比较中间DMA与最终native output，随机source/sink停顿、容量上限、TLAST、满mask均通过。
没有把软件pack roundtrip升级为RTL结果，也没有把RTL仿真时间当板端latency。

Multi-head descriptor、increasing-stride顺序、DDR offsets/byte lengths已冻结并实际执行reference scheduler。
P2 tower只需固定行缓冲；upsample低帧是显式外部DDR scratch，不默认全P2 feature驻留BRAM。
bounded queue数值回放通过：深度4、观察peak={validation['bounded_upsample']['observed_peaks']}、背压步骤={validation['bounded_upsample']['backpressure_steps']}。
该回放是逻辑调度验证，不是HLS/RTL cycles测量；HLS库预装载队列不作为硬件容量假设。
完整detector RTL、DDR控制器时序及全部HLS pipeline的RTL cosim不是本轮已完成证据，后续measurement必须按实际实现分层记录。

## Platform / H3

PLATFORM_CONTRACT_V1_DRAFT明确historical_reference/live_verified/UNKNOWN；shell reservation为UNKNOWN。
H3_MEASUREMENT_SCHEMA.json仅为schema，没有插入旧raw arithmetic成本或启动大规模synthesis。
key包含operator/shape/activation、两个contract hashes、PE/SIMD/tile/buffer profile、part/tool/clock/source hash。
value按CSIM、CSYNTH、PLACE_ROUTE、BOARD分离；CSIM禁止latency/resource字段，CSYNTH用estimate字段，BOARD用measured字段。
当前阻止建立**原语contract-bound measurement database**的问题：无。shell reservation/整网route/board accuracy仍是后续整网部署准入要求，不妨碍先建立分层测量数据库。
H3可以开始；本轮未启动H3采样campaign，更未启动hardware-aware NAS。

## Attempts / commands / provenance

attempt01遇Xilinx ap_int三元表达式宽度推导编译错误；修为显式64bit临时变量后attempt02数值测试通过。
早期RTL缺一致timescale，xelab拒绝；补timescale后attempt03 RTL通过。attempt03同时验证Add loader从binary读取参数。
原始bundle和日志分目录保留，未将失败改名为成功。最终source hash绑定当前已验证的生成源码、公共numeric header、loader testbench和TCL。
closure_manifest.json包含全部合同与artifact hashes；旧阶段不变。工具wall time不进入measurement latency。

本地Python：`.venv_cuda/Scripts/python.exe`，cwd为repository root。主要命令：
```
tools/fpga_h2/build.py
tools/fpga_h2/boundary.py
tools/fpga_h2/packet_tests.py
tools/fpga_h2/remote.py launch attempt02
tools/fpga_h2/prepare_add_loader.py
tools/fpga_h2/remote.py launch attempt03 upsample_add
tools/fpga_h2/remote.py fetch attempt02
tools/fpga_h2/remote.py fetch attempt03
tools/fpga_h2/measurement_schema.py
tools/fpga_h2/validate.py
tools/fpga_h2/report.py
```
源码包含修复后的版本；重现使用新的独立run目录，不覆盖已保存graph或远程attempt。
远端C-sim由Vitis HLS2022.2执行，RTL由Vivado2022.2 xvlog/xelab/xsim执行；credentials只通过本地Credential Manager helper提供。
''',encoding='utf-8')

if __name__=='__main__':main()
