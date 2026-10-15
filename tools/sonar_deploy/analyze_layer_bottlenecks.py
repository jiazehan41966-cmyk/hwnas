"""Read-only offline layer timing diagnosis; no hardware operations."""
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/sonar_real_int8_deployment_v1/layer_bottlenecks_v1'
OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
inputs={}
def read(p):
 inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return json.loads(p.read_text(encoding='utf-8'))
def write(name,rows):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=[];sums=[];groups=[]
for net in ['D','C','B1']:
 base=ROOT/'results/sonar_real_int8_deployment_v1'
 bound=read(base/'formal_performance_20260928T231114'/net/'benchmark_manifest.json')
 candidates=[p for p in base.rglob('graph.json') if net in p.parts and 'seed_42' in p.parts and hashlib.sha256(p.read_bytes()).hexdigest()==bound['graph_sha256']]
 if not candidates:raise ValueError('Missing hash-matched graph: '+net)
 g=read(candidates[0])
 layers=read(base/'single_image_restart_20260928T201519'/net/'evidence/layers.json')
 perf=read(base/'formal_performance_20260928T231114'/net/'PERFORMANCE_REPORT.json')
 nodes={v['id']:v for v in g['nodes']};total=sum(v['submit_wait_seconds'] for v in layers)
 aggregate=defaultdict(float)
 for layer in layers:
  n=nodes[layer['node']];shape=g['edges'][n['id']]['shape'];co,oh,ow=shape
  kind=n['op'];macs=0
  if kind=='conv':
   kind='DW'+str(n['kernel']) if n['groups']==n['Cin'] and n['Cin']>1 else ('PW1' if n['kernel']==1 else 'Conv'+str(n['kernel']))
   macs=co*oh*ow*(n['Cin']//n['groups'])*n['kernel']**2
  aggregate[kind]+=layer['submit_wait_seconds']
  rows.append(dict(network=net,node=n['id'],operator=kind,output_shape=str(shape),seconds=layer['submit_wait_seconds'],
   percent=100*layer['submit_wait_seconds']/total,nominal_MACs=macs,
   cycles_per_nominal_MAC=layer['submit_wait_seconds']*187498125/macs if macs else 'N/A',
   source_level_input_reads_upper_estimate=macs if macs else 'NOT_ESTIMATED',
   DDR_bus_bytes='NOT_MEASURED',mismatch=layer['mismatch_count']))
 for k,v in aggregate.items():groups.append(dict(network=net,operator=k,seconds=v,percent=100*v/total))
 e=perf['e2e']['timing'];sums.append(dict(network=net,trace_seconds=total,
   native_median_seconds=perf['native']['timing']['graph_median_ms']/1000 if 'timing' in perf['native'] else 0))
write('LAYER_TIMING.csv',rows);write('OPERATOR_GROUP_TIMING.csv',groups)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(17,6))
for ax,net in zip(axes,['D','C','B1']):
 top=sorted([v for v in rows if v['network']==net],key=lambda v:v['seconds'],reverse=True)[:8][::-1]
 ax.barh([v['node'].replace('backbone.','').replace('stages.','s') for v in top],[v['seconds'] for v in top])
 ax.set_title(net);ax.set_xlabel('Submit/wait time (s), one smoke image')
fig.tight_layout();fig.savefig(OUT/'figures/top_layers.png',dpi=160);plt.close(fig)
lines=['# 逐层计算与访存瓶颈定位','',
'范围：只读已有单图逐层核对与正式性能数据，不操作板子、不改变 valid99。',
'逐层时间来自单图 smoke 的 submit_wait_seconds（启动到完成轮询），不是纯运算周期计数。含硬件访存停顿与轮询尾延迟；不包含随后 invalidate 和逐字节核对。',
'无 AXI 性能计数器；不能把循环访问次数或有效带宽当实测 DDR 总线带宽。未获取 achieved II，源码 II=1 仅为目标。','',
'## 实测主要耗时','']
for net in ['D','C','B1']:
 rr=sorted([v for v in rows if v['network']==net],key=lambda v:v['seconds'],reverse=True)
 lines+=['### '+net,'',f"逐层累计 {sum(v['seconds'] for v in rr):.3f} s；前5层占 {sum(v['percent'] for v in rr[:5]):.2f}%。",'','|层|类型|秒|占比|','|---|---|---:|---:|']
 lines += [f"|{v['node']}|{v['operator']}|{v['seconds']:.3f}|{v['percent']:.2f}%|" for v in rr[:5]]
 lines+=['','类型汇总：'+ '；'.join(f"{v['operator']} {v['percent']:.1f}%" for v in groups if v['network']==net),'']
lines+=['## 源码证据及解释','',
'- 冻结 h2_graph.cpp：卷积输出位置/输出通道循环串行；内循环逐次访问 arena，未设置特征图 tile/line buffer。仅 weights/biases 已先载入 local_w/local_b，不能称所有权重都逐 MAC 读 DDR。',
'- arena、weights、biases、descriptor 共用 parameters AXI bundle；无显式 DATAFLOW、计算 PE 展开或独立 load/compute/store 通道。实际总线事务和停顿比例尚未测量。',
'- requant 和 Add 显式 PIPELINE off；即使没有大量 MAC，也可能形成高耗时。不能只按 MACs 排优化优先级。',
'- execute_node 每2ms轮询。48–57层的单次末尾轮询量级约0.10–0.11秒（非严格上界，操作系统调度可延长）；不能据此解释百秒级总耗时。',
'- 正式无逐层核对仍约132/137/367秒，故慢速不是仅由逐层输出比较造成。',
'','## 下一步最小测量与条件化优化顺序','',
'1. valid99 完成后，固定同一输入，对 Top 耗时层增加硬件周期及 AXI 读写字节/停顿计数；同时采集 achieved II、loop latency。保留旧镜像，不直接替换正式结果。',
'2. 若高分辨率 PW/Conv 访存重读主导：先 tile 缓存及宽突发搬运，再增加 PE/SIMD；DW 候选采用行缓存。缓存容量由实际层 shape 与 BRAM 预算确定。',
'3. 若 requant/Add 串行算术主导：先对 canonical 舍入/饱和运算分级流水，不删除消费者重新量化节点；更改后重跑整数零失配与 route 时序。',
'4. 每次只优化一个确认瓶颈，比较逐层周期、整网耗时、资源与功耗。无证据前不承诺 FPS 或加速倍数。','',
'结论：串行执行与无显式特征缓存已确认；计算/DDR 停顿各占多少尚不能分离。当前不证明 DDR 带宽已饱和。']
(OUT/'analysis-report.md').write_text('\n'.join(lines),encoding='utf-8')
(OUT/'stats-appendix.md').write_text('逐层 n=1 smoke，不能估计跨会话置信区间。MACs 含 padding 名义运算；输入读取估计不是 AXI 事务。无新板端实验。',encoding='utf-8')
(OUT/'figure-catalog.md').write_text('top_layers.png：三个网络各自最慢8层，数据见 LAYER_TIMING.csv；单图启动等待耗时。',encoding='utf-8')
(OUT/'MANIFEST.json').write_text(json.dumps(dict(inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),test_evaluated=False,board_jobs_started=False),indent=2),encoding='utf-8')
print('\n'.join(lines[:55]))
