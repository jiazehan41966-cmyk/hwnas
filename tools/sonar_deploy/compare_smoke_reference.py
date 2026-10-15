"""One-shot post-run comparison with user slide; never initiate board work."""
import argparse
import csv
import ctypes
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.sonar_deploy import admission as a


def compare(run, output):
    output.mkdir(parents=True,exist_ok=True)
    with (ROOT/'reports/sonar_real_int8_deployment_v1/FINAL_DEPLOYMENT_COMPARISON.csv').open(encoding='utf-8') as f:
        software={r['network']:r for r in csv.DictReader(f)}
    rows=[]
    resources={'LUT':('lut_used','201178 / 230400 (87.32%)','lut_utilization_pct'),
        'FF':('ff_used','238591 / 460800 (51.78%)','ff_utilization_pct'),
        'DSP':('dsp_used','1532 / 1728 (88.66%)','dsp_utilization_pct'),
        'BRAM':('bram_used','211.5 / 312 (67.79%)','bram_utilization_pct')}
    def add(network,metric,reference,current,scope):
        rows.append(dict(network=network,metric=metric,reference_slide=reference,current=current,comparison_scope=scope))
    for network in ['D','C','B1']:
        sw=software[network];folder=run/network/'evidence';path=folder/'completion.json'
        data=None
        if (folder/'VERIFIED.json').exists():
            receipt=a.read_json(folder/'VERIFIED.json')
            for name,digest in receipt['files'].items():
                if a.sha256(folder/name)!=digest:raise ValueError('Verified result changed: '+name)
            data=a.read_json(path)
        add(network,'end_to_end_FPS','23.2','PENDING','Not measured in one-image layer-check smoke')
        add(network,'graph_submit_wait_s','NOT_SHOWN',data['graph_submit_wait_seconds'] if data else 'PENDING','single train image; excludes host preprocessing/decode')
        elapsed=data['graph_with_layer_checks_seconds'] if data else None
        add(network,'single_image_trace_s','NOT_SHOWN',elapsed if data else 'PENDING','Includes all layer checking; not production latency')
        add(network,'clock_MHz','NOT_SHOWN',data['board_clock_mhz'] if data else 'PENDING','actual PYNQ clock read')
        power=data['power'] if data else {}
        p=power.get('measurements',{}) if power.get('status')=='MEASURED_SINGLE_SMOKE' else {}
        add(network,'active_power_W','8.8 (chip)',p.get('active_power_w','NOT_MEASURED'),'whole-board VCC12 vs slide chip scope; NOT_DIRECTLY_COMPARABLE')
        add(network,'idle_power_W','NOT_SHOWN',p.get('idle_power_w','NOT_MEASURED'),'whole-board measured idle window')
        add(network,'energy_mJ_image','NOT_SHOWN',p.get('total_energy_mj_per_image','NOT_MEASURED'),'one complete trace window; includes comparison overhead')
        add(network,'diagnostic_FPS_W','2.64',1000/p['total_energy_mj_per_image'] if p else 'PENDING','derived images/J for trace; not end-to-end efficiency benchmark')
        add(network,'diagnostic_GOPS','470',float(sw['flops'])/elapsed/1e9 if data else 'PENDING','2*model_MACs / trace wall time; effective algorithmic proxy, not peak HW throughput')
        add(network,'diagnostic_GOPS_W','53.4',float(sw['flops'])/(p['total_energy_mj_per_image']/1000)/1e9 if p else 'PENDING','same trace energy window; different power scope from slide')
        for label,(key,ref,pct) in resources.items():
            add(network,label,ref,f"{float(sw[key]):g} ({float(sw[pct]):.3f}%)",'current routed generic serial executor; different architecture from reference slide')
        add(network,'URAM','80 / 96 (83.33%)','PENDING','not extracted into current comparison ledger; do not assume zero')
        for key in ['wns_ns','tns_ns','timing_met']:
            add(network,key,'NOT_SHOWN',sw[key],'place-and-route evidence, not sensor data')
        for metric,ref in [('Precision','0.7213'),('Recall','0.6629'),('F1','0.6881'),('AP50','0.6793'),('mAP50_95','0.2652')]:
            add(network,'board_'+metric,ref,'PENDING','requires complete valid evaluation; slide dataset/protocol not verified')
    with (output/'REFERENCE_COMPARISON.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# 用户截图与当前单图板测对照','',
        '截图数字仅作为用户提供的参考，未核验其网络、数据集和测量协议。当前是串行 DDR 功能执行器，不是吞吐优化部署。',
        '单图逐层核对不等于完整部署验收；本表不启动任何实验。test_evaluated=false。',
        'FPS/W 使用同一完整窗口的 images/J；GOPS 按 2×模型 MACs / 实际单图核对耗时计算，仅作诊断，不称硬件峰值吞吐。',
        '整板 VCC12 功耗不能与截图的芯片功耗直接比较。正式端到端 FPS 和完整板端 AP 仍需独立测量。','',
        '| 网络 | 项目 | 截图参考 | 当前 | 口径/限制 |','|---|---|---|---|---|']
    text += [f"| {r['network']} | {r['metric']} | {r['reference_slide']} | {r['current']} | {r['comparison_scope']} |" for r in rows]
    (output/'REFERENCE_COMPARISON.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    a.atomic_json(output/'COMPARISON_MANIFEST.json',dict(run=str(run),queue_status=a.read_json(run/'STATUS.json'),
        files={n:a.sha256(output/n) for n in ['REFERENCE_COMPARISON.csv','REFERENCE_COMPARISON.md']},
        source_sha256=a.sha256(__file__),test_evaluated=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-pid',type=int)
    args=p.parse_args()
    if args.wait_pid:
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        handle=kernel.OpenProcess(0x00100000,False,args.wait_pid)
        if handle:
            try:
                if kernel.WaitForSingleObject(handle,3600000)!=0:raise RuntimeError('One-shot completion wait timed out')
            finally:kernel.CloseHandle(handle)
        elif ctypes.get_last_error()!=87:
            raise OSError(ctypes.get_last_error(),'Unable to wait on queue process')
    compare(args.run,args.output)
    print('COMPARISON_WRITTEN',args.output)


if __name__=='__main__':main()
