"""Local-only immutable snapshot of finished formal board measurements."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.sonar_deploy import admission as a
from tools.sonar_deploy.power import verify_iio_evidence
from hwnas_fpga.metrics_ledger import read_metric_records


def csv_write(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def describe(values):
    values=sorted(values);position=.95*(len(values)-1);lo=int(position)
    return dict(mean=statistics.mean(values),sample_std=statistics.stdev(values),
        median=statistics.median(values),p95=values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(position-lo),
        min=min(values),max=max(values))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run=args.run.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False);fig=out/'figures';fig.mkdir()
    sources={};summaries=[];points=[];resources=[];inventory=[]
    def bind(path):sources[str(path.resolve())]=a.sha256(path)
    route_path=ROOT/'reports/sonar_real_int8_deployment_v1/FINAL_DEPLOYMENT_COMPARISON.csv'
    bind(route_path)
    with route_path.open(encoding='utf-8') as f:route={r['network']:r for r in csv.DictReader(f)}
    # Take a fixed completed-network snapshot; never poll or read running B1 data.
    complete=[n for n in ['D','C','B1'] if (run/n/'PERFORMANCE_REPORT.json').is_file()]
    for n in ['D','C','B1']:
        inventory.append(dict(network=n,performance_status='AVAILABLE' if n in complete else 'PENDING_AT_SNAPSHOT',
                              board_accuracy='PENDING_VALID99',test_evaluated=False))
    for n in complete:
        folder=run/n;report_path=folder/'PERFORMANCE_REPORT.json';bind(report_path)
        report=a.read_json(report_path);manifest_path=folder/'benchmark_manifest.json';bind(manifest_path)
        manifest=a.read_json(manifest_path)
        if report['status']!='PERFORMANCE_PASS' or report['test_evaluated'] is not False:raise ValueError('Not admitted measurement')
        binding={k:manifest[k] for k in ('deployment_binding_sha256','package_manifest_sha256','board_manifest_sha256','graph_sha256')}
        binding['benchmark_manifest_sha256']=a.sha256(manifest_path)
        ledger=read_metric_records(folder/'METRICS.jsonl');bind(folder/'METRICS.jsonl')
        if report['metric_event_id'] not in {r['event_id'] for r in ledger}:raise ValueError('Missing durable metrics')
        for group in ['native','e2e']:
            prefix='native_results' if group=='native' else 'results'
            raw_path=folder/'native_results/completion.json' if group=='native' else folder/'e2e.json'
            bind(raw_path);data=a.read_json(raw_path)
            if data!=report[group]:raise ValueError('Aggregated timing differs from raw result')
            if data['warmup']!=1 or data['repetitions']!=10 or len(data['rows'])!=11:raise ValueError('Wrong repetition protocol')
            rows=data['rows']
            if [r['call_index'] for r in rows]!=list(range(11)):raise ValueError('Call indices changed')
            powerdir=folder/prefix/'power'
            rpc_path=folder/prefix/('completion.json' if group=='native' else 'e2e_rpc_session.json')
            bind(rpc_path);rpc=a.read_json(rpc_path)
            power=verify_iio_evidence(powerdir,binding,manifest['power_monitor']['source_sha256'],rpc['rows'])
            if power!=report['power'][prefix]:raise ValueError('Power report differs from verified raw data')
            for name in ['power.json','power_samples.csv']:bind(powerdir/name)
            selected=rows[1:];stats=describe([x['graph_seconds'] for x in selected]);ps=power.get('summary',{})
            calls=ps.get('calls',[])[1:]
            for i,x in enumerate(selected):
                q=calls[i] if calls else {}
                points.append(dict(network=n,group=group,call=x['call_index'],graph_seconds=x['graph_seconds'],
                    e2e_seconds=x.get('end_to_end_seconds','N/A'),board_power_w=q.get('mean_power_w','NOT_MEASURED'),
                    board_energy_mj=q.get('total_energy_mj','NOT_MEASURED'),dynamic_energy_mj=q.get('dynamic_energy_mj','NOT_MEASURED')))
            energy=ps.get('energy_per_inference_mj')
            row=dict(network=n,group=group,seed=42,n_measured=10,n_sessions=1,
                **{'graph_'+k+'_seconds':v for k,v in stats.items()},
                completed_window_fps=data['throughput_fps'] if group=='native' else data['completed_window_fps'],
                board_clock_mhz=data['clock_mhz'] if group=='native' else data['actual_clock_mhz'],
                idle_power_w=ps.get('idle_power_w','NOT_MEASURED'),board_power_w=ps.get('power_w','NOT_MEASURED'),
                board_energy_mj_image=energy if energy is not None else 'NOT_MEASURED',
                board_dynamic_energy_mj_image=ps.get('dynamic_energy_per_inference_mj','NOT_MEASURED'),
                board_images_per_j=1000/energy if energy is not None else 'NOT_MEASURED',
                derived_GOPS_infer_window=float(route[n]['flops'])*10/ps['active_duration_s']/1e9 if energy is not None else 'NOT_MEASURED',
                derived_GOPS_per_W=float(route[n]['flops'])/1e9/(energy/1000) if energy is not None else 'NOT_MEASURED',
                e2e_median_seconds=statistics.median([x['end_to_end_seconds'] for x in selected]) if group=='e2e' else 'N/A',
                e2e_p95_seconds=describe([x['end_to_end_seconds'] for x in selected])['p95'] if group=='e2e' else 'N/A',
                confidence_interval='NOT_ESTIMATED_SINGLE_SESSION',test_evaluated=False)
            summaries.append(row)
        keys=['params','macs','flops','lut_used','lut_utilization_pct','ff_used','ff_utilization_pct','dsp_used','dsp_utilization_pct',
              'bram_used','bram_utilization_pct','wns_ns','tns_ns','timing_met','route_clock_mhz','route_evidence_sha256']
        resources.append(dict(network=n,**{k:route[n][k] for k in keys},uram='NOT_EXTRACTED',evidence_layer='place_and_route'))
    if not summaries:raise ValueError('No complete formal measurements available')
    csv_write(out/'PERFORMANCE_SUMMARY.csv',summaries);csv_write(out/'REPETITION_METRICS.csv',points)
    csv_write(out/'RESOURCE_TIMING.csv',resources);csv_write(out/'METRIC_INVENTORY.csv',inventory)
    reference=[]
    for r in summaries:
        if r['group']!='e2e':continue
        for metric,ref,value,boundary in [
            ('FPS',23.2,r['completed_window_fps'],'Current host+board E2E; reference pipeline unknown'),
            ('Power_W',8.8,r['board_power_w'],'Reference chip power vs current whole-board VCC12; not directly comparable'),
            ('FPS_W_or_images_J',2.64,r['board_images_per_j'],'Current board-only infer window; reference end-to-end/chip boundary differs'),
            ('GOPS',470,r['derived_GOPS_infer_window'],'Current effective 2*MAC count / measured infer time; reference operation count unknown'),
            ('GOPS_W',53.4,r['derived_GOPS_per_W'],'Current whole-board energy; not reference chip energy')]:
            reference.append(dict(network=r['network'],metric=metric,screenshot_reference=ref,current=value,comparability=boundary))
    csv_write(out/'SCREENSHOT_COMPARISON.csv',reference)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figobj,axes=plt.subplots(1,2,figsize=(10,3.8))
    for n in complete:
        rows=[r for r in points if r['network']==n and r['group']=='e2e']
        axes[0].plot([r['call'] for r in rows],[r['e2e_seconds'] for r in rows],marker='o',label=n)
        axes[1].plot([r['call'] for r in rows],[r['board_energy_mj']/1000 for r in rows],marker='o',label=n)
    axes[0].set(ylabel='Host + board latency (s)',xlabel='Measured repetition')
    axes[1].set(ylabel='Board infer-call energy (J/image)',xlabel='Measured repetition')
    for ax in axes:ax.legend();ax.grid(alpha=.25)
    figobj.suptitle('Formal measurements: one session, seed42, warmup excluded')
    figobj.tight_layout();figobj.savefig(fig/'repetition_measurements.png',dpi=160);figobj.savefig(fig/'repetition_measurements.svg');plt.close(figobj)
    csv_write(fig/'repetition_measurements.csv',[r for r in points if r['group']=='e2e'])
    text=['# 已完成正式板测：离线整理','',f'快照时间：{datetime.now().isoformat(timespec="seconds")}。仅读取本地结果，不操作服务器/板子。',
          '',f'完整结果：{", ".join(complete)}。其他网络保留 PENDING。所有记录 test_evaluated=false。',
          '', '| 网络 | E2E median (秒) | E2E P95 (秒) | 实际 FPS | 平均整板 W | 板级 mJ/image |', '|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        if r['group']=='e2e':text.append(f"| {r['network']} | {r['e2e_median_seconds']:.6f} | {r['e2e_p95_seconds']:.6f} | {r['completed_window_fps']:.8f} | {r['board_power_w']:.6f} | {r['board_energy_mj_image']:.3f} |")
    text += ['', '## 证据口径与结论','',
        '- 原生 FPGA 与端到端各预热1次＋测量10次；不含逐层核对。每次输出与冻结整数见证核对。',
        '- 端到端包含服务器 CPU 预处理、网络传输、FPGA 与 CPU decode/NMS。FPS 为实际完成数/测量窗口，不用中位延迟倒数代替。',
        '- 功耗来自整板 VCC12；能耗只积分板端 infer 调用窗口，包含 PS/PL，不含服务器能量。不得标成整个系统端到端能耗。',
        '- images/J 用实际板级能耗计算，等价于同一边界的 FPS/W；不将端到端 FPS 除以不同边界的功耗。GOPS 按1MAC=2 operations派生，不是硬件理论峰值。',
        '- 仅 seed42、同图、同板、同会话10次重复；描述性 mean/std/min/max/median/P95 已保存，不把重复当成10个独立模型/板卡实验。95% CI与显著性检验不作跨会话推广。',
        '- 资源和WNS/TNS来自对应布局布线证据，与运行传感器数据分开。URAM未提取，不填零。',
        '- 当前串行执行器速度远低于截图23.2 FPS，不能声称已实现实时检测；资源占用较低不能替代吞吐效率。',
        '- 截图芯片8.8 W与当前整板功耗边界不同；截图网络、数据集、操作计数及精度协议未核验，不进行精度高低结论。',
        '- 完整 valid99 未运行，板端 Precision/Recall/F1/AP 保留 PENDING；原有软件精度不填充到板端精度。','',
        '## 文件','', '`PERFORMANCE_SUMMARY.csv`、`REPETITION_METRICS.csv`、`RESOURCE_TIMING.csv`、`SCREENSHOT_COMPARISON.csv`、`METRIC_INVENTORY.csv`。',
        '', '![重复测量](figures/repetition_measurements.png)']
    (out/'analysis-report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    (out/'stats-appendix.md').write_text('重复单位：同一会话中的10次推理；1次预热已排除。sample std使用ddof=1；P95线性插值。功耗为时间积分/有效窗口时长，非未加权采样均值。功耗摘要按8 ULP重验，原始CSV/绑定哈希必须一致。未做显著性检验或跨会话95%CI：独立会话数=1，连续重复并不独立。\n',encoding='utf-8')
    (out/'figure-catalog.md').write_text('repetition_measurements.png/svg：左图完整E2E延迟，右图仅板端infer窗口能耗；CSV含已完成网络的全部端到端正式重复测量点（实际数量以CSV为准）。用于描述会话内波动，不解释因果或跨种子稳定性。\n',encoding='utf-8')
    # Check raw inputs again to catch concurrent modification during snapshot.
    for path,digest in sources.items():
        if a.sha256(path)!=digest:raise ValueError('Input changed during offline analysis: '+path)
    a.atomic_json(out/'MANIFEST.json',dict(completed_networks=complete,source_sha256=a.sha256(__file__),
        inputs=sources,outputs={p.relative_to(out).as_posix():a.sha256(p) for p in out.rglob('*') if p.is_file()},
        local_only=True,no_remote_jobs=True,test_evaluated=False))
    print(json.dumps(dict(output=str(out),completed_networks=complete,summary_rows=len(summaries),repetition_rows=len(points))))


if __name__=='__main__':main()
