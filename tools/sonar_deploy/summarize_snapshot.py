"""Render measured partial board evidence separately from deployment admission."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.sonar_deploy import admission as a
from hwnas_fpga.metrics_ledger import build_metric_record,append_metric_record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--report-dir',type=Path,default=ROOT/'reports/sonar_real_int8_deployment_v1')
    args=parser.parse_args();folder=args.snapshot.resolve();report=args.report_dir.resolve()
    manifest=a.read_json(folder/'manifest.json')
    for name,digest in manifest['files'].items():
        if a.sha256(folder/name)!=digest:raise ValueError('Snapshot SHA mismatch: '+name)
    data=a.read_json(folder/'snapshot.json')
    rows=list(csv.DictReader((report/'FINAL_DEPLOYMENT_COMPARISON.csv').open(encoding='utf-8')))
    D=next(r for r in rows if r['network']=='D')
    if data['graph_sha256']!='de6516bbb7e4b4f66da8d5794f12974513a5fc22fd909c1e2b81345711ad8f36':
        raise ValueError('Not the admitted D graph')
    frames=data['frames'];valid=[r for r in frames if r['split']=='valid'];train=[r for r in frames if r['split']=='train']
    if any(r['mismatch_count'] for r in frames+data['layers']):raise ValueError('Observed board mismatch')
    times=[r['graph_submit_wait_seconds'] for r in valid]
    median=statistics.median(times);ordered=sorted(times);position=.95*(len(times)-1);low=int(position)
    p95=ordered[low]+(ordered[min(low+1,len(ordered)-1)]-ordered[low])*(position-low)
    power=data['power'];samples=data['power_samples']
    if power['status']=='MEASURED_ACTIVE_WINDOW_ONLY':
        for sample in samples:
            if sample['power_w']!=sample['power_raw']*12.5/1000:raise ValueError('Power unit conversion differs')
        duration=samples[-1]['monotonic_s']-samples[0]['monotonic_s']
        mean=sum((b['monotonic_s']-v['monotonic_s'])*(b['power_w']+v['power_w'])/2 for v,b in zip(samples,samples[1:]))/duration
        if abs(mean-power['mean_power_w'])>1e-12:raise ValueError('Power integration differs')
    when=datetime.fromtimestamp(data['captured_at_unix'],timezone(timedelta(hours=8))).isoformat()
    measured={}
    provenance=dict(evidence_layer='board_measurement',target_device='ZCU104',
        source=str(folder/'snapshot.json'),artifact_sha256=a.sha256(folder/'snapshot.json'),
        measurement_method='bounded 30s INA226 IIO snapshot during ongoing D validation',
        power_scope='whole_board_12v_input_active_window_not_per_image',idle_subtracted=False,
        graph_sha256=data['graph_sha256'],checkpoint_sha256=D['checkpoint_sha256_seed42'],
        numeric_contract='NUMERIC_CONTRACT_V1',formal_deployment_acceptance=False)
    if power['status']=='MEASURED_ACTIVE_WINDOW_ONLY':measured['power_w']=dict(value=mean,provenance=provenance)
    measured['actual_clock_mhz']=dict(value=data['actual_clock_mhz'],provenance=dict(provenance,measurement_method='PYNQ clock register read'))
    record=build_metric_record(task_type='detection',dataset='Cylinder2',network='D',seed=42,
        stage='partial_board_power_snapshot',identity=dict(run_id=folder.name,snapshot_sha256=a.sha256(folder/'snapshot.json'),
        checkpoint_sha256=D['checkpoint_sha256_seed42'],test_evaluated=False,clean_reference_available=False),measured=measured)
    append_metric_record(folder/'PARTIAL_METRICS.jsonl',record)
    text=f'''# 当前真实板测数据（部分结果）

快照时间：{when}。本文件不是完整部署验收报告；不覆盖历史结果或正式板测准入。

## 当前网络状态

| 网络 | 逐层核对 | train图 | valid图 | 完整板端mAP | 状态 |
|---|---|---:|---:|---|---|
| D seed42 | 首图48节点、train逐层均零失配 | {len(train)}/8 | {len(valid)}/99 | PENDING | 实际板端运行中 |
| C seed42 | PENDING | 0/8 | 0/99 | PENDING | 未上板，完整任务派发已暂停 |
| B1 seed42 | PENDING | 0/8 | 0/99 | PENDING | 未上板，完整任务派发已暂停 |

软件INT8精度已完成，但不把软件指标填成板端精度。当前所有已完成图像记录均零失配。

## D运行时间（完整valid过程中的部分样本）

- 实际板端时钟：{data['actual_clock_mhz']:.6f} MHz；布局布线报告时钟：187.512 MHz。
- 首张固定train图48节点提交/等待时间之和：{sum(x['submit_wait_seconds'] for x in data['layers']):.6f}秒。
- 已完成{len(valid)}张valid图：整图提交/等待时间中位数 **{median:.6f}秒/图**，P95 **{p95:.6f}秒/图**。
- 对应中位延迟倒数：**{1/median:.8f} FPS**。这是串行执行速度诊断，不是并行吞吐实测。
- 此处不包含主机预处理、网络传输、decode/NMS，不是端到端耗时。
- 这些是完整valid中不同图像的部分统计，不冒充预注册的固定图预热1次＋测量10次基准。

## 实测功耗与尚缺能效

| 项目 | 当前结果 | 证据范围 |
|---|---|---|
| 运行期间平均功耗 | {power.get('mean_power_w','NOT_MEASURED')} W | 一次固定30秒窗口的时间加权平均 |
| 运行期间最小/最大功耗 | {power.get('min_power_w','NOT_MEASURED')} / {power.get('max_power_w','NOT_MEASURED')} W | 原始传感器样本范围 |
| 实际采样时长/点数 | {power.get('duration_seconds','NOT_MEASURED')}秒 / {len(samples)}点 | 约10Hz轮询，不声称样本独立 |
| 空闲功耗 | NOT_MEASURED | 板子仍在执行，不能标成idle |
| 每图能量 mJ/image | NOT_MEASURED | 需完整推理边界内积分 |
| 能效 images/J（FPS/W） | NOT_MEASURED | 需匹配同一完整推理窗口的功耗和耗时 |
| 端到端时间/功耗 | PENDING / NOT_MEASURED | 当前30秒只测整板12V输入，未测主机能量 |

来源：ZCU104 U183 INA226 / VCC12；读取IIO raw与scale，未改传感器设置。包括整板PS/PL功耗，不是仅FPGA逻辑功耗。不把此次30秒功耗乘131秒估成单图能量。

## 实际布局布线资源

| 资源 | 使用量 | 利用率 |
|---|---:|---:|
| LUT | {float(D['lut_used']):.0f} | {float(D['lut_utilization_pct']):.3f}% |
| FF | {float(D['ff_used']):.0f} | {float(D['ff_utilization_pct']):.3f}% |
| DSP | {float(D['dsp_used']):.0f} | {float(D['dsp_utilization_pct']):.3f}% |
| BRAM（36Kb tile） | {float(D['bram_used']):.0f} | {float(D['bram_utilization_pct']):.3f}% |

WNS={D['wns_ns']}ns；TNS={D['tns_ns']}ns；Timing Met={D['timing_met']}。
三候选复用同一通用执行器，所以资源表相同不代表运行时间或功耗相同；C/B1未测数据不从D外推。

## 后续必须补齐（不自动启动长测）

先完成C/B1单图功能试跑。正式性能验收恢复时，分别记录空闲5秒与完整推理窗口，保存原始功耗CSV及起止时间；计算整板总能量、扣空闲的诊断能量与images/J，并区分核心执行与端到端边界。
此前长测自动派发已按用户要求暂停；功耗代码已具备，但本快照不代表该完整配对测量已执行。

77字段记录：本次power_w与actual_clock_mhz写入独立PARTIAL_METRICS.jsonl；其余未测字段保留PENDING，检测Top-1保持不适用。`test_evaluated=false`。

## 原始文件

- `{folder/'snapshot.json'}`
- `{folder/'layer_parity.csv'}`
- `{folder/'frame_times.csv'}`
- `{folder/'power_samples.csv'}`
- `{folder/'PARTIAL_METRICS.jsonl'}`
'''
    (report/'CURRENT_BOARD_MEASUREMENTS.md').write_text(text,encoding='utf-8')
    summary=dict(captured_at=when,network='D',seed=42,train_frames=len(train),valid_frames=len(valid),
        graph_median_seconds=median,graph_p95_seconds=p95,serial_fps=1/median,power=power,
        formal_board_acceptance='PENDING',test_evaluated=False)
    a.atomic_json(report/'CURRENT_BOARD_MEASUREMENTS.json',summary)
    a.atomic_json(report/'CURRENT_BOARD_MEASUREMENTS_MANIFEST.json',dict(source_sha256=a.sha256(__file__),
        input_snapshot_sha256=a.sha256(folder/'snapshot.json'),comparison_sha256=a.sha256(report/'FINAL_DEPLOYMENT_COMPARISON.csv'),
        report_sha256=a.sha256(report/'CURRENT_BOARD_MEASUREMENTS.md'),partial_metric_event=record['event_id'],test_evaluated=False))
    print(json.dumps({k:v for k,v in summary.items() if k!='power'}))


if __name__=='__main__':main()
