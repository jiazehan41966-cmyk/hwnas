"""Read-only evidence audit of the frozen sonar INT8 detector results."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'reports/sonar_quantization_effectiveness'
PTQ = ROOT/'reports/phase5a_quantization/PTQ_PAIRED_RESULTS.csv'
INTEGER = ROOT/'reports/sonar_real_int8_deployment_v1/INTEGER_ACCURACY_RESULTS.csv'
QAT = ROOT/'reports/phase5b_qat/QAT_RUN_RESULTS.csv'
PROTOCOL = ROOT/'reports/phase5a_quantization/PHASE5A_PROTOCOL.md'
QUANT_CONFIG = ROOT/'reports/phase5a_quantization/quant_config.yaml'
CALIBRATION = ROOT/'reports/phase5a_quantization/CALIBRATION_MANIFEST.json'
SMOKE = ROOT/'results/sonar_real_int8_deployment_v1/single_image_restart_20260928T201519'
NETWORKS = ('D','C','B1')
SEEDS = (42,43,44)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_index(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    index = {(r['network'],int(r['seed'])):r for r in rows}
    if len(rows)!=9 or len(index)!=9 or set(index)!={(n,s) for n in NETWORKS for s in SEEDS}:
        raise ValueError('Expected 3 networks x 3 unique seeds in '+str(path))
    return index


def sample_std(values):
    return statistics.stdev(values) if len(values)>1 else 0.


def main():
    ptq=csv_index(PTQ);integer=csv_index(INTEGER);qat=csv_index(QAT)
    smoke_status=json.loads((SMOKE/'STATUS.json').read_text())
    if smoke_status.get('status')!='THREE_SINGLE_IMAGE_SMOKES_COMPLETE' or smoke_status.get('test_evaluated') is not False:
        raise ValueError('Three sealed board smokes not complete')
    rows=[];smokes={}
    for network in NETWORKS:
        evidence=SMOKE/network/'evidence'
        receipt=json.loads((evidence/'VERIFIED.json').read_text())
        result=json.loads((evidence/'completion.json').read_text())
        if receipt['status']!='SINGLE_TRAIN_SMOKE_PASS' or result['status']!=receipt['status']:
            raise ValueError('Smoke not verified: '+network)
        if any(sha(evidence/name)!=digest for name,digest in receipt['files'].items()):
            raise ValueError('Smoke artifact changed: '+network)
        if result['test_evaluated'] is not False or result['valid_evaluated'] is not False:
            raise ValueError('Smoke split seal changed: '+network)
        layers=json.loads((evidence/'layers.json').read_text())
        if len(layers)!=result['nodes'] or any(r['mismatch_count'] for r in layers):
            raise ValueError('Incomplete or nonzero layer parity: '+network)
        smokes[network]=result
        for seed in SEEDS:
            key=(network,seed);p=ptq[key];i=integer[key];q=qat[key]
            if p['checkpoint_sha256']!=i['checkpoint_sha256'] or p['checkpoint_sha256']!=q['source_best_pt_sha256']:
                raise ValueError('Unmatched checkpoint: '+str(key))
            if any(v.lower()!='false' for v in (p['test_evaluated'],i['test_evaluated'],q['test_evaluated'])):
                raise ValueError('Test seal violated: '+str(key))
            fp=float(p['fp32_mAP50_95']);pt=float(p['int8_mAP50_95']);iv=float(i['integer_mAP50_95'])
            if abs(pt-float(i['historical_mAP50_95']))>1e-12:
                raise ValueError('Historical PTQ result changed: '+str(key))
            rows.append(dict(network=network,seed=seed,checkpoint_sha256=p['checkpoint_sha256'],
            fp32_mAP50_95=fp,ptq_mAP50_95=pt,pure_integer_mAP50_95=iv,
            fp32_AP50=float(p['fp32_mAP50']),ptq_AP50=float(p['int8_mAP50']),
            qat_mAP50_95=float(q['qat_mAP50_95']),
                fp32_Recall=float(p['fp32_Recall']),ptq_Recall=float(p['int8_Recall']),
                pure_integer_Recall=float(i['integer_Recall']),
                quantization_delta=pt-fp,integer_minus_ptq=iv-pt,
                fp32_to_ptq_retention=pt/fp if fp else None,
                test_evaluated=False))
    by={(r['network'],r['seed']):r for r in rows}
    summary=[]
    for network in NETWORKS:
        part=[by[(network,s)] for s in SEEDS]
        fp=statistics.mean(r['fp32_mAP50_95'] for r in part)
        pt=statistics.mean(r['ptq_mAP50_95'] for r in part)
        iv=statistics.mean(r['pure_integer_mAP50_95'] for r in part)
        qatmean=statistics.mean(r['qat_mAP50_95'] for r in part)
        recalls=statistics.mean(r['pure_integer_Recall'] for r in part)
        control=[by[('D',s)] for s in SEEDS]
        map_differences=[by[(network,s)]['pure_integer_mAP50_95']-by[('D',s)]['pure_integer_mAP50_95'] for s in SEEDS]
        recall_differences=[by[(network,s)]['pure_integer_Recall']-by[('D',s)]['pure_integer_Recall'] for s in SEEDS]
        if network=='D':map_differences=[0.]*3;recall_differences=[0.]*3
        observed=smokes[network]
        summary.append(dict(network=network,fp32_mean=fp,ptq_mean=pt,
            pure_integer_mean=iv,qat_mean=qatmean,
            fp32_AP50_mean=statistics.mean(r['fp32_AP50'] for r in part),
            ptq_AP50_mean=statistics.mean(r['ptq_AP50'] for r in part),
            fp32_Recall_mean=statistics.mean(r['fp32_Recall'] for r in part),
            ptq_Recall_mean=statistics.mean(r['ptq_Recall'] for r in part),
            ptq_delta_mean=statistics.mean(r['quantization_delta'] for r in part),
            ptq_delta_sample_std=sample_std([r['quantization_delta'] for r in part]),
            ptq_worse_than_fp32_seed_count=sum(r['quantization_delta']<0 for r in part),
            ptq_retention_pct=100*pt/fp,
            integer_recall_mean=recalls,
            integer_minus_D_map_mean=statistics.mean(map_differences),
            integer_minus_D_recall_mean=statistics.mean(recall_differences),
            integer_mAP_above_D_seeds=sum(v>0 for v in map_differences),
            integer_Recall_above_D_seeds=sum(v>0 for v in recall_differences),
            smoke_layers=observed['nodes'],smoke_mismatch_count=observed['mismatch_count'],
            smoke_graph_submit_wait_s=observed['graph_submit_wait_seconds'],
            smoke_trace_total_s=observed['graph_with_layer_checks_seconds'],
            smoke_power_status=observed['power']['status'],
            smoke_actual_clock_mhz=observed['board_clock_mhz'],
            full_board_valid_status='PENDING',test_evaluated=False))
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'PAIRED_RESULTS.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (REPORT/'SUMMARY.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    manifest=dict(scope='frozen_fp32_ptq_qat_integer_and_single_image_board_evidence',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            [PTQ,INTEGER,QAT,PROTOCOL,QUANT_CONFIG,CALIBRATION,SMOKE/'STATUS.json',*[SMOKE/n/'evidence/VERIFIED.json' for n in NETWORKS],
             *[SMOKE/n/'evidence/completion.json' for n in NETWORKS]]},
        audit_script_sha256=sha(__file__),paired_rows=len(rows),
        sources_same_checkpoint=True,test_evaluated=False,
        outputs_sha256={name:sha(REPORT/name) for name in ('PAIRED_RESULTS.csv','SUMMARY.csv')})
    lines=['# INT8 量化有效性核验','',
        '本核验只使用既有冻结结果。同一网络和 seed 的 FP32、PTQ、纯整数结果绑定同一 `ALL_BEST_PT` checkpoint。板端仅使用 seed42 的固定 train 图。',
        '', '| 网络 | FP32 mAP50:95 | PTQ INT8 | 精度下降 | 保留率 | 纯整数 | QAT | FP32/PTQ Recall | 单图逐层 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for v in summary:
        lines.append(f"| {v['network']} | {v['fp32_mean']:.6f} | {v['ptq_mean']:.6f} | {v['ptq_delta_mean']:+.6f} | {v['ptq_retention_pct']:.1f}% | {v['pure_integer_mean']:.6f} | {v['qat_mean']:.6f} | {v['fp32_Recall_mean']:.3f} / {v['ptq_Recall_mean']:.3f} | {v['smoke_layers']}层，零失配 |")
    lines+=['', '## 判断', '',
        '- **预注册 PTQ 精度门槛：未通过。** Phase5A 规定任一网络的平均绝对下降 >0.003 或相对下降 >20% 即进入 `QAT_REQUIRED`。D/C/B1 的平均绝对下降均 >0.003；当前不能判定 `PTQ_SUFFICIENT`。三个网络仍保留了部分 FP32 精度，C 降幅最大。',
        '- **整数部署正确性：单图通过。** D/C/B1 的真实 seed42 参数在 ZCU104 上分别完成 48/50/57 层及全部输出 head 的零失配核对；这是一个固定 train 图的功能证据。',
        '- **QAT 恢复：未证实。** QAT 没有恢复 FP32 均值，也没有稳定超过 PTQ。',
        '- **相对 D 底线：B1/C 的三 seed 纯整数 mAP50:95 与 Recall 均值较 D 高。** B1 的逐 seed mAP 优势为 2/3、Recall 为 3/3；C 分别为 1/3、2/3。均值通过不能表述为逐 seed 稳定优势。',
        '- **部署效率收益：未证实。** 当前是串行图执行、单图带逐层核对的诊断数据；没有匹配的 FP32 板端对照或正式重复推理基准，不能声称量化提高 FPS 或降低能耗。',
        '- **正式板端检测精度：待测。** 单图无 GT 统计，valid99 尚未由板端完整执行；test_evaluated=false。',
        '- **量化方法的独特收益：未验证。** 当前合同是固定的对称、per-tensor W8A8 PTQ，没有同预算的其他 INT8 方案对照；不能声称这一量化设计优于通用 PTQ。',
        '', '## 板端单图诊断', '',
        '| 网络 | 图执行提交/等待 | 含逐层核对总耗时 | 实际时钟 | 功耗/能量状态 |',
        '|---|---:|---:|---:|---|']
    for v in summary:
        lines.append(f"| {v['network']} | {v['smoke_graph_submit_wait_s']:.3f}s | {v['smoke_trace_total_s']:.3f}s | {v['smoke_actual_clock_mhz']:.6f}MHz | {v['smoke_power_status']}，仅调试过程 |")
    lines+=['', '这段执行时长包含串行逐节点调用；能量积分也覆盖逐层检查造成的空隙。原始传感器数据已保存，但单图动态能量未做重复性验证，不能作为最终每图能耗。三 seed 和 valid99 都来自同一官方开发划分，不代表跨数据集泛化。',
        '', '## 完成“量化设计有效”判断仍需', '',
        '1. 关闭逐层核对，在同一镜像、batch1、640 输入上做固定预热和重复推理，报告板端延迟、FPS、功耗和能量积分。',
        '2. 若要声称量化带来硬件效率增益，需要同协议 FP32/高精度硬件对照，或将论断严格限定为 INT8 可部署与成本。',
        '3. 如果目标是板端检测精度与软件整数结果一致，需要在冻结 valid99 上完成板端评估。单图通过不能代替此项。',
        '', '数据：`PAIRED_RESULTS.csv`、`SUMMARY.csv`；来源与哈希：`AUDIT_MANIFEST.json`。']
    (REPORT/'QUANTIZATION_EFFECTIVENESS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest['outputs_sha256']['QUANTIZATION_EFFECTIVENESS.md']=sha(REPORT/'QUANTIZATION_EFFECTIVENESS.md')
    (REPORT/'AUDIT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(report=str(REPORT/'QUANTIZATION_EFFECTIVENESS.md'),summary=summary,test_evaluated=False)))


if __name__=='__main__':main()
