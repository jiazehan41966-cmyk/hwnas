"""Read-only remote completion proof and traceable local pilot reporting.

No training, summarize, checkpoint writes, test access or hardware execution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASE = ROOT/'results/detector_elastic_pilot_20261005'
RAW = BASE/'remote_snapshot/raw_bytes'
REPORT = ROOT/'reports/detector_elastic_pilot_20261005'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remote_checkpoint_proof():
    from tools.detector_elastic_pilot import remote
    code = '''from pathlib import Path
import json,hashlib,torch
from tools.detector_elastic_pilot import train
torch.set_num_threads(2)
root=Path(REMOTE);result=root/RESULT
manifest,baseline=train.load_manifest(result,True)
manifest_sha=train.sha(result/'execution_manifest.json')
queue=json.loads((result/'queue_status.json').read_bytes())
assert queue['status']=='COMPLETE'
assert queue['binding']['execution_manifest_sha256']==manifest_sha
out={'status':'PASS','execution_manifest_sha256':manifest_sha,'test_evaluated':False,'prototypes':{}}
for prototype in train.PROTOTYPES:
 folder=train.result_dir(result,prototype,42)
 record=json.loads((folder/'result.json').read_bytes())
 latest=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
 best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
 expected=train.binding(result,prototype,42,True)
 assert latest['binding']==best['binding']==expected
 assert latest['epoch']==80 and latest['global_step']==6640
 history=latest['history']
 assert [r['epoch'] for r in history]==list(range(1,81))
 assert all(r['batches']==83 and r['valid']['num_images']==99 for r in history)
 assert all(r.get('gradient_finite') is True for r in history)
 assert json.loads((folder/'epoch_log.json').read_bytes())==history
 selection=max(history,key=lambda r:r['valid']['mAP50_95'])
 assert best['epoch']==selection['epoch']==record['best_epoch']
 assert best['global_step']==best['epoch']*83
 assert best['history']==history[:best['epoch']]
 for key in train.METRICS:
  assert record[key]==selection['valid'][key]
 assert record['endpoint_mAP50_95']==history[-1]['valid']['mAP50_95']
 assert train.sha(folder/'best.pt')==record['best_checkpoint_sha256']
 assert train.sha(folder/'latest.pt')==record['latest_checkpoint_sha256']
 assert train.sha(folder/'epoch_log.json')==record['history_sha256']
 assert record['execution_manifest_sha256']==manifest_sha
 assert record['formal_eligible'] is True and record['test_evaluated'] is False
 identity=json.loads((folder/'initialization_identity.json').read_bytes())
 smoke=train.result_dir(result,prototype,42,False)
 assert identity==json.loads((smoke/'initialization_identity.json').read_bytes())
 assert latest['initialization_identity']==best['initialization_identity']==identity
 assert latest['initial_state_sha256']==best['initial_state_sha256']
 assert latest['architecture_contract']==best['architecture_contract']
 status=json.loads((folder/'status.json').read_bytes())
 assert status['status']=='COMPLETE' and status['binding']==expected
 info=json.loads((folder/'model_info.json').read_bytes())
 assert record['Params']==info['total_params'] and record['MACs']==info['estimated_macs']
 assert info['architecture_contract']==json.loads(json.dumps(latest['architecture_contract']))
 assert json.loads((folder/'best_valid_metrics.json').read_bytes())==selection['valid']
 records=json.loads((folder/'best_valid_records.json').read_bytes())
 assert len(records)==99
 assert train.supplemental(records,baseline)==record['supplemental']
 initial=train.build_model(prototype,42)
 assert initial.initialization_identity()==identity
 assert train.frozen.state_hash(initial)==latest['initial_state_sha256']
 changes={}
 for group in ['backbone','neck','head']:
  prefix=group+'.'
  initial_state=initial.state_dict()
  names=[name for name in initial_state if name.startswith(prefix)]
  assert set(latest['model_state_dict'])==set(initial_state)
  assert all(torch.isfinite(latest['model_state_dict'][name]).all() for name in names)
  count=sum(not torch.equal(initial_state[name],latest['model_state_dict'][name]) for name in names)
  assert count>0
  changes[group]={'changed_state_tensors':count,'total_state_tensors':len(names)}
 out['prototypes'][prototype]={'status':'PASS','epochs':80,'global_step':6640,
  'earliest_best_epoch':best['epoch'],'best_metrics':{k:record[k] for k in train.METRICS},
  'best_checkpoint_sha256':record['best_checkpoint_sha256'],
  'latest_checkpoint_sha256':record['latest_checkpoint_sha256'],
  'best_valid_records_sha256':train.sha(folder/'best_valid_records.json'),
  'trained_state_changes':changes,'result_sha256':train.sha(folder/'result.json')}
print(json.dumps(out))
'''
    # execute starts the existing pinned interpreter, then imports the staged
    # module from this owned workspace; no credentials are included in output.
    code = "import os,sys\nos.chdir("+repr(remote.REMOTE)+")\nsys.path[:0]=['.', 'src', 'tools/sonar_spec', 'tools/sonar_analysis', 'scripts']\n" + remote.expand(code)
    proof=json.loads(remote.execute(code,120))
    target=BASE/'completion_checkpoint_proof.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=proof:
        raise ValueError('Existing completed checkpoint proof differs')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(proof,indent=2,ensure_ascii=False),encoding='utf-8')
    return proof


def report():
    proof=json.loads((BASE/'completion_checkpoint_proof.json').read_text(encoding='utf-8'))
    if proof['status']!='PASS':
        raise ValueError('Completion proof required')
    manifest=json.loads((RAW/'execution_manifest.json').read_text(encoding='utf-8'))
    if digest(RAW/'execution_manifest.json')!=proof['execution_manifest_sha256']:
        raise ValueError('Wrong completion manifest')
    rows=[]
    for prototype in ('E0','E1'):
        path=RAW/'formal'/prototype/'seed_42/result.json'
        record=json.loads(path.read_text(encoding='utf-8'))
        if digest(path)!=proof['prototypes'][prototype]['result_sha256']:
            raise ValueError('Raw result differs from checkpoint proof')
        classes=record['supplemental']['per_class_precision_recall']
        tp=sum(r['true_positives'] for r in classes.values())
        fp=sum(r['false_positives'] for r in classes.values())
        gt=sum(r['ground_truths'] for r in classes.values())
        assert gt==99 and (tp/(tp+fp) if tp+fp else 0)==record['Precision'] and tp/gt==record['Recall']
        p,r=record['Precision'],record['Recall']
        rows.append(dict(prototype=prototype,seed=42,best_epoch=record['best_epoch'],
            AP50=record['AP50'],AP75=record['AP75'],mAP50_95=record['mAP50_95'],
            Precision=p,Recall=r,F1=2*p*r/(p+r) if p+r else 0,
            TP=tp,FP=fp,FN=gt-tp,FP_per_image=fp/99,Params=record['Params'],
            MACs=record['MACs'],endpoint_mAP50_95=record['endpoint_mAP50_95'],
            result_sha256=digest(path)))
    REPORT.mkdir(parents=True,exist_ok=True)
    with (BASE/'completed_accuracy.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    comparisons={}
    historical_b1=.039263507668511013
    historical_recall=.46464646464646464
    for row in rows:
        comparisons[row['prototype']]=dict(delta_mAP50_95_vs_historical_B1=row['mAP50_95']-historical_b1,
            delta_Recall_vs_historical_B1=row['Recall']-historical_recall)
    comparisons['E1_minus_E0']={key:rows[1][key]-rows[0][key]
        for key in ('mAP50_95','AP50','AP75','Precision','Recall','FP_per_image','MACs')}
    (BASE/'completed_comparison.json').write_text(json.dumps(comparisons,indent=2),encoding='utf-8')
    lines=['# ELASTIC 式检测后端：完成结果', '',
        '两组均完成 seed42、80轮、6640个训练步；每轮完整 valid99。结果与实际 best/latest 检查点及源/配置/数据绑定通过核验。', '',
        '| 方案 | 最佳轮 | AP50 | AP75 | mAP50:95 | Precision | Recall | FP/图 | 参数量 | G MAC | 第80轮 mAP |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        '| 原B1历史参考 | 53 | 0.102040 | 0.022009 | 0.039264 | 0.042396 | 0.464646 | — | — | 6.339 | — |',
        '| L1历史参考 | 56 | 0.091929 | 0.013589 | 0.032070 | 0.026380 | 0.434343 | 16.030 | 436,220 | 0.797 | 0.012554 |',
        '| L0历史参考 | 49 | 0.070210 | 0.011939 | 0.025729 | 0.026081 | 0.414141 | — | 424,508 | 0.704 | 0.019251 |']
    for row in rows:
        lines.append(f"| {row['prototype']} | {row['best_epoch']} | {row['AP50']:.6f} | {row['AP75']:.6f} | {row['mAP50_95']:.6f} | {row['Precision']:.6f} | {row['Recall']:.6f} | {row['FP_per_image']:.3f} | {row['Params']:,} | {row['MACs']/1e9:.3f} | {row['endpoint_mAP50_95']:.6f} |")
    delta=comparisons['E1_minus_E0']
    lines+=['',f"E1−E0：mAP50:95 {delta['mAP50_95']:+.6f}，Recall {delta['Recall']:+.6f}，Precision {delta['Precision']:+.6f}，FP/图 {delta['FP_per_image']:+.3f}，额外 {delta['MACs']/1e9:.3f} G MAC。该差异对应新增 PAN 整个分支，不是单独相加连接。",'',
        '## 分类结果', '', '| 方案 | 类别 | AP50 | 平均AP | Precision | Recall | TP | FP | FN |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for prototype in ('E0','E1'):
        record=json.loads((RAW/'formal'/prototype/'seed_42/result.json').read_text(encoding='utf-8'))
        for name,count in record['supplemental']['per_class_precision_recall'].items():
            aps=record['supplemental']['all_and_small_area_ap']['per_class_ap']['all'][name]
            lines.append(f"| {prototype} | {name} | {aps['0.50']:.6f} | {sum(aps.values())/len(aps):.6f} | {count['Precision']:.6f} | {count['Recall']:.6f} | {count['true_positives']} | {count['false_positives']} | {count['ground_truths']-count['true_positives']} |")
    lines+=['','## 证据与解释边界','',
        'E0 是固定完整 B1＋四尺度 MBConv FPN＋直接1×1预测；E1 进一步增加三级 PAN。原144骨干池未改动。本轮没有训练 YOLOv8n，未改变标签、阈值或数据划分，也未进行量化、HLS、route（布局布线）、板端时延或功率测试。', '',
        'mAP 为10个IoU阈值、101点插值的冻结检测评估器结果；P/R 固定 score≥0.5、IoU≥0.5。官方 valid99 已用于每轮模型选择，因此这些是开发集结果。单seed用于候选筛选，不能证明统计显著性或正式部署可用性。历史B1/L0/L1是同协议历史参考，不是本轮新初始化的配对控制。', '',
        '源码与运行协议：[实验协议与运行命令](实验协议与运行命令.md)。结构来源：[作者结构到实验适配](作者结构到实验适配.md)。', '',
        '原始结果：`results/detector_elastic_pilot_20261005/remote_snapshot/raw_bytes/formal/{E0,E1}/seed_42/result.json`。检查点核验：`completion_checkpoint_proof.json`；完整CSV：`completed_accuracy.csv`；数值差异：`completed_comparison.json`。', '',
        f"执行清单 SHA256：`{proof['execution_manifest_sha256']}`。绑定源文件数：{len(manifest['source_sha256'])}。"]
    (REPORT/'完成结果与对照.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(status='PASS',rows=rows,comparisons=comparisons,report=str(REPORT/'完成结果与对照.md'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('checkpoint-proof','report'))
    args=parser.parse_args()
    print(json.dumps(remote_checkpoint_proof() if args.mode=='checkpoint-proof' else report(),indent=2,ensure_ascii=False))
