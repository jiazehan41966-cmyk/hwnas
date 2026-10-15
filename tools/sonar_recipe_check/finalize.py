"""Finish the registered nine-run check by replaying evidence and writing D14."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.sonar_recipe_check import train,analysis
from tools.sonar_recipe_check.evidence import verify_r0_artifacts

def report_markdown(report):
    d=report['decision'];primary=report['subsets']['full99']
    lines=['# D14 训练配方检查：最终选择','',f"选择：**{d['selected_recipe']}**；状态：{d['decision_status']}。",'',
        '固定 YOLOv9-T＋quality_aware；R0 复用 C01 三次运行，R1–R3 新增九次运行。全部 seed 为 42–44，每次 80 轮。',
        '主口径为 full99 的 mAP50–95，第 76–80 轮算术均值，再对三个 seed 求均值。下表百分数为 AP×100，差值及区间为百分点。','',
        '| 配方 | 稳定 mAP | 相对 R0 | 95% 分层配对区间 | 有效 |', '|---|---:|---:|---|---|']
    for recipe in analysis.RECIPES:
        mean=d['stable_mean_by_recipe'][recipe]
        if recipe=='R0':lines.append(f'| R0 | {100*mean:.4f}% | — | — | 基线 |')
        else:
            g=d['candidate_gates'][recipe];ci=g['hierarchical_ci']
            lines.append(f"| {recipe} | {100*mean:.4f}% | {g['delta_percentage_points']:+.4f} | [{100*ci['low']:.4f}, {100*ci['high']:.4f}] | {'是' if g['effective'] else '否'} |")
    lines += ['', '有效条件同时要求相对 R0 提升至少 1 个百分点、95% 分层配对区间下限严格大于 0。有效候选中，选择距最好有效候选不超过 0.5 个百分点的最简单配方；都无效时保留 R0。',
      '', 'R1：EMA（模型权重指数滑动平均）；R2：R1＋3 轮预热及余弦下降；R3：R2＋mosaic 和作者仿射尺度抖动的受控增量。检测框数据无分割轮廓，复制粘贴关闭。',
      '', '区间使用训练 seed 与验证图像分层配对抽样，2000 次，随机种子 20261013。三个预登记比较的区间未作多重比较校正；这是验证集开发选择，不能表述为独立测试泛化结论。89 图子集仅作辅助。',
      '', '网络推理结构固定。此次不新增资源、板端时延和功耗测量结论，也不启动 72/192 搜索。',
      '', '历史 11.66% 是单 seed 的 best；对应第 76–80 轮均值为 7.280218%。历史运行同样不是完整作者训练配方，因此不能把与 C01 的差异归因于配方。',
      '', '完整逐 seed、辅助子集、终轮与 best 兼容指标见 recipe_analysis.json、compatibility_metrics.csv；归档与源码核验见 source_verification.json、completion_receipt.json。']
    return '\n'.join(lines)+'\n'

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output;out=root/'finalization'
    train.torch.set_num_threads(2)
    statuspath=root/'finalization_status.json'
    status=dict(status='RUNNING',started_at=train.p.now(),test_evaluated=False)
    train.p.dump(statuspath,status)
    try:
        cfg,baseline=train.verify_manifest(root)
        queue=train.read_json(root/'queue_status.json')
        if queue['status']!='COMPLETE' or queue['completed']!=[list(x) for x in train.queue_jobs()]:raise ValueError('All nine runs required')
        gate=train.read_gate(cfg);r0files=verify_r0_artifacts(cfg,gate)
        # Independently replay checkpoint histories and archived predictions before bootstrap.
        for recipe,seed in train.queue_jobs():train.run_result(root,recipe,seed)
        # R0's original manifest must be verified in its own unchanged source root.
        # Here its original result SHA and five prediction SHAs bind the prior replayed artifacts.
        rows=analysis.load_rows(cfg['r0_root'],root)
        report=analysis.recipe_summary(rows,cfg['decision'],subset_contract=train.old.load_subset_contract(cfg),output_dir=out)
        train.verify_manifest(root);verify_r0_artifacts(cfg,gate)
        receipt=dict(status='PASS',completed_at=train.p.now(),execution_manifest_sha256=train.p.sha(root/'execution_manifest.json'),
            preregistration_sha256=cfg['preregistration_sha256'],r0_reuse_gate_sha256=cfg['r0_reuse_gate_sha256'],
            r0_artifacts=r0files,recipe_analysis_sha256=train.p.sha(out/'recipe_analysis.json'),
            archived_predictions=60,records=5940,new_runs=9,reused_runs=3,
            selected_recipe=report['decision']['selected_recipe'],test_evaluated=False)
        train.p.dump(out/'completion_receipt.json',receipt)
        train.p.dump(out/'D14_selected_recipe.json',dict(**report['decision'],completion_receipt=receipt))
        (out/'D14_training_recipe.md').write_text(report_markdown(report),encoding='utf-8')
        status.update(status='COMPLETE',selected_recipe=receipt['selected_recipe'],completed_at=train.p.now(),output=str(out))
        train.p.dump(statuspath,status);print(json.dumps(status,ensure_ascii=False))
    except Exception as error:
        status.update(status='FAILED_NO_AUTORETRY',error=str(error),error_type=type(error).__name__,updated_at=train.p.now())
        train.p.dump(statuspath,status);raise

if __name__=='__main__':main()
