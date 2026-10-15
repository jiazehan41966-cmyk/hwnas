"""Build a traceable report from train/validation diagnostics and seeded runs."""
import argparse
import itertools
import json
from pathlib import Path
import statistics


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_report(root):
    root=Path(root).resolve()
    figures=root/"figures"
    figures.mkdir(exist_ok=True)
    diagnostics={}
    for name in ("baseline","noobj02"):
        p=root/f"legacy_{name}_diagnostics/diagnostics/diagnostics.json"
        if p.exists():
            diagnostics[name]=load(p)
    lines=["# 检测分支：逐类诊断与受控重复实验", "",
           "本报告只使用训练集和验证集。旧检查点用于解释现象，新运行用于比较背景负样本权重；两类结果不合并统计。",
           "", "## 完整数据诊断", "",
           "工作阈值由各检查点的验证集总体 F1 选择；这属于工作点诊断，不是独立泛化评估。",
           "", "| 检查点 | 阈值 | 数据划分 | 类别 | GT | TP | 漏检 | 背景误检 | 定位误检 | Recall |",
           "|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm,d in diagnostics.items():
        threshold=d["selected_validation_threshold"]["score_threshold"]
        for split in ("train","valid"):
            for name,c in d["splits"][split]["operating_errors"]["per_class"].items():
                lines.append(f"| {arm} | {threshold:.2f} | {split} | {name} | {c.get('targets',0)} | {c.get('true_positive',0)} | {c.get('missed',0)} | {c.get('background',0)} | {c.get('localization',0)} | {c['recall']:.4f} |")
    lines += ["", "背景外观按图像局部亮度/标准差分箱；textured 只是纹理较强的外观代理，不等同于人工确认的海底、阴影等语义标签。",
              "", "### 相同误检约束下的验证集召回率", "",
              "| 检查点 | FP/图上限 | 选取阈值 | 实际 FP/图 | Recall |",
              "|---|---:|---:|---:|---:|"]
    for arm,d in diagnostics.items():
        for limit,row in d["validation_recall_at_fp_per_image"].items():
            lines.append(f"| {arm} | {limit} | {row['threshold']:.2f} | {row['fp_per_image']:.4f} | {row['recall']:.4f} |")
    lines += ["", "## 相同预算的重复实验", "",
              "两组为 no_object_weight=0.1 与 0.2；每组 seeds=42/43/44，每次固定 80 epoch，保持训练数据、增强、锚框、网络、优化器和后处理一致。",
              "所有随机源重新固定；旧入口未显式固定模型初始化，因此旧 seed42 不作为本次配对样本。",
              "", "| 组别 | seed | epoch | 最佳验证 map(.50,.75) | 验证 AP50 | 验证 AP75 | 验证工作 F1 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    runs={}
    manifests={}
    for arm in ("baseline","noobj02"):
        for seed in (42,43,44):
            folder=root/"runs"/f"{arm}_seed{seed}"
            if not (folder/"status.json").exists():
                lines.append(f"| {arm} | {seed} | 待完成 | — | — | — | — |")
                continue
            status=load(folder/"status.json")
            if status["status"]!="COMPLETE":
                continue
            runs[(arm,seed)]=status
            manifests[(arm,seed)]=load(folder/"run_manifest.json")
            v=status["best_validation"]
            lines.append(f"| {arm} | {seed} | {status['epochs']} | {v['map']:.6f} | {v['map50']:.6f} | {v['map75']:.6f} | {status['validation_operating_point']['f1']:.6f} |")
    stats={"completed_runs":len(runs),"expected_runs":6,"test_accessed":False,
           "inference_scope":"optimization seed variability on fixed validation images; not dataset generalization"}
    if len(runs)==6:
        same_initial=all(manifests[("baseline",s)]["initial_state_sha256"]==manifests[("noobj02",s)]["initial_state_sha256"] for s in (42,43,44))
        same_data=len({json.dumps(m["data_bindings"],sort_keys=True) for m in manifests.values()})==1
        budgets=all(r["epochs"]==80 for r in runs.values())
        stats["paired_integrity"]={"same_initial_state_each_seed":same_initial,"same_data":same_data,"all_80_epochs":budgets}
        if not (same_initial and same_data and budgets):
            raise RuntimeError("Pairing integrity failed; no comparative conclusion is allowed")
        delta=[runs[("noobj02",s)]["best_validation"]["map"]-runs[("baseline",s)]["best_validation"]["map"] for s in (42,43,44)]
        mean=statistics.mean(delta)
        sd=statistics.stdev(delta)
        half=4.302652729911275*sd/(3**.5)
        perm=[abs(sum(x*y for x,y in zip(delta,sign))/3) for sign in itertools.product((-1,1),repeat=3)]
        stats["primary_delta"]={"mean":mean,"std":sd,"seed_deltas":delta,
            "seed_t95_interval":[mean-half,mean+half],"seed_t95_assumption":"approximately normal paired seed deltas; n=3, highly uncertain",
            "exact_sign_flip_two_sided_p":sum(x>=abs(mean)-1e-12 for x in perm)/len(perm),
            "positive_seeds":sum(x>0 for x in delta)}
        lines += ["",f"主比较 Δmap（0.2−0.1）={mean:+.6f}，配对差值标准差={sd:.6f}，正向种子 {sum(x>0 for x in delta)}/3。",
                  f"按种子计算的 t 95% 区间=[{mean-half:.6f}, {mean+half:.6f}]；精确双侧符号翻转 p={stats['primary_delta']['exact_sign_flip_two_sided_p']:.3f}。",
                  "只有 3 个种子，共用同一验证集；该区间只描述优化随机性，正态假设难以验证，不能作为数据泛化或统计显著提升的证明。"]
    else:
        lines += ["",f"当前完成 {len(runs)}/6。重复性结论待完整配对结果，不提前冻结最优配置。"]
    lines += ["", "## 后续决策边界", "",
        "- 不修改原数据与标注；不使用测试集选择权重、阈值或结构。",
        "- AP10 补充诊断使用 .50:.95 十档、all-points 积分；不是官方 COCO 101 点评测。旧 map 保持两档口径。",
        "- 区分背景误检、分类混淆、定位偏差、重复框与漏检；不把所有距离较远的误检归类为框回归问题。",
        "- stride=4 需要尺寸分组证据支持；本轮未改变固定 Neck/Head。",
        "- 声呐算子实验需先冻结检测对照协议；本轮没有开展算子搜索、硬件执行或功耗测量。",
        "", "## 复现入口与证据", "",
        "- study_manifest.json：有限实验矩阵。",
        "- snapshot/：本轮源代码快照；每次 run_manifest.json 绑定源码、配置、数据和初始权重。",
        "- legacy_*_diagnostics/diagnostics/：完整 train/valid 预测、错误统计与代表图。",
        "- runs/*/：80 epoch 历史、验证预测、检查点与完成状态。"]
    (root/"analysis-report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (root/"stats-appendix.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    if diagnostics:
        draw_figures(diagnostics,figures)
    return stats


def draw_figures(diags,folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,split in zip(axes,("train","valid")):
        names=["cylider","manta"]
        for i,(arm,d) in enumerate(diags.items()):
            c=d["splits"][split]["operating_errors"]["per_class"]
            ax.bar(np.arange(2)+i*.32,[c[n]["recall"] for n in names],width=.3,label=arm)
        ax.set_xticks(np.arange(2)+.16,names)
        ax.set_ylim(0,1)
        ax.set_title(split+": recall at validation-selected F1 threshold")
        ax.set_ylabel("Recall @ IoU=.50")
        ax.legend()
    fig.savefig(folder/"per_class_recall.png",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,len(diags),figsize=(5*len(diags),4),squeeze=False,constrained_layout=True)
    for ax,(arm,d) in zip(axes[0],diags.items()):
        conf=d["splits"]["valid"]["low_threshold_errors"]["confidence"]
        for name,groups in conf.items():
            for kind in ("true_positive","background"):
                if kind in groups:
                    h=np.asarray(groups[kind]["histogram_0_to_1_step_01"])
                    ax.plot(np.arange(10)*.1+.05,h,marker="o",label=name+" / "+kind)
        ax.set_yscale("symlog",linthresh=1)
        ax.set_xlabel("Prediction score")
        ax.set_ylabel("Predictions (symlog)")
        ax.set_title(arm+": validation confidence")
        ax.legend(fontsize=8)
    fig.savefig(folder/"confidence_histograms.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    print(write_report(p.parse_args().root))
