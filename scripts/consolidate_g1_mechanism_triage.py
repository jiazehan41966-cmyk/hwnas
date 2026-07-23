#!/usr/bin/env python3
"""Consolidate the staged G1 mechanism diagnostics into one audit bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "curves": (
        "artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/diagnostics/"
        "g1_capacity_diagnostic.json"
    ),
    "clean_fit": (
        "artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/diagnostics/"
        "g1_checkpoint_clean_train_fit_v1.json"
    ),
    "micro": (
        "artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/diagnostics/"
        "g1_micro_overfit_v1.json"
    ),
    "full": (
        "artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/diagnostics/"
        "g1_full_train_recipe_triage_fold0_seed42_v1.json"
    ),
    "record": (
        "results/protocol/g1_clean_20260711/g1_rl_arch_135_legacy_selected/"
        "run_fold0_seed42.json"
    ),
    "output": (
        "artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/diagnostics/"
        "g1_mechanism_triage_v3"
    ),
}


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO / candidate


def load(path: str) -> dict:
    target = resolve(path)
    if not target.exists():
        raise FileNotFoundError(target)
    return json.loads(target.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_values(clean_fit: dict, metric: str) -> tuple[np.ndarray, np.ndarray]:
    tags = clean_fit["selected_fold_seed_pairs"]
    nas = np.asarray(
        [clean_fit["methods"]["nas_rl_arch_135"][tag]["metrics"][metric] for tag in tags]
    )
    scratch = np.asarray(
        [clean_fit["methods"]["mnv2_scratch"][tag]["metrics"][metric] for tag in tags]
    )
    return nas, scratch


def make_figures(clean_fit: dict, micro: dict, full: dict, output: Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
    catalog = []

    nas, scratch = paired_values(clean_fit, "top1")
    fig, ax = plt.subplots(figsize=(6.2, 4.5), constrained_layout=True)
    for index in range(len(nas)):
        ax.plot([0, 1], [nas[index], scratch[index]], color="#999999", alpha=0.55, linewidth=0.9)
        ax.scatter([0, 1], [nas[index], scratch[index]], color=["#D55E00", "#0072B2"], s=24)
    ax.set(
        xticks=[0, 1],
        xticklabels=["rl_arch_135", "MNV2 scratch"],
        ylabel="Clean eval-mode accuracy on each run's training indices",
        ylim=(0, 1.03),
    )
    ax.grid(axis="y", alpha=0.2)
    path = figures_dir / "figure-01-clean-train-fit-paired.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    catalog.append(
        {
            "id": "Figure 1",
            "path": path.relative_to(REPO).as_posix(),
            "caption": "Paired deterministic clean-training fit across all 15 fold/seed runs.",
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for name, variant in micro["variants"].items():
        epochs = [row["epoch"] for row in variant["history"]]
        values = [row["eval_mode_top1"] for row in variant["history"]]
        ax.plot(epochs, values, linewidth=1.5, label=name.replace("_", " "))
    ax.axhline(0.99, color="#555555", linestyle="--", linewidth=1, label="0.99 gate")
    ax.set(xlabel="Epoch", ylabel="Clean subset eval-mode top-1", ylim=(0, 1.03))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    path = figures_dir / "figure-02-micro-overfit-dynamics.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    catalog.append(
        {
            "id": "Figure 2",
            "path": path.relative_to(REPO).as_posix(),
            "caption": "Fresh-model micro-overfit dynamics on the fixed 96-sample balanced subset.",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    for name, variant in full["variants"].items():
        eval_rows = [row for row in variant["history"] if "clean_train_eval" in row]
        epochs = [row["epoch"] for row in eval_rows]
        clean = [row["clean_train_eval"]["macro_f1"] for row in eval_rows]
        inner = [row["inner_eval"]["macro_f1"] for row in eval_rows]
        label = name.replace("clean_input_", "").replace("_", " ")
        axes[0].plot(epochs, clean, linewidth=1.5, label=label)
        axes[1].plot(epochs, inner, linewidth=1.5, label=label)
    axes[0].axhline(0.95, color="#555555", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Epoch", ylabel="Clean full-train macro F1", ylim=(0, 1.03))
    axes[1].set(xlabel="Epoch", ylabel="Inner-validation macro F1", ylim=(0, 1.03))
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    path = figures_dir / "figure-03-full-train-recipe-triage.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    catalog.append(
        {
            "id": "Figure 3",
            "path": path.relative_to(REPO).as_posix(),
            "caption": (
                "Single-fold full-training recipe triage. Outer validation was never consumed; "
                "inner-validation curves are diagnostic only."
            ),
        }
    )
    return catalog


def derive_decision(full: dict) -> dict:
    status = full["decision"]["status"]
    if "SENSITIVITY_CONFIRMED" in status:
        return {
            "status": "CAPACITY_SWEEP_HOLD__RECIPE_FACTOR_VALIDATION_NEXT",
            "reason": (
                "单折诊断中，历史架构在至少一种修改配方下可以充分拟合完整 clean 训练索引；"
                "因此通用表达容量尚未被隔离为一阶机制"
            ),
            "next_gate": (
                "在 fold0 的 seed43、44 上重复最小配方因子对照；然后再决定是否值得投入"
                "正式 15-run 协议比较"
            ),
            "distillation": "HOLD，等待配方因子验证",
            "capacity_sweep": "HOLD",
            "preprocessing_four_arm": "HOLD，等待源图/采集场次分组 split 合约",
        }
    if full["decision"].get("capacity_sweep_gate") == "PROVISIONAL_PASS":
        return {
            "status": "CAPACITY_SWEEP_PROVISIONAL_PASS",
            "reason": (
                "单折诊断中，clean 输入、去正则与学习率括区间均未恢复完整训练拟合"
            ),
            "next_gate": "预注册模型规模点，并在一个冻结配方下执行容量扫描",
            "distillation": "HOLD，等待容量曲线",
            "capacity_sweep": "PROVISIONAL_PASS",
            "preprocessing_four_arm": "HOLD，等待源图/采集场次分组 split 合约",
        }
    return {
        "status": "MECHANISM_AMBIGUOUS__NO_LARGE_CAMPAIGN",
        "reason": "单折全训练门诊未跨过预注册的决策阈值",
        "next_gate": "在另外两个 seed 上重复最佳诊断臂",
        "distillation": "HOLD",
        "capacity_sweep": "HOLD",
        "preprocessing_four_arm": "HOLD，等待源图/采集场次分组 split 合约",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("curves", "clean_fit", "micro", "full", "record", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", default=DEFAULTS[name])
    args = parser.parse_args()

    curves = load(args.curves)
    clean_fit = load(args.clean_fit)
    micro = load(args.micro)
    full = load(args.full)
    record = load(args.record)
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    figures = make_figures(clean_fit, micro, full, output)
    decision = derive_decision(full)

    online = curves["primary_paired_analysis"]
    online_delta = online["paired_delta_scratch_minus_nas"]
    top1 = clean_fit["paired_summaries"]["top1"]
    macro = clean_fit["paired_summaries"]["macro_f1"]
    micro_rows = []
    for name, variant in micro["variants"].items():
        micro_rows.append(
            f"| {name} | {variant['best_eval_top1']:.4f} | "
            f"{variant['best_eval_macro_f1']:.4f} | "
            f"{variant['first_epoch_eval_top1_ge_0_99'] or '未达到'} |"
        )
    full_rows = []
    for name, variant in full["variants"].items():
        full_rows.append(
            f"| {name} | {variant['best_clean_train_top1']:.4f} | "
            f"{variant['best_clean_train_macro_f1']:.4f} | "
            f"{variant['best_inner_macro_f1']:.4f} | {variant['epochs_executed']} |"
        )

    report = f"""# G1 机制诊断总报告

## 结论

最终门禁：**{decision['status']}**。

训练曲线与 15 对 checkpoint 重评确认 `rl_arch_135` 在当前冻结配方下存在真实训练侧
拟合差距；但这不等价于已经隔离「参数容量」。micro-overfit 排除了小子集上的粗粒度
实现/可训练性故障；单折全训练因子实验的状态为
`{full['decision']['status']}`。因此当前动作是：{decision['next_gate']}。

## 证据阶梯

1. **在线增强训练曲线（15 对完整运行）**：NAS={online['nas_mean']:.4f}，
   scratch={online['scratch_mean']:.4f}，配对差={online_delta['mean']:.4f}，
   95% CI [{online_delta['bootstrap_95_ci'][0]:.4f}, {online_delta['bootstrap_95_ci'][1]:.4f}]。
   该指标受随机增强与 train mode 影响。
2. **best checkpoint 的无增强训练索引重评（15 对）**：clean top-1 差
   {top1['delta_scratch_minus_nas_mean']:.4f}，95% CI
   [{top1['bootstrap_95_ci'][0]:.4f}, {top1['bootstrap_95_ci'][1]:.4f}]；clean macro_f1
   差 {macro['delta_scratch_minus_nas_mean']:.4f}。真实 clean-fit gap 成立。
3. **96 张固定类平衡子集**：至少一个新鲜 NAS 模型达到 0.99 以上，说明该架构并非
   连小样本都无法记忆；学习率存在可见敏感性。
4. **全训练索引的单折三臂诊断**：只使用 fold0/seed42 的 train 与 inner，未访问 outer。
   它只能定位机制并决定下一门禁，不能产生正式泛化结论。

## micro-overfit 结果

| 变体 | best clean top-1 | best clean macro_f1 | 首次 top-1≥0.99 epoch |
|---|---:|---:|---:|
{chr(10).join(micro_rows)}

## 单折全训练因子结果

| 变体 | best clean top-1 | best clean macro_f1 | best inner macro_f1 | epochs |
|---|---:|---:|---:|---:|
{chr(10).join(full_rows)}

历史原配方 fold0/seed42 best checkpoint：clean top-1
{full['baseline']['best_checkpoint_clean_train_top1']:.4f}，clean macro_f1
{full['baseline']['best_checkpoint_clean_train_macro_f1']:.4f}；原记录 inner macro_f1
{float(record['inner_val']['macro_f1']):.4f}（不同训练过程，仅作同一运行背景）。

## 实验决策

- capacity sweep（容量扫描）：**{decision['capacity_sweep']}**。
- distillation（知识蒸馏）：**{decision['distillation']}**。
- 四臂预处理：**{decision['preprocessing_four_arm']}**。
- 下一门禁：{decision['next_gate']}。
"""
    appendix = f"""# 统计附录

## 15 对 clean checkpoint 重评

| 指标 | NAS mean | scratch mean | paired delta | bootstrap 95% CI | dz | exact sign-flip p |
|---|---:|---:|---:|---|---:|---:|
| top-1 | {top1['nas_mean']:.6f} | {top1['scratch_mean']:.6f} | {top1['delta_scratch_minus_nas_mean']:.6f} | [{top1['bootstrap_95_ci'][0]:.6f}, {top1['bootstrap_95_ci'][1]:.6f}] | {top1['paired_cohens_dz']:.3f} | {top1['exact_two_sided_sign_flip_p']:.8f} |
| macro_f1 | {macro['nas_mean']:.6f} | {macro['scratch_mean']:.6f} | {macro['delta_scratch_minus_nas_mean']:.6f} | [{macro['bootstrap_95_ci'][0]:.6f}, {macro['bootstrap_95_ci'][1]:.6f}] | {macro['paired_cohens_dz']:.3f} | {macro['exact_two_sided_sign_flip_p']:.8f} |

分析单位是 `(fold, seed)`，先在每次运行内汇总，再做 15 对配对推断。bootstrap 使用
50,000 次配对重采样，seed=20260722；精确检验枚举全部 2^15 个符号翻转。

micro-overfit 与单折全训练因子实验是机制诊断，不做跨运行推断，不报告 p 值，也不得
作为 outer-fold 泛化性能。其阈值是预先写入脚本的操作性门禁，而非统计显著性阈值。
"""
    catalog_lines = ["# Figure catalog", ""]
    for figure in figures:
        catalog_lines.extend(
            [f"## {figure['id']}", "", f"- File: `{figure['path']}`", f"- {figure['caption']}", ""]
        )
    source_paths = {
        name: resolve(getattr(args, name)) for name in ("curves", "clean_fit", "micro", "full", "record")
    }
    generator_paths = [
        REPO / "scripts" / "diagnose_g1_capacity.py",
        REPO / "scripts" / "evaluate_g1_checkpoint_train_fit.py",
        REPO / "scripts" / "diagnose_g1_micro_overfit.py",
        REPO / "scripts" / "diagnose_g1_full_train_recipe.py",
        Path(__file__).resolve(),
    ]
    provenance = {
        "schema_version": 1,
        "decision": decision,
        "source_artifacts": {
            name: {"path": path.relative_to(REPO).as_posix(), "sha256": sha256(path)}
            for name, path in source_paths.items()
        },
        "generators": [
            {"path": path.relative_to(REPO).as_posix(), "sha256": sha256(path)}
            for path in generator_paths
        ],
        "figures": figures,
    }
    (output / "analysis-report.md").write_text(report, encoding="utf-8")
    (output / "stats-appendix.md").write_text(appendix, encoding="utf-8")
    (output / "figure-catalog.md").write_text("\n".join(catalog_lines), encoding="utf-8")
    (output / "decision.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
