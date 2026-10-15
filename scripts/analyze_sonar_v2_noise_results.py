#!/usr/bin/env python3
"""Build the strict analysis bundle for the v2 sonar synthetic-noise results."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality-summary",
        default="results/sonar_v2_synthetic_quality_20260815/summary.csv",
    )
    parser.add_argument(
        "--network-per-run",
        default="results/g5v2_noise_robustness_20260815/per_run.csv",
    )
    parser.add_argument(
        "--network-summary",
        default="results/g5v2_noise_robustness_20260815/summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/sonar_v2_noise_analysis_20260815",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def configure_chinese_font() -> str:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in installed), "DejaVu Sans")
    plt.rcParams["font.family"] = selected
    plt.rcParams["axes.unicode_minus"] = False
    return selected


def exact_sign_flip_p(deltas: np.ndarray) -> float:
    observed = abs(float(deltas.mean()))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        statistic = abs(float(np.mean(deltas * np.asarray(signs))))
        extreme += int(statistic >= observed - 1e-15)
        total += 1
    return extreme / total


def bootstrap_fold_ci(deltas: np.ndarray, *, resamples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(resamples, len(deltas)))
    estimates = deltas[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(rows: list[dict[str, Any]], key: str = "p_value") -> None:
    order = sorted(range(len(rows)), key=lambda index: float(rows[index][key]))
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (total - rank) * float(rows[index][key]))
        running = max(running, adjusted)
        rows[index]["p_value_holm"] = running


def analyze_network(per_run: list[dict[str, str]], resamples: int, seed: int) -> list[dict[str, Any]]:
    by_key = {
        (row["method"], int(row["fold"]), int(row["seed"]), row["condition"]): row
        for row in per_run
    }
    conditions = ["speckle_var_0p01", "speckle_var_0p04", "speckle_var_0p09"]
    output: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(conditions):
        pair_deltas: list[float] = []
        fold_values: dict[int, list[float]] = defaultdict(list)
        for fold in range(5):
            for run_seed in (42, 43, 44):
                adaptive = float(by_key[("adaptive_denoise_v2", fold, run_seed, condition)]["macro_f1"])
                control = float(by_key[("control", fold, run_seed, condition)]["macro_f1"])
                delta = adaptive - control
                pair_deltas.append(delta)
                fold_values[fold].append(delta)
        fold_deltas = np.asarray(
            [np.mean(fold_values[fold]) for fold in sorted(fold_values)], dtype=np.float64
        )
        pair_array = np.asarray(pair_deltas, dtype=np.float64)
        ci_low, ci_high = bootstrap_fold_ci(
            fold_deltas,
            resamples=resamples,
            seed=seed + condition_index,
        )
        fold_std = float(fold_deltas.std(ddof=1))
        output.append(
            {
                "condition": condition,
                "variance": [0.01, 0.04, 0.09][condition_index],
                "paired_runs": 15,
                "independent_folds": 5,
                "paired_delta_mean": float(pair_array.mean()),
                "paired_delta_std": float(pair_array.std()),
                "fold_mean_delta": float(fold_deltas.mean()),
                "fold_ci95_low": ci_low,
                "fold_ci95_high": ci_high,
                "positive_folds": int(np.sum(fold_deltas > 0)),
                "pair_wins": int(np.sum(pair_array > 0)),
                "paired_effect_dz_fold": (
                    float(fold_deltas.mean() / fold_std) if fold_std > 0 else math.inf
                ),
                "p_value": exact_sign_flip_p(fold_deltas),
            }
        )
    holm_adjust(output)
    return output


def plot_quality(rows: list[dict[str, str]], output_path: Path) -> None:
    datasets = ["nksid", "roboflow", "figshare"]
    dataset_labels = {"nksid": "现有八类数据", "roboflow": "Roboflow", "figshare": "Figshare"}
    methods = ["noisy_input", "v1_fixed_smooth", "v2_adaptive_projection"]
    method_labels = {
        "noisy_input": "仅加噪",
        "v1_fixed_smooth": "旧版固定平滑",
        "v2_adaptive_projection": "新版自适应降噪",
    }
    colors = {"noisy_input": "#777777", "v1_fixed_smooth": "#D97706", "v2_adaptive_projection": "#087E8B"}
    variances = [0.01, 0.04, 0.09]
    by_key = {
        (row["dataset"], float(row["variance"]), row["method"]): row for row in rows
    }
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharex=True)
    for column, dataset in enumerate(datasets):
        for method in methods:
            psnr = [float(by_key[(dataset, variance, method)]["psnr_mean"]) for variance in variances]
            ssim = [float(by_key[(dataset, variance, method)]["ssim_mean"]) for variance in variances]
            axes[0, column].plot(variances, psnr, marker="o", linewidth=2.0, color=colors[method], label=method_labels[method])
            axes[1, column].plot(variances, ssim, marker="o", linewidth=2.0, color=colors[method], label=method_labels[method])
        axes[0, column].set_title(dataset_labels[dataset], fontsize=12)
        axes[0, column].grid(alpha=0.25)
        axes[1, column].grid(alpha=0.25)
        axes[1, column].set_xlabel("噪声方差")
    axes[0, 0].set_ylabel("峰值信噪比")
    axes[1, 0].set_ylabel("结构相似度")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("三套声呐数据的加噪图像质量", y=1.04, fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_network(rows: list[dict[str, str]], output_path: Path) -> None:
    methods = ["control", "adaptive_denoise_v2", "edge_v2", "adaptive_denoise_edge_v2"]
    labels = {
        "control": "普通网络",
        "adaptive_denoise_v2": "新版自适应降噪",
        "edge_v2": "新版边缘增强",
        "adaptive_denoise_edge_v2": "降噪与边缘增强",
    }
    colors = {
        "control": "#555555",
        "adaptive_denoise_v2": "#087E8B",
        "edge_v2": "#D97706",
        "adaptive_denoise_edge_v2": "#9B5DE5",
    }
    condition_order = ["clean", "speckle_var_0p01", "speckle_var_0p04", "speckle_var_0p09"]
    x = np.arange(4)
    by_key = {(row["method"], row["condition"]): row for row in rows}
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    for method in methods:
        means = [float(by_key[(method, condition)]["macro_f1_mean"]) for condition in condition_order]
        stds = [float(by_key[(method, condition)]["macro_f1_std"]) for condition in condition_order]
        ax.errorbar(x, means, yerr=stds, marker="o", linewidth=2.0, capsize=3, color=colors[method], label=labels[method])
    ax.set_xticks(x, ["无噪声", "弱噪声", "中噪声", "强噪声"])
    ax.set_ylabel("各类别平均识别得分")
    ax.set_title("完整识别网络的加噪稳定性（15次训练平均）")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> int:
    args = parse_args()
    quality_rows = read_csv(Path(args.quality_summary))
    network_per_run = read_csv(Path(args.network_per_run))
    network_summary = read_csv(Path(args.network_summary))
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    selected_font = configure_chinese_font()

    comparisons = analyze_network(
        network_per_run,
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    plot_quality(quality_rows, figure_dir / "figure-01-three-dataset-image-quality.png")
    plot_network(network_summary, figure_dir / "figure-02-network-noise-robustness.png")

    quality_key = {
        (row["dataset"], row["noise_level"], row["method"]): row for row in quality_rows
    }
    report = [
        "# 新版声呐算子加噪结果分析",
        "",
        "## 分析问题",
        "",
        "1. 新版自适应降噪在三套数据上能否改善加噪图像的峰值信噪比和结构相似度？",
        "2. 加入新版算子后，完整识别网络是否比普通网络更抗噪？",
        "",
        "## 主要结果",
        "",
        "- 三套数据共 4265 张图像，采用弱、中、强三级乘性斑点噪声。",
        "- 新版自适应降噪在三套数据的弱噪声条件下均优于旧版固定平滑。",
        "- 中强噪声下并非全面占优：Roboflow 的中强噪声以及现有八类数据的强噪声，旧版固定平滑的峰值信噪比更高。",
        "- 完整网络的 5 份数据划分 x 3 次训练表明，新版自适应降噪网络在三级噪声下均高于普通网络。",
        "- 新版边缘增强网络在干净图和噪声图上都低于普通网络，不能作为正向贡献。",
        "",
        "## 图像质量核心表",
        "",
        "| 数据集 | 噪声 | 旧版峰值信噪比 | 新版峰值信噪比 | 新版减旧版 | 旧版结构相似度 | 新版结构相似度 | 新版减旧版 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, dataset_label in [("nksid", "现有八类数据"), ("roboflow", "Roboflow"), ("figshare", "Figshare")]:
        for level, level_label in [("weak", "弱"), ("medium", "中"), ("strong", "强")]:
            old = quality_key[(dataset, level, "v1_fixed_smooth")]
            new = quality_key[(dataset, level, "v2_adaptive_projection")]
            report.append(
                f"| {dataset_label} | {level_label} | {fmt(float(old['psnr_mean']))} | {fmt(float(new['psnr_mean']))} | "
                f"{fmt(float(new['psnr_mean']) - float(old['psnr_mean']))} | {fmt(float(old['ssim_mean']))} | "
                f"{fmt(float(new['ssim_mean']))} | {fmt(float(new['ssim_mean']) - float(old['ssim_mean']))} |"
            )
    report += [
        "",
        "## 完整网络抗噪结果",
        "",
        "下表为现有八类数据上15次训练的平均值，括号内为15次结果的标准差。",
        "",
        "| 网络 | 无噪声 | 弱噪声 | 中噪声 | 强噪声 |",
        "|---|---:|---:|---:|---:|",
    ]
    network_key = {(row["method"], row["condition"]): row for row in network_summary}
    condition_order = ["clean", "speckle_var_0p01", "speckle_var_0p04", "speckle_var_0p09"]
    for method, method_label in [
        ("control", "普通网络"),
        ("adaptive_denoise_v2", "新版自适应降噪"),
        ("edge_v2", "新版边缘增强"),
        ("adaptive_denoise_edge_v2", "降噪与边缘增强"),
    ]:
        values = []
        for condition in condition_order:
            row = network_key[(method, condition)]
            values.append(f"{fmt(float(row['macro_f1_mean']))} ± {fmt(float(row['macro_f1_std']))}")
        report.append(f"| {method_label} | " + " | ".join(values) + " |")
    report += [
        "",
        "## 结论边界",
        "",
        "- 图像质量中的新版降噪结果，是把15个已训练模型中的降噪核心投影到单通道图像后得到，不是完整分类网络输出。",
        "- 新版边缘结果只能作为图像可视化近似，不用于证明去噪质量。",
        "- 完整网络抗噪结果目前只覆盖现有八类数据。Roboflow 和 Figshare 必须先训练各自类别的网络，才能进行同口径完整网络比较。",
    ]
    (output_dir / "analysis-report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    stats = [
        "# 配对统计附录",
        "",
        "比较对象：新版自适应降噪网络减普通网络；主指标为各类别平均识别得分。",
        "每个噪声条件包含15个配对结果，但三个重复训练共享同一份外部验证图，因此推断统计以5份数据划分为独立单位。",
        "置信区间通过按数据划分重抽样10000次得到；显著性采用5份划分的精确双侧符号翻转检验，并对三级噪声进行Holm校正。",
        "",
        "| 噪声 | 平均提升 | 95%区间 | 提升的数据划分数 | 15次中胜出数 | 校正后概率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row, label in zip(comparisons, ("弱", "中", "强")):
        stats.append(
            f"| {label} | {fmt(row['paired_delta_mean'])} | [{fmt(row['fold_ci95_low'])}, {fmt(row['fold_ci95_high'])}] | "
            f"{row['positive_folds']}/5 | {row['pair_wins']}/15 | {fmt(row['p_value_holm'])} |"
        )
    stats += [
        "",
        "校正后的概率没有低于0.05，因此可以写成稳定的描述性优势，但暂不写成统计显著优势。",
    ]
    (output_dir / "stats-appendix.md").write_text("\n".join(stats) + "\n", encoding="utf-8")

    catalog = [
        "# 图表目录",
        "",
        "## 图1：三套声呐数据的加噪图像质量",
        "",
        "- 文件：`figures/figure-01-three-dataset-image-quality.png`",
        "- 用途：比较仅加噪、旧版固定平滑和新版自适应降噪。",
        "- 观察重点：新版在弱噪声下稳定保结构，但强噪声下不总是优于旧版固定平滑。",
        "- 结论影响：新版贡献应表述为自适应保结构，而不是所有噪声下绝对最优。",
        "",
        "## 图2：完整识别网络的加噪稳定性",
        "",
        "- 文件：`figures/figure-02-network-noise-robustness.png`",
        "- 用途：比较普通网络、新版自适应降噪、新版边缘增强及两者组合。",
        "- 观察重点：新版自适应降噪随噪声增强保持领先，边缘增强整体较低。",
        "- 结论影响：保留新版自适应降噪作为鲁棒性候选，继续排除边缘增强。",
        "",
        f"绘图字体：`{selected_font}`。",
    ]
    (output_dir / "figure-catalog.md").write_text("\n".join(catalog) + "\n", encoding="utf-8")
    (output_dir / "statistics.json").write_text(
        json.dumps({"network_comparisons": comparisons}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
