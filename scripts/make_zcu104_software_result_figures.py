"""Create software-side result figures for the ZCU104 MobileNetA RL campaign."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RETRAIN_CSV = (
    ROOT
    / "artifacts"
    / "zcu104_full_campaign_v1"
    / "retrain_protocol_v1_summaries"
    / "zcu104_mobilenetA_retrain_protocol_v1_summary.csv"
)
PTQ_CSV = (
    ROOT
    / "artifacts"
    / "zcu104_full_campaign_v1"
    / "int8_complexity"
    / "mobilenetA_rl200_20260908_1640"
    / "zcu104_mobilenetA_ptq_complexity_summary.csv"
)
SEARCH_METRICS_JSON = (
    ROOT
    / "artifacts"
    / "zcu104_full_campaign_v1"
    / "search_launch"
    / "rl_search_metrics_rl200_20260908_155713.json"
)
SEARCH_CURVE_CSV = (
    ROOT
    / "artifacts"
    / "zcu104_full_campaign_v1"
    / "search_launch"
    / "rl_search_curve_rl200_20260908_155713.csv"
)
PER_CLASS_CSV = (
    ROOT
    / "artifacts"
    / "zcu104_full_campaign_v1"
    / "software_metrics_tables"
    / "software_metrics_per_class_f1.csv"
)
OUT_DIR = (
    ROOT
    / "artifacts"
    / "zcu104_full_campaign_v1"
    / "software_metrics_visualization"
)

PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#6B7280",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> list[str]:
    paths: list[str] = []
    for suffix in (".png", ".pdf"):
        path = OUT_DIR / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    retrain = pd.read_csv(RETRAIN_CSV)
    ptq = pd.read_csv(PTQ_CSV)
    per_class = pd.read_csv(PER_CLASS_CSV)
    search_curve = pd.read_csv(SEARCH_CURVE_CSV)
    search_metrics = json.loads(SEARCH_METRICS_JSON.read_text(encoding="utf-8"))
    merged = retrain.merge(ptq, on=["role", "arch_id", "run_name"], suffixes=("", "_ptq"))
    merged = merged.sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    return merged, per_class, search_curve, search_metrics, retrain


def fig_classification_performance(df: pd.DataFrame) -> list[str]:
    x = np.arange(len(df))
    width = 0.34
    labels = df["arch_id"].tolist()
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.bar(
        x - width / 2,
        df["macro_f1_mean"],
        width,
        yerr=df["macro_f1_std"],
        label="Macro-F1",
        color=PALETTE["blue"],
        capsize=3,
        linewidth=0,
    )
    ax.bar(
        x + width / 2,
        df["top1_mean"],
        width,
        yerr=df["top1_std"],
        label="Top-1",
        color=PALETTE["orange"],
        capsize=3,
        linewidth=0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Software FP32 classification performance")
    ax.legend(frameon=False, ncols=2)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    return save_figure(fig, "fig01_fp32_classification_performance")


def fig_int8_drop(df: pd.DataFrame) -> list[str]:
    x = np.arange(len(df))
    width = 0.34
    labels = df["arch_id"].tolist()
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.bar(
        x - width / 2,
        df["int8_macro_f1_drop_mean"],
        width,
        label="Macro-F1 drop",
        color=PALETTE["green"],
        linewidth=0,
    )
    ax.bar(
        x + width / 2,
        df["int8_top1_drop_mean"],
        width,
        label="Top-1 drop",
        color=PALETTE["purple"],
        linewidth=0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("FP32 - INT8 score")
    ax.set_title("PTQ fake-quant software change")
    ax.legend(frameon=False, ncols=2)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    return save_figure(fig, "fig02_int8_fake_quant_drop")


def fig_complexity_tradeoff(df: pd.DataFrame) -> list[str]:
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    sizes = 450 * (df["params"] / df["params"].max())
    ax.scatter(
        df["macs"] / 1e6,
        df["macro_f1_mean"],
        s=sizes,
        color=PALETTE["sky"],
        edgecolor="#1F2937",
        linewidth=0.7,
        alpha=0.9,
    )
    for _, row in df.iterrows():
        ax.annotate(
            row["arch_id"].replace("rl_arch_", "#"),
            (row["macs"] / 1e6, row["macro_f1_mean"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("MACs (M)")
    ax.set_ylabel("FP32 Macro-F1")
    ax.set_title("Software performance vs. model complexity")
    ax.grid(alpha=0.18, linewidth=0.6)
    return save_figure(fig, "fig03_complexity_vs_fp32_macro_f1")


def fig_search_convergence(curve: pd.DataFrame, metrics: dict) -> list[str]:
    curve = curve.copy()
    curve["best_so_far_macro_f1"] = curve["best_so_far_macro_f1"].ffill()
    x95 = metrics["sample_efficiency_candidates_to_95pct_final_best_macro_f1"]
    best = metrics["best_so_far_macro_f1"]
    target = 0.95 * best
    fig, ax1 = plt.subplots(figsize=(6.8, 3.2))
    ax1.plot(
        curve["evaluation"],
        curve["best_so_far_macro_f1"],
        color=PALETTE["blue"],
        linewidth=1.8,
        label="Best-so-far Macro-F1",
    )
    feasible = curve[curve["feasible"].astype(str).str.lower() == "true"]
    ax1.scatter(
        feasible["evaluation"],
        feasible["macro_f1"],
        s=10,
        color=PALETTE["sky"],
        alpha=0.35,
        linewidth=0,
        label="Feasible candidates",
    )
    ax1.axhline(target, color=PALETTE["gray"], linestyle="--", linewidth=0.9)
    ax1.axvline(x95, color=PALETTE["red"], linestyle="--", linewidth=0.9)
    ax1.set_xlabel("Evaluated candidates")
    ax1.set_ylabel("Macro-F1")
    ax1.set_ylim(0.55, max(0.9, best + 0.02))
    ax2 = ax1.twinx()
    ax2.plot(
        curve["evaluation"],
        curve["best_so_far_reward"],
        color=PALETTE["orange"],
        linewidth=1.4,
        label="Best-so-far reward",
    )
    ax2.set_ylabel("Reward")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="lower right")
    ax1.set_title("RL search convergence and sample efficiency")
    ax1.grid(alpha=0.18, linewidth=0.6)
    return save_figure(fig, "fig04_rl_search_convergence")


def fig_per_class_heatmap(per_class: pd.DataFrame, order: list[str]) -> list[str]:
    pivot = per_class.pivot(index="arch_id", columns="class_name", values="fp32_per_class_f1_mean")
    pivot = pivot.loc[[arch for arch in order if arch in pivot.index]]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="cividis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if pivot.iloc[i, j] < 0.55 else "black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label("Per-class F1")
    ax.set_title("Software FP32 per-class F1")
    return save_figure(fig, "fig05_per_class_f1_heatmap")


def write_reports(df: pd.DataFrame, metrics: dict, figure_paths: dict[str, list[str]]) -> None:
    best = df.iloc[0]
    report = [
        "# ZCU104 MobileNetA Software Experiment Summary",
        "",
        "## How to describe the software-side experiment",
        "",
        "1. Task and data protocol: NKSID closed-set sonar image classification, input size 1x224x224, 8 classes, 5 outer folds x seeds 42/43/44.",
        "2. Search protocol: RL-HW-NAS only, 200 candidate evaluations, reward includes classification and ZCU104 proxy hardware constraints from the formal operator-cost LUT.",
        "3. Candidate freezing: five Pareto candidates are reported with role labels for selection rationale, not as architecture components.",
        "4. Final software retraining: selected candidates are retrained under the formal classification protocol and summarized by Macro-F1, Top-1, Tail-class F1, and per-class F1.",
        "5. Quantization and complexity: report Params, MACs, FP32 metrics, PTQ fake-quant INT8 metrics, and FP32-INT8 drops.",
        "6. Search efficiency and Pareto quality: report sample efficiency, convergence speed, feasible rate, Pareto front size, and hypervolume separately from final classification accuracy.",
        "7. Evidence boundary: current software metrics do not claim HLS/full-route closure, board classification correctness, FPS, or power.",
        "",
        "## Key current results",
        "",
        f"- Best software candidate: `{best['arch_id']}` with FP32 Macro-F1 {best['macro_f1_mean']:.4f} +/- {best['macro_f1_std']:.4f}, Top-1 {best['top1_mean']:.4f} +/- {best['top1_std']:.4f}.",
        f"- Search sample efficiency: {metrics['sample_efficiency_candidates_to_95pct_final_best_macro_f1']} candidates to reach 95% of final best search Macro-F1.",
        f"- Search convergence speed: {metrics['convergence_speed_fraction_of_budget_to_95pct']:.2f} of the 200-candidate budget.",
        f"- Hardware-feasible rate in search proxy: {metrics['hardware_feasible_rate']:.4f}.",
        f"- Pareto front size: {metrics['pareto_front_size']}; normalized hypervolume: {metrics['hypervolume']:.4f}.",
        "",
        "## Generated figures",
        "",
    ]
    for name, paths in figure_paths.items():
        report.append(f"- {name}: `{paths[0]}`, `{paths[1]}`")
    report.extend(
        [
            "",
            "## Caveats",
            "",
            "- INT8 values are PTQ fake-quant software validation only; integer reference parity remains fail-closed / unsupported-or-unverified.",
            "- Source summaries currently have source_freeze_verified=false, so keep formal claim wording guarded until the freeze archive gate is closed.",
            "- Board Macro-F1, board Top-1, SW-FPGA agreement, latency, FPS, and power remain pending until a routed bitstream exists.",
        ]
    )
    (OUT_DIR / "software_experiment_configuration_and_results.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    catalog = [
        "# Figure Catalog",
        "",
        "| Figure | Purpose | Data source | Interpretation |",
        "| --- | --- | --- | --- |",
        "| fig01 | Compare FP32 Macro-F1 and Top-1 across frozen candidates with std bars. | Retrain protocol summary | Shows rl_arch_13 as the software accuracy leader. |",
        "| fig02 | Show PTQ fake-quant score change. | PTQ complexity summary | Indicates small software-side INT8 changes, but not board parity. |",
        "| fig03 | Relate software Macro-F1 to MACs and Params. | PTQ complexity summary | Separates classification quality from model cost. |",
        "| fig04 | Show RL best-so-far convergence and sample efficiency. | RL search curve and metrics | The 95% best-search Macro-F1 threshold is reached at candidate 14. |",
        "| fig05 | Show class-wise FP32 F1 pattern. | Per-class F1 summary | Tail classes remain the main weakness, especially fishing_net. |",
    ]
    (OUT_DIR / "figure_catalog.md").write_text("\n".join(catalog) + "\n", encoding="utf-8")

    summary = {
        "schema_version": 1,
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "source_files": [
            str(RETRAIN_CSV.relative_to(ROOT)),
            str(PTQ_CSV.relative_to(ROOT)),
            str(SEARCH_METRICS_JSON.relative_to(ROOT)),
            str(SEARCH_CURVE_CSV.relative_to(ROOT)),
            str(PER_CLASS_CSV.relative_to(ROOT)),
        ],
        "figures": figure_paths,
        "best_software_arch_id": best["arch_id"],
        "best_fp32_macro_f1_mean": float(best["macro_f1_mean"]),
        "claim_boundary": "Software-side visualization only; no board/runtime/power claim.",
    }
    (OUT_DIR / "visualization_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    df, per_class, curve, metrics, _ = load_tables()
    order = df["arch_id"].tolist()
    figure_paths = {
        "FP32 classification performance": fig_classification_performance(df),
        "INT8 fake-quant drop": fig_int8_drop(df),
        "Complexity vs Macro-F1": fig_complexity_tradeoff(df),
        "RL search convergence": fig_search_convergence(curve, metrics),
        "Per-class F1 heatmap": fig_per_class_heatmap(per_class, order),
    }
    write_reports(df, metrics, figure_paths)
    print(json.dumps({"output_dir": str(OUT_DIR), "figures": figure_paths}, indent=2))


if __name__ == "__main__":
    main()
