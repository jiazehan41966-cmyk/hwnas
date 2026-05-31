#!/usr/bin/env python3
"""Plot HW-NAS server training results as publication-ready figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "outputs" / "server_training_figures_raw"
OUT_DIR = REPO_ROOT / "outputs" / "server_training_figures"

OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}


FALLBACK_FINAL_REPORT: dict[str, Any] = {
    "status": "completed",
    "search_run_dir": "results/nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_v1",
    "retrain_run_dir": "results/nas_board_lut_strict_current84_arch84_full_best_retrain150_cuda_v1",
    "best_arch_id": "rl_arch_224",
    "search_stage_metrics": {
        "macro_f1": 0.6519424663813247,
        "top1": 0.8076923076923077,
        "top5": 0.9980769230769231,
        "latency_ms": 7.2867750000000004,
        "lut": 128445,
        "dsp": 1172,
        "bram": 188,
        "power_w": 15.7426,
    },
    "final_retrain_validation_metrics": {
        "macro_f1": 0.8456907326085281,
        "top1": 0.9134615384615384,
        "top5": 1.0,
        "loss": 0.26361760979052634,
        "num_samples": 520.0,
    },
    "retrain_checkpoint_metrics": {
        "best_epoch": 133,
        "epochs_completed": 150,
        "best_val_acc": 0.9134615384615384,
        "final_val_acc": 0.9134615384615384,
    },
    "estimated_hardware_cost": {
        "latency_ms": 7.2867750000000004,
        "lut": 128445,
        "dsp": 1172,
        "bram": 188,
        "power_w": 15.7426,
    },
    "strict_lut_full_hit": True,
    "lut_stats": {
        "hits": 2100,
        "misses": 0,
        "true_misses": 0,
        "deferred_hits": 0,
        "strict_formal_lut": True,
        "hit_rate": 1.0,
    },
    "search_summary": {
        "total_evaluated": 300,
        "feasible": 300,
        "infeasible": 0,
        "selection_metric": "macro_f1",
    },
}


FALLBACK_SEARCH_SAMPLES = [
    (0, 0.5849),
    (10, 0.5436),
    (20, 0.5312),
    (30, 0.5636),
    (40, 0.5727),
    (50, 0.5566),
    (60, 0.5410),
    (70, 0.5708),
    (80, 0.5550),
    (90, 0.6101),
    (100, 0.5537),
    (110, 0.6332),
    (120, 0.5738),
    (130, 0.6130),
    (140, 0.5742),
    (150, 0.5935),
    (160, 0.5691),
    (170, 0.5478),
    (180, 0.5069),
    (190, 0.5788),
    (200, 0.5912),
    (210, 0.4889),
    (220, 0.5546),
    (230, 0.6047),
    (240, 0.6025),
    (250, 0.5612),
    (260, 0.6044),
    (270, 0.4982),
    (280, 0.5711),
    (290, 0.5590),
]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_final_report() -> dict[str, Any]:
    path = RAW_DIR / "final_report.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fallback_path = RAW_DIR / "final_report_from_session.json"
    fallback_path.write_text(
        json.dumps(FALLBACK_FINAL_REPORT, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return FALLBACK_FINAL_REPORT


def load_search_curve(report: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    candidates_path = RAW_DIR / "candidates.jsonl"
    if candidates_path.exists():
        episodes: list[int] = []
        values: list[float] = []
        for index, line in enumerate(candidates_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            record = json.loads(line)
            metrics = record["candidate"]["metrics"]
            value = metrics.get("macro_f1")
            if value is None:
                value = metrics.get("accuracy")
            if value is None:
                continue
            episodes.append(index)
            values.append(float(value))
        if episodes:
            return np.array(episodes), np.array(values), "all 300 evaluated candidates"

    sample_path = RAW_DIR / "search_log_samples_from_session.csv"
    with sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "macro_f1_sample"])
        writer.writerows(FALLBACK_SEARCH_SAMPLES)
        writer.writerow(
            [
                224,
                report["search_stage_metrics"]["macro_f1"],
            ]
        )
    data = sorted(FALLBACK_SEARCH_SAMPLES + [(224, report["search_stage_metrics"]["macro_f1"])])
    return (
        np.array([item[0] for item in data], dtype=float),
        np.array([item[1] for item in data], dtype=float),
        "log samples plus exact best candidate",
    )


def annotate_bar_values(ax: plt.Axes, bars, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=7,
        )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def make_overview_figure(report: dict[str, Any]) -> Path:
    episodes, metric_values, curve_source = load_search_curve(report)
    cumulative_best = np.maximum.accumulate(metric_values)
    search = report["search_stage_metrics"]
    final = report["final_retrain_validation_metrics"]
    search_summary = report["search_summary"]
    lut_stats = report["lut_stats"]
    cost = report["estimated_hardware_cost"]

    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.36)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(
        episodes,
        metric_values,
        color=OKABE_ITO["sky"],
        marker="o",
        markersize=2.8,
        linewidth=1.0,
        label="candidate metric",
    )
    ax.plot(
        episodes,
        cumulative_best,
        color=OKABE_ITO["blue"],
        linewidth=1.5,
        label="cumulative best",
    )
    best_episode = 224
    best_macro = search["macro_f1"]
    ax.scatter(
        [best_episode],
        [best_macro],
        s=42,
        color=OKABE_ITO["vermillion"],
        zorder=5,
        label="selected best",
    )
    ax.set_xlabel("RL episode")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Search trajectory")
    ax.set_xlim(-5, 305)
    ax.set_ylim(0.45, 0.70)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.02,
        0.04,
        curve_source,
        transform=ax.transAxes,
        fontsize=6.5,
        color="0.35",
    )
    add_panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    metrics = ["macro-F1", "top-1", "top-5"]
    search_vals = [search["macro_f1"], search["top1"], search["top5"]]
    final_vals = [final["macro_f1"], final["top1"], final["top5"]]
    x = np.arange(len(metrics))
    width = 0.36
    bars1 = ax.bar(
        x - width / 2,
        search_vals,
        width,
        color=OKABE_ITO["yellow"],
        edgecolor="black",
        linewidth=0.4,
        label="search eval",
    )
    bars2 = ax.bar(
        x + width / 2,
        final_vals,
        width,
        color=OKABE_ITO["green"],
        edgecolor="black",
        linewidth=0.4,
        label="150-epoch retrain",
    )
    annotate_bar_values(ax, bars1)
    annotate_bar_values(ax, bars2)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Validation score")
    ax.set_title("Sonar classification performance")
    ax.legend(frameon=False, loc="lower right")
    add_panel_label(ax, "B")

    ax = fig.add_subplot(gs[1, 0])
    labels = ["evaluated", "feasible", "LUT hit rate", "true miss", "deferred hit"]
    values = [
        search_summary["total_evaluated"],
        search_summary["feasible"],
        lut_stats["hit_rate"] * 100,
        lut_stats["true_misses"],
        lut_stats["deferred_hits"],
    ]
    colors = [
        OKABE_ITO["blue"],
        OKABE_ITO["green"],
        OKABE_ITO["sky"],
        OKABE_ITO["vermillion"],
        OKABE_ITO["purple"],
    ]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Count or percent")
    ax.set_title("Search efficiency and LUT coverage")
    for yi, value in enumerate(values):
        suffix = "%" if labels[yi] == "LUT hit rate" else ""
        ax.text(value + max(values) * 0.02, yi, f"{value:g}{suffix}", va="center", fontsize=7)
    ax.set_xlim(0, max(values) * 1.22)
    add_panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 1])
    hw_labels = ["Latency\n(ms)", "Power\n(W)", "LUT\n(k)", "DSP", "BRAM"]
    hw_values = [
        cost["latency_ms"],
        cost["power_w"],
        cost["lut"] / 1000.0,
        cost["dsp"],
        cost["bram"],
    ]
    bar_colors = [
        OKABE_ITO["blue"],
        OKABE_ITO["orange"],
        OKABE_ITO["green"],
        OKABE_ITO["purple"],
        OKABE_ITO["sky"],
    ]
    bars = ax.bar(
        np.arange(len(hw_labels)),
        hw_values,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xticks(np.arange(len(hw_labels)), hw_labels)
    ax.set_yscale("log")
    ax.set_ylabel("Value (log scale)")
    ax.set_title("Strict-LUT estimated hardware cost")
    for bar, value in zip(bars, hw_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.16,
            f"{value:.2f}" if value < 100 else f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.text(
        0.0,
        -0.28,
        "Latency is a strict LUT aggregate estimate, not COM5 end-to-end latency.",
        transform=ax.transAxes,
        fontsize=6.5,
        color="0.35",
    )
    add_panel_label(ax, "D")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_base = OUT_DIR / "hwnas_server_training_overview"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_base.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_base.with_suffix(".png")


def make_architecture_figure(report: dict[str, Any]) -> Path:
    structure = report.get("best_network_structure") or {
        "stages": [
            {"channels": 16, "stride": 1, "blocks": [{"op": "conv", "kernel_size": 1, "expand_ratio": 1}]},
            {"channels": 24, "stride": 2, "blocks": [{"op": "mbconv", "kernel_size": 3, "expand_ratio": 6}]},
            {"channels": 32, "stride": 2, "blocks": [{"op": "mbconv", "kernel_size": 3, "expand_ratio": 3}]},
            {"channels": 32, "stride": 1, "blocks": [{"op": "mbconv", "kernel_size": 3, "expand_ratio": 3}]},
        ]
    }
    stages = structure["stages"]
    stage_names = [f"S{i}" for i in range(len(stages))]
    channels = [stage["channels"] for stage in stages]
    strides = [stage["stride"] for stage in stages]
    op_labels = [
        f"{stage['blocks'][0]['op']}\nk{stage['blocks'][0]['kernel_size']} e{stage['blocks'][0]['expand_ratio']}"
        for stage in stages
    ]

    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    x = np.arange(len(stages))
    bars = ax.bar(
        x,
        channels,
        color=[OKABE_ITO["sky"], OKABE_ITO["green"], OKABE_ITO["blue"], OKABE_ITO["purple"]],
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, op_label, stride in zip(bars, op_labels, strides):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{op_label}\nstride {stride}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_xticks(x, stage_names)
    ax.set_ylabel("Output channels")
    ax.set_title(f"Selected architecture: {report['best_arch_id']}")
    ax.set_ylim(0, max(channels) + 18)
    ax.text(
        0.0,
        -0.28,
        "Input: 1 x 224 x 224; stem: Conv 3x3, 32 channels, stride 2; head: classifier for 8 sonar classes.",
        transform=ax.transAxes,
        fontsize=7,
        color="0.30",
    )
    output_base = OUT_DIR / "hwnas_best_architecture"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_base.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_base.with_suffix(".png")


def write_summary_tables(report: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT_DIR / "hwnas_key_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "metric", "value"])
        for key, value in report["search_stage_metrics"].items():
            writer.writerow(["search_stage", key, value])
        for key, value in report["final_retrain_validation_metrics"].items():
            writer.writerow(["final_retrain_validation", key, value])
        for key, value in report["lut_stats"].items():
            writer.writerow(["strict_lut", key, value])
        for key, value in report["search_summary"].items():
            writer.writerow(["search_summary", key, value])


def main() -> None:
    configure_style()
    report = load_final_report()
    write_summary_tables(report)
    overview = make_overview_figure(report)
    architecture = make_architecture_figure(report)
    print(f"Saved overview: {overview}")
    print(f"Saved architecture: {architecture}")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
