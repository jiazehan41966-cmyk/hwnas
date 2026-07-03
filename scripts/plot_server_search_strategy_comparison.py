#!/usr/bin/env python3
"""Plot server-side HW-NAS strategy comparison figures.

Input:
  outputs/server_training_figures_raw/search_strategy_comparison_data.json

Outputs:
  outputs/server_training_figures/hwnas_strategy_comparison_overview.{png,pdf,svg}
  outputs/server_training_figures/hwnas_search_process_comparison.{png,pdf,svg}
  outputs/server_training_figures/hwnas_strategy_comparison_metrics.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = (
    REPO_ROOT
    / "outputs"
    / "server_training_figures_raw"
    / "search_strategy_comparison_data.json"
)
OUT_DIR = REPO_ROOT / "outputs" / "server_training_figures"

STRATEGIES = ("RL-NAS", "Random", "ProxylessNAS")
LABELS = {
    "RL-NAS": "RL-NAS",
    "Random": "Random",
    "ProxylessNAS": "Official\nProxylessNAS",
}
COLORS = {
    "RL-NAS": "#0072B2",
    "Random": "#E69F00",
    "ProxylessNAS": "#009E73",
}
MARKERS = {
    "RL-NAS": "o",
    "Random": "s",
    "ProxylessNAS": "^",
}


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
            "ytick.direction": "out",
            "xtick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_bundle() -> dict[str, Any]:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Missing raw comparison data: {RAW_PATH}. "
            "Run the server extraction step first."
        )
    return json.loads(RAW_PATH.read_text(encoding="utf-8-sig"))


def metric(bundle: dict[str, Any], strategy: str, key: str) -> float:
    return float(bundle["runs"][strategy]["final_eval"]["metrics"][key])


def search_metric(bundle: dict[str, Any], strategy: str, key: str) -> float:
    return float(bundle["runs"][strategy]["best_candidate"]["metrics"][key])


def cost_metric(bundle: dict[str, Any], strategy: str, key: str) -> float:
    metrics = bundle["runs"][strategy]["best_candidate"]["metrics"]
    if key in metrics and metrics[key] is not None:
        return float(metrics[key])
    return float(bundle["runs"][strategy]["baseline_cost"][key])


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def annotate_bars(
    ax: plt.Axes,
    bars,
    fmt: str = "{:.3f}",
    pad: float = 0.012,
    rotation: float = 0,
) -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + pad,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=6.5,
            rotation=rotation,
        )


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in ("png", "pdf", "svg"):
        path = OUT_DIR / f"{stem}.{ext}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


def write_metrics_csv(bundle: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "hwnas_strategy_comparison_metrics.csv"
    fields = [
        "strategy",
        "search_macro_f1",
        "search_top1",
        "search_top5",
        "final_macro_f1",
        "final_top1",
        "final_top5",
        "epochs_completed",
        "best_epoch",
        "latency_ms",
        "lut",
        "dsp",
        "bram",
        "power_w",
        "strict_lut_hit_rate",
        "true_misses",
        "deferred_hits",
        "search_run_dir",
        "retrain_run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for strategy in STRATEGIES:
            run = bundle["runs"][strategy]
            retrain_metrics = run["retrain_summary"]["metrics"]
            lut_stats = run["lut_stats"]
            writer.writerow(
                {
                    "strategy": strategy,
                    "search_macro_f1": search_metric(bundle, strategy, "macro_f1"),
                    "search_top1": search_metric(bundle, strategy, "top1"),
                    "search_top5": search_metric(bundle, strategy, "top5"),
                    "final_macro_f1": metric(bundle, strategy, "macro_f1"),
                    "final_top1": metric(bundle, strategy, "top1"),
                    "final_top5": metric(bundle, strategy, "top5"),
                    "epochs_completed": retrain_metrics.get("epochs_completed"),
                    "best_epoch": retrain_metrics.get("best_epoch"),
                    "latency_ms": cost_metric(bundle, strategy, "latency_ms"),
                    "lut": cost_metric(bundle, strategy, "lut"),
                    "dsp": cost_metric(bundle, strategy, "dsp"),
                    "bram": cost_metric(bundle, strategy, "bram"),
                    "power_w": cost_metric(bundle, strategy, "power_w"),
                    "strict_lut_hit_rate": lut_stats.get("hit_rate"),
                    "true_misses": lut_stats.get("true_misses"),
                    "deferred_hits": lut_stats.get("deferred_hits"),
                    "search_run_dir": run["search_run_dir"],
                    "retrain_run_dir": run["retrain_run_dir"],
                }
            )
    return path


def plot_overview(bundle: dict[str, Any]) -> list[Path]:
    fig = plt.figure(figsize=(7.6, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.56, wspace=0.35)

    # A: final validation metrics after full retraining.
    ax = fig.add_subplot(gs[0, 0])
    metrics = ("macro_f1", "top1", "top5")
    x = np.arange(len(metrics))
    width = 0.24
    for offset, strategy in zip((-width, 0, width), STRATEGIES):
        vals = [metric(bundle, strategy, key) for key in metrics]
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=COLORS[strategy],
            label=LABELS[strategy].replace("\n", " "),
        )
        annotate_bars(ax, bars, "{:.3f}", pad=0.012, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(["macro-F1", "top-1", "top-5"])
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("Validation score")
    ax.set_title("Final 150-epoch retrain performance")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    add_panel_label(ax, "A")

    # B: search-stage macro-F1 versus final macro-F1.
    ax = fig.add_subplot(gs[0, 1])
    for idx, strategy in enumerate(STRATEGIES):
        x0, x1 = idx - 0.12, idx + 0.12
        s_val = search_metric(bundle, strategy, "macro_f1")
        f_val = metric(bundle, strategy, "macro_f1")
        ax.plot(
            [x0, x1],
            [s_val, f_val],
            color=COLORS[strategy],
            marker=MARKERS[strategy],
            linewidth=1.4,
            markersize=5,
        )
        if s_val < 0.2:
            ax.text(x0, s_val + 0.035, f"{s_val:.3f}", ha="center", va="bottom", fontsize=6.5)
        else:
            ax.text(x0, s_val - 0.045, f"{s_val:.3f}", ha="center", va="top", fontsize=6.5)
        ax.text(x1, f_val + 0.025, f"{f_val:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(STRATEGIES)))
    ax.set_xticklabels([LABELS[s] for s in STRATEGIES])
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("macro-F1")
    ax.set_title("Search candidate to retrained model")
    add_panel_label(ax, "B")

    # C: normalized hardware cost for selected candidates.
    ax = fig.add_subplot(gs[1, 0])
    cost_keys = ("latency_ms", "lut", "dsp", "bram", "power_w")
    cost_labels = ("Latency", "LUT", "DSP", "BRAM", "Power")
    raw_costs = {
        strategy: np.array([cost_metric(bundle, strategy, key) for key in cost_keys], dtype=float)
        for strategy in STRATEGIES
    }
    max_cost = np.max(np.vstack(list(raw_costs.values())), axis=0)
    max_cost[max_cost == 0] = 1.0
    x = np.arange(len(cost_keys))
    for offset, strategy in zip((-width, 0, width), STRATEGIES):
        ax.bar(
            x + offset,
            raw_costs[strategy] / max_cost,
            width=width,
            color=COLORS[strategy],
            label=LABELS[strategy].replace("\n", " "),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(cost_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Normalized cost (max = 1)")
    ax.set_title("Selected candidate hardware footprint")
    add_panel_label(ax, "C")

    # D: strict LUT acceptance and reproducibility checks.
    ax = fig.add_subplot(gs[1, 1])
    hit_rates = [float(bundle["runs"][s]["lut_stats"].get("hit_rate", 0.0)) for s in STRATEGIES]
    bars = ax.bar(
        np.arange(len(STRATEGIES)),
        hit_rates,
        color=[COLORS[s] for s in STRATEGIES],
        width=0.6,
    )
    annotate_bars(ax, bars, "{:.2f}", pad=0.018)
    ax.set_xticks(np.arange(len(STRATEGIES)))
    ax.set_xticklabels([LABELS[s] for s in STRATEGIES])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Strict LUT hit rate")
    ax.set_title("Strict hardware lookup-table validity")
    for idx, strategy in enumerate(STRATEGIES):
        lut = bundle["runs"][strategy]["lut_stats"]
        ax.text(
            idx,
            0.11,
            f"miss={lut.get('true_misses')}\ndef={lut.get('deferred_hits')}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="0.25",
        )
    add_panel_label(ax, "D")

    return save_figure(fig, "hwnas_strategy_comparison_overview")


def _candidate_arrays(bundle: dict[str, Any], strategy: str) -> dict[str, np.ndarray]:
    records = [
        rec
        for rec in bundle["runs"][strategy]["candidates"]
        if rec.get("feasible") and rec.get("macro_f1") is not None
    ]
    if not records:
        return {
            "episode": np.array([], dtype=float),
            "macro_f1": np.array([], dtype=float),
            "latency_ms": np.array([], dtype=float),
            "lut": np.array([], dtype=float),
        }
    return {
        "episode": np.array([rec["index"] for rec in records], dtype=float),
        "macro_f1": np.array([rec["macro_f1"] for rec in records], dtype=float),
        "latency_ms": np.array([rec["latency_ms"] for rec in records], dtype=float),
        "lut": np.array([rec["lut"] for rec in records], dtype=float),
    }


def plot_search_process(bundle: dict[str, Any]) -> list[Path]:
    fig = plt.figure(figsize=(7.2, 5.25))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.35)

    # A: RL and random search trajectories.
    ax = fig.add_subplot(gs[0, 0])
    for strategy in ("RL-NAS", "Random"):
        arrays = _candidate_arrays(bundle, strategy)
        if arrays["episode"].size == 0:
            continue
        cumulative = np.maximum.accumulate(arrays["macro_f1"])
        ax.plot(
            arrays["episode"],
            arrays["macro_f1"],
            color=COLORS[strategy],
            alpha=0.22,
            linewidth=0.8,
        )
        ax.plot(
            arrays["episode"],
            cumulative,
            color=COLORS[strategy],
            linewidth=1.6,
            label=f"{LABELS[strategy]} cumulative best",
        )
        best_idx = int(np.argmax(arrays["macro_f1"]))
        ax.scatter(
            [arrays["episode"][best_idx]],
            [arrays["macro_f1"][best_idx]],
            color=COLORS[strategy],
            edgecolor="white",
            linewidth=0.5,
            s=34,
            zorder=4,
        )
    ax.set_xlim(-5, 305)
    ax.set_ylim(0.45, 0.70)
    ax.set_xlabel("Evaluated candidate")
    ax.set_ylabel("Search validation macro-F1")
    ax.set_title("RL and random search efficiency")
    ax.legend(frameon=False, loc="lower right")
    add_panel_label(ax, "A")

    # B: official ProxylessNAS supernet validation trajectory.
    ax = fig.add_subplot(gs[0, 1])
    hist = bundle["runs"]["ProxylessNAS"]["official_epoch_history"]
    epochs = np.array([item["epoch"] for item in hist], dtype=float)
    top1 = np.array([item["top1"] for item in hist], dtype=float) / 100.0
    warmup = np.array([bool(item["warmup"]) for item in hist], dtype=bool)
    if epochs.size:
        ax.plot(epochs[warmup], top1[warmup], color="#56B4E9", linewidth=1.4, label="warmup")
        ax.plot(epochs[~warmup], top1[~warmup], color=COLORS["ProxylessNAS"], linewidth=1.4, label="search")
        best_idx = int(np.nanargmax(top1))
        ax.scatter(
            [epochs[best_idx]],
            [top1[best_idx]],
            color=COLORS["ProxylessNAS"],
            edgecolor="white",
            linewidth=0.5,
            s=38,
            zorder=4,
        )
        ax.text(
            epochs[best_idx],
            top1[best_idx] + 0.025,
            f"best top-1={top1[best_idx]:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    ax.set_xlim(0, 315)
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("Official ProxylessNAS epoch")
    ax.set_ylabel("Validation top-1")
    ax.set_title("Official ProxylessNAS supernet training")
    ax.legend(frameon=False, loc="lower right")
    add_panel_label(ax, "B")

    # C: candidate macro-F1 distribution.
    ax = fig.add_subplot(gs[1, 0])
    data = [_candidate_arrays(bundle, s)["macro_f1"] for s in ("RL-NAS", "Random")]
    parts = ax.violinplot(data, positions=[0, 1], widths=0.7, showmeans=False, showextrema=False)
    for body, strategy in zip(parts["bodies"], ("RL-NAS", "Random")):
        body.set_facecolor(COLORS[strategy])
        body.set_edgecolor("none")
        body.set_alpha(0.32)
    for idx, strategy in enumerate(("RL-NAS", "Random")):
        vals = data[idx]
        ax.scatter(
            np.full_like(vals, idx, dtype=float),
            vals,
            s=8,
            color=COLORS[strategy],
            alpha=0.22,
            linewidths=0,
        )
        ax.hlines(np.median(vals), idx - 0.25, idx + 0.25, color=COLORS[strategy], linewidth=1.4)
    proxy_search = search_metric(bundle, "ProxylessNAS", "macro_f1")
    ax.scatter([2], [proxy_search], color=COLORS["ProxylessNAS"], marker="^", s=42, zorder=4)
    ax.text(2, proxy_search + 0.035, f"{proxy_search:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([LABELS[s] for s in STRATEGIES])
    ax.set_ylim(0.0, 0.72)
    ax.set_ylabel("Search validation macro-F1")
    ax.set_title("Candidate quality distribution")
    ax.text(
        0.02,
        0.04,
        "ProxylessNAS records one final converted candidate",
        transform=ax.transAxes,
        fontsize=6.3,
        color="0.35",
    )
    add_panel_label(ax, "C")

    # D: latency-accuracy search landscape.
    ax = fig.add_subplot(gs[1, 1])
    for strategy in ("RL-NAS", "Random"):
        arrays = _candidate_arrays(bundle, strategy)
        lut_sizes = arrays["lut"]
        if arrays["episode"].size == 0:
            continue
        size = 12 + 30 * (lut_sizes - lut_sizes.min()) / max(1.0, (lut_sizes.max() - lut_sizes.min()))
        ax.scatter(
            arrays["latency_ms"],
            arrays["macro_f1"],
            s=size,
            color=COLORS[strategy],
            alpha=0.25,
            linewidths=0,
            label=f"{LABELS[strategy]} candidates",
        )
    for strategy in STRATEGIES:
        ax.scatter(
            [cost_metric(bundle, strategy, "latency_ms")],
            [search_metric(bundle, strategy, "macro_f1")],
            marker=MARKERS[strategy],
            s=76,
            color=COLORS[strategy],
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
    legend_handles = [
        Line2D([0], [0], marker=MARKERS[s], color="none", markerfacecolor=COLORS[s],
               markeredgecolor="white", markersize=7, label=LABELS[s].replace("\n", " "))
        for s in STRATEGIES
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right")
    ax.set_xlabel("Estimated latency (ms)")
    ax.set_ylabel("Search validation macro-F1")
    ax.set_title("Accuracy-latency search landscape")
    add_panel_label(ax, "D")

    return save_figure(fig, "hwnas_search_process_comparison")


def main() -> None:
    configure_style()
    bundle = load_bundle()
    csv_path = write_metrics_csv(bundle)
    overview_paths = plot_overview(bundle)
    process_paths = plot_search_process(bundle)
    print("Metrics CSV:", csv_path)
    for path in overview_paths + process_paths:
        print("Figure:", path)


if __name__ == "__main__":
    main()
