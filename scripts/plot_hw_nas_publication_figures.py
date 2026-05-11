#!/usr/bin/env python3
"""Create publication-ready HW-NAS comparison figures.

The default inputs are the formal NKSID/AV7K325 backbone run and the
macro-F1 search-strategy runs currently present in this repository.
Outputs are written under results/publication_figures by default.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


DEFAULT_BACKBONE_CSV = Path(
    "results/backbone_nksid_av7k325_fair_scratch_seed42_gpu_20260417/results/backbone_summary.csv"
)

DEFAULT_SEARCH_RUNS = {
    "Random": Path(
        "results/random_baseline_mobile_anchor_shufflenet_av7k325_200_formal/results/candidates.csv"
    ),
    "RL": Path("results/rl_mobile_anchor_shufflenet_av7k325_200_formal/results/candidates.csv"),
    "ProxylessNAS": Path("results/proxyless_mobile_anchor_shufflenet_av7k325_formal/results/candidates.csv"),
}

OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}

METHOD_COLORS = {
    "Random": OKABE_ITO["blue"],
    "RL": OKABE_ITO["vermillion"],
    "ProxylessNAS": OKABE_ITO["bluish_green"],
}

METHOD_MARKERS = {
    "Random": "o",
    "RL": "s",
    "ProxylessNAS": "^",
}

NUMERIC_COLUMNS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "top1",
    "top5",
    "latency_ms",
    "energy_mj",
    "lut",
    "bram",
    "dsp",
    "power_w",
    "fpga_latency_ms",
    "peak_lut",
    "peak_bram",
    "peak_dsp",
]


@dataclass(frozen=True)
class OutputRecord:
    figure: str
    png: str
    pdf: str


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 9,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def ensure_numeric(df: pd.DataFrame, columns: Iterable[str] = NUMERIC_COLUMNS) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_backbone_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing backbone summary: {path}")
    df = ensure_numeric(pd.read_csv(path))
    required = {
        "display_name",
        "top1",
        "macro_f1",
        "fpga_latency_ms",
        "peak_lut",
        "peak_dsp",
        "peak_bram",
        "power_w",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df = df.dropna(subset=["top1", "macro_f1", "fpga_latency_ms", "peak_lut", "peak_dsp", "peak_bram"])
    return df.sort_values(["macro_f1", "top1", "fpga_latency_ms"], ascending=[False, False, True])


def load_search_csvs(paths: dict[str, Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for method, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing search candidates: {path}")
        df = ensure_numeric(pd.read_csv(path))
        required = {"arch_id", "macro_f1", "top1", "latency_ms", "lut", "dsp", "bram", "power_w"}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        df = df.dropna(subset=["macro_f1", "top1", "latency_ms", "lut", "dsp", "bram", "power_w"]).copy()
        df["method"] = method
        df["source_path"] = relpath(path)
        frames.append(df)
    if not frames:
        raise ValueError("No search candidate tables were loaded")
    return pd.concat(frames, ignore_index=True)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.8)


def compact_backbone_labels(labels: Iterable[str]) -> list[str]:
    aliases = {
        "MobileNetV2": "MobileNet",
        "EfficientNet-B0": "EffNet-B0",
        "ShuffleNetV2": "ShuffleNet",
        "FBNet-A": "FBNet-A",
    }
    return [aliases.get(str(label), str(label)) for label in labels]


def set_category_ticks(ax: plt.Axes, x: np.ndarray, labels: list[str], rotation: float = 25) -> None:
    ax.set_xticks(x, labels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(rotation)
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> OutputRecord:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return OutputRecord(figure=stem, png=relpath(png), pdf=relpath(pdf))


def annotate_bars(ax: plt.Axes, bars, fmt: str, ypad_frac: float = 0.012) -> None:
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * ypad_frac
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + pad,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=6.5,
        )


def plot_backbone_tradeoff(backbone: pd.DataFrame, output_dir: Path) -> OutputRecord:
    labels = compact_backbone_labels(backbone["display_name"])
    x = np.arange(len(backbone))
    colors = [OKABE_ITO["blue"], OKABE_ITO["orange"]]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), constrained_layout=True)
    axes = axes.ravel()

    width = 0.36
    bars_top1 = axes[0].bar(x - width / 2, backbone["top1"] * 100, width, color=colors[0], label="Top-1")
    bars_macro = axes[0].bar(
        x + width / 2, backbone["macro_f1"] * 100, width, color=colors[1], label="Macro-F1"
    )
    axes[0].set_ylabel("Score (%)")
    set_category_ticks(axes[0], x, labels)
    axes[0].set_ylim(0, max(float((backbone["top1"] * 100).max()), float((backbone["macro_f1"] * 100).max())) * 1.14)
    axes[0].legend(frameon=False, loc="lower right")
    annotate_bars(axes[0], bars_top1, "{:.1f}")
    annotate_bars(axes[0], bars_macro, "{:.1f}")
    add_panel_label(axes[0], "A")
    despine(axes[0])

    metric_specs = [
        ("fpga_latency_ms", "FPGA latency (ms)", OKABE_ITO["sky_blue"], "{:.2f}", "B"),
        ("peak_lut", "LUT", OKABE_ITO["bluish_green"], "{:.0f}", "C"),
        ("peak_dsp", "DSP", OKABE_ITO["vermillion"], "{:.0f}", "D"),
        ("peak_bram", "BRAM", OKABE_ITO["reddish_purple"], "{:.0f}", "E"),
        ("power_w", "Power (W)", OKABE_ITO["yellow"], "{:.2f}", "F"),
    ]
    for ax, (column, ylabel, color, fmt, panel) in zip(axes[1:], metric_specs):
        values = backbone[column]
        bars = ax.bar(x, values, color=color, edgecolor="black", linewidth=0.35)
        ax.set_ylabel(ylabel)
        set_category_ticks(ax, x, labels)
        ax.set_ylim(0, float(values.max()) * 1.18)
        annotate_bars(ax, bars, fmt)
        add_panel_label(ax, panel)
        despine(ax)

    return save_figure(fig, output_dir, "figure1_backbone_accuracy_resource_tradeoff")


def pareto_front(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    ordered = df.sort_values([x_col, y_col], ascending=[True, False])
    keep = []
    best_y = -np.inf
    for idx, row in ordered.iterrows():
        y = float(row[y_col])
        if y > best_y:
            keep.append(idx)
            best_y = y
    return ordered.loc[keep]


def plot_search_strategy_tradeoff(search: pd.DataFrame, output_dir: Path) -> OutputRecord:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    ax_macro, ax_top1, ax_curve, ax_dist = axes.ravel()

    for method, group in search.groupby("method", sort=False):
        color = METHOD_COLORS.get(method, OKABE_ITO["black"])
        marker = METHOD_MARKERS.get(method, "o")
        ax_macro.scatter(
            group["latency_ms"],
            group["macro_f1"] * 100,
            s=18,
            alpha=0.58,
            color=color,
            marker=marker,
            linewidths=0,
            label=f"{method} (n={len(group)})",
        )
        best = group.sort_values(["macro_f1", "top1"], ascending=[False, False]).iloc[0]
        ax_macro.scatter(
            [best["latency_ms"]],
            [best["macro_f1"] * 100],
            s=90,
            marker="*",
            color=color,
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
        )

        ax_top1.scatter(
            group["latency_ms"],
            group["top1"] * 100,
            s=18,
            alpha=0.58,
            color=color,
            marker=marker,
            linewidths=0,
            label=method,
        )
        best_top1 = group.sort_values(["top1", "macro_f1"], ascending=[False, False]).iloc[0]
        ax_top1.scatter(
            [best_top1["latency_ms"]],
            [best_top1["top1"] * 100],
            s=90,
            marker="*",
            color=color,
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
        )

        best_so_far = group["macro_f1"].cummax() * 100
        ax_curve.plot(
            np.arange(1, len(best_so_far) + 1),
            best_so_far,
            color=color,
            linewidth=1.7,
            label=method,
        )

    front = pareto_front(search, "latency_ms", "macro_f1")
    ax_macro.plot(
        front["latency_ms"],
        front["macro_f1"] * 100,
        linestyle="--",
        color=OKABE_ITO["black"],
        linewidth=1.2,
        label="Global Pareto",
    )

    ax_macro.set_xlabel("Latency (ms)")
    ax_macro.set_ylabel("Macro-F1 (%)")
    ax_macro.legend(frameon=False, loc="lower right")
    add_panel_label(ax_macro, "A")
    despine(ax_macro)

    ax_top1.set_xlabel("Latency (ms)")
    ax_top1.set_ylabel("Top-1 (%)")
    add_panel_label(ax_top1, "B")
    despine(ax_top1)

    ax_curve.set_xlabel("Evaluated candidates")
    ax_curve.set_ylabel("Best macro-F1 so far (%)")
    ax_curve.legend(frameon=False, loc="lower right")
    add_panel_label(ax_curve, "C")
    despine(ax_curve)

    method_order = [m for m in DEFAULT_SEARCH_RUNS if m in set(search["method"])]
    distribution = [search.loc[search["method"] == method, "macro_f1"].to_numpy() * 100 for method in method_order]
    box = ax_dist.boxplot(distribution, patch_artist=True, widths=0.62, showfliers=False)
    for patch, method in zip(box["boxes"], method_order):
        patch.set_facecolor(METHOD_COLORS.get(method, OKABE_ITO["black"]))
        patch.set_alpha(0.45)
        patch.set_edgecolor("black")
    for item in box["medians"]:
        item.set_color("black")
        item.set_linewidth(1.0)
    rng = np.random.default_rng(0)
    for i, method in enumerate(method_order, start=1):
        values = search.loc[search["method"] == method, "macro_f1"].to_numpy() * 100
        jitter = rng.normal(0, 0.035, size=len(values))
        ax_dist.scatter(
            np.full(len(values), i) + jitter,
            values,
            s=8,
            alpha=0.25,
            color=METHOD_COLORS.get(method, OKABE_ITO["black"]),
            linewidths=0,
        )
    ax_dist.set_xticks(np.arange(1, len(method_order) + 1), method_order)
    ax_dist.set_ylabel("Macro-F1 (%)")
    add_panel_label(ax_dist, "D")
    despine(ax_dist)

    return save_figure(fig, output_dir, "figure2_search_strategy_macro_f1_tradeoff")


def best_by_method(search: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in search.groupby("method", sort=False):
        rows.append(group.sort_values(["macro_f1", "top1"], ascending=[False, False]).iloc[0])
    return pd.DataFrame(rows)


def plot_best_candidate_resources(search: pd.DataFrame, output_dir: Path) -> OutputRecord:
    best = best_by_method(search)
    method_order = [m for m in DEFAULT_SEARCH_RUNS if m in set(best["method"])]
    best = best.set_index("method").loc[method_order].reset_index()
    x = np.arange(len(best))
    colors = [METHOD_COLORS.get(method, OKABE_ITO["black"]) for method in best["method"]]

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.7), constrained_layout=True)
    axes = axes.ravel()
    specs = [
        ("macro_f1", "Macro-F1 (%)", lambda s: s * 100, "{:.1f}", "A"),
        ("top1", "Top-1 (%)", lambda s: s * 100, "{:.1f}", "B"),
        ("latency_ms", "Latency (ms)", lambda s: s, "{:.3f}", "C"),
        ("lut", "LUT", lambda s: s, "{:.0f}", "D"),
        ("dsp", "DSP", lambda s: s, "{:.0f}", "E"),
        ("bram", "BRAM", lambda s: s, "{:.0f}", "F"),
        ("power_w", "Power (W)", lambda s: s, "{:.2f}", "G"),
    ]
    for ax, (column, ylabel, transform, fmt, panel) in zip(axes, specs):
        values = transform(best[column].astype(float))
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.35)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, best["method"], rotation=20, ha="right")
        upper = float(np.nanmax(values)) * 1.18 if len(values) else 1.0
        ax.set_ylim(0, upper)
        annotate_bars(ax, bars, fmt)
        add_panel_label(ax, panel)
        despine(ax)

    axes[-1].axis("off")

    return save_figure(fig, output_dir, "figure3_best_search_candidate_resource_profile")


def write_source_tables(
    output_dir: Path,
    backbone: pd.DataFrame,
    search: pd.DataFrame,
    best_search: pd.DataFrame,
) -> dict[str, str]:
    backbone_path = output_dir / "source_backbone_summary.csv"
    search_path = output_dir / "source_search_candidates.csv"
    best_path = output_dir / "source_best_search_candidates.csv"
    backbone.to_csv(backbone_path, index=False)
    search.to_csv(search_path, index=False)
    best_search.to_csv(best_path, index=False)
    return {
        "backbone_summary": relpath(backbone_path),
        "search_candidates": relpath(search_path),
        "best_search_candidates": relpath(best_path),
    }


def write_readme(
    output_dir: Path,
    figures: list[OutputRecord],
    source_tables: dict[str, str],
    backbone_path: Path,
    search_paths: dict[str, Path],
    search: pd.DataFrame,
) -> None:
    lines = [
        "# HW-NAS Publication Figures",
        "",
        "Generated by `scripts/plot_hw_nas_publication_figures.py`.",
        "",
        "## Source data",
        "",
        f"- Backbone baseline: `{relpath(backbone_path)}`",
    ]
    for method, path in search_paths.items():
        rows = int((search["method"] == method).sum())
        lines.append(f"- {method} search candidates: `{relpath(path)}` ({rows} rows with complete metrics)")
    lines.extend(
        [
            "",
            "## Generated figures",
            "",
        ]
    )
    for record in figures:
        lines.append(f"- `{record.figure}`: `{record.png}`, `{record.pdf}`")
    lines.extend(
        [
            "",
            "## Derived source tables",
            "",
        ]
    )
    for name, path in source_tables.items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "Command:",
            "",
            "```bash",
            "python scripts/plot_hw_nas_publication_figures.py --output results/publication_figures",
            "```",
            "",
            "The jitter seed for panel D in Figure 2 is fixed to 0 and only affects point display.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    output_dir: Path,
    figures: list[OutputRecord],
    source_tables: dict[str, str],
    backbone_path: Path,
    search_paths: dict[str, Path],
    backbone: pd.DataFrame,
    search: pd.DataFrame,
) -> None:
    manifest = {
        "script": "scripts/plot_hw_nas_publication_figures.py",
        "output_dir": relpath(output_dir),
        "inputs": {
            "backbone_csv": relpath(backbone_path),
            "search_candidate_csvs": {method: relpath(path) for method, path in search_paths.items()},
        },
        "row_counts": {
            "backbone_rows": int(len(backbone)),
            "search_rows_by_method": {
                method: int((search["method"] == method).sum()) for method in search_paths
            },
        },
        "metrics": ["macro_f1", "top1", "latency_ms", "lut", "dsp", "bram", "power_w"],
        "figures": [record.__dict__ for record in figures],
        "source_tables": source_tables,
        "display_jitter_seed": 0,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_search_runs(items: list[str] | None) -> dict[str, Path]:
    if not items:
        return DEFAULT_SEARCH_RUNS.copy()
    parsed: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected METHOD=PATH for --search-run, got: {item}")
        method, path = item.split("=", 1)
        method = method.strip()
        if not method:
            raise ValueError(f"Empty method label in --search-run: {item}")
        parsed[method] = Path(path.strip())
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Create publication-ready HW-NAS figures")
    parser.add_argument("--backbone-csv", type=Path, default=DEFAULT_BACKBONE_CSV)
    parser.add_argument(
        "--search-run",
        action="append",
        default=None,
        help="Search candidate CSV as METHOD=PATH. Can be repeated. Defaults to Random/RL/ProxylessNAS formal runs.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/publication_figures"))
    args = parser.parse_args()

    configure_style()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    search_paths = parse_search_runs(args.search_run)
    backbone = load_backbone_csv(args.backbone_csv)
    search = load_search_csvs(search_paths)
    best_search = best_by_method(search)

    figures = [
        plot_backbone_tradeoff(backbone, output_dir),
        plot_search_strategy_tradeoff(search, output_dir),
        plot_best_candidate_resources(search, output_dir),
    ]
    source_tables = write_source_tables(output_dir, backbone, search, best_search)
    write_readme(output_dir, figures, source_tables, args.backbone_csv, search_paths, search)
    write_manifest(output_dir, figures, source_tables, args.backbone_csv, search_paths, backbone, search)

    print(f"[OK] Wrote {len(figures)} figures to {output_dir}")
    for record in figures:
        print(f"[OK] {record.png}")
        print(f"[OK] {record.pdf}")
    print(f"[OK] {relpath(output_dir / 'README.md')}")
    print(f"[OK] {relpath(output_dir / 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
