from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARETO = ROOT / "results" / "fnas_style_pareto_outputs_20260816"
OUT = ROOT / "results" / "v1_polished_figures_20260816"

COL = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "black": "#111111",
    "gray": "#777777",
    "grid": "#E8E8E8",
}

ROLE_COLORS = {
    "精度优先": COL["blue"],
    "延迟优先": COL["orange"],
    "资源优先": COL["green"],
    "额外候选": COL["purple"],
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def cast(row: dict[str, Any], keys: list[str]) -> None:
    for key in keys:
        row[key] = float(row[key])


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei", "SimHei"],
            "axes.unicode_minus": False,
            "font.size": 7.2,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "figure.dpi": 180,
            "savefig.dpi": 360,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "grid.color": COL["grid"],
            "grid.linewidth": 0.7,
        }
    )


def load_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    search = read_csv(PARETO / "table08_search_pareto_ranked_candidates.csv")
    board = read_csv(PARETO / "table09_board_pareto_candidates.csv")
    for row in search:
        cast(row, ["candidate_index", "macro_f1", "top1", "latency_ms", "lut", "dsp", "bram"])
        row["pareto_front"] = str(row["pareto_front"]).lower() == "true"
        row["selected"] = str(row["selected"]).lower() == "true"
    for row in board:
        cast(row, ["macro_f1", "top1", "latency_ms", "nas_latency_ms", "lut", "dsp", "bram"])
    return search, board


def panel(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.12,
        1.055,
        letter,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="black",
    )


def clean(ax: plt.Axes) -> None:
    ax.grid(axis="y")
    ax.set_axisbelow(True)


def short_id(arch_id: str) -> str:
    return arch_id.replace("rl_arch_", "#")


def save_data(rows: list[dict[str, Any]], board: list[dict[str, Any]]) -> None:
    out = OUT / "fig07_v1_closed_loop_search_part_scivis_data.csv"
    board_ids = {r["arch_id"] for r in board}
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate_index",
                "arch_id",
                "selected_for_board",
                "search_pareto_front",
                "estimated_latency_ms",
                "macro_f1_percent",
                "top1_percent",
                "lut",
                "dsp",
                "bram",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_index": int(row["candidate_index"]),
                    "arch_id": row["arch_id"],
                    "selected_for_board": row["arch_id"] in board_ids,
                    "search_pareto_front": row["pareto_front"],
                    "estimated_latency_ms": f"{row['latency_ms']:.5f}",
                    "macro_f1_percent": f"{row['macro_f1'] * 100:.2f}",
                    "top1_percent": f"{row['top1'] * 100:.2f}",
                    "lut": f"{row['lut']:.0f}",
                    "dsp": f"{row['dsp']:.0f}",
                    "bram": f"{row['bram']:.1f}",
                }
            )


def draw() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    search, board = load_data()

    board_ids = {r["arch_id"] for r in board}
    board_role = {r["arch_id"]: r["role"] for r in board}
    selected = [r for r in search if r["arch_id"] in board_ids]
    others = [r for r in search if r["arch_id"] not in board_ids]
    front = sorted([r for r in search if r["pareto_front"]], key=lambda r: r["latency_ms"])

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.05))
    ax_a, ax_b, ax_c, ax_d = axs.ravel()

    ax_a.scatter(
        [r["latency_ms"] for r in others],
        [r["macro_f1"] * 100 for r in others],
        s=16,
        color=COL["sky"],
        alpha=0.55,
        edgecolors="none",
        label=f"Candidates (n={len(search)})",
    )
    ax_a.plot(
        [r["latency_ms"] for r in front],
        [r["macro_f1"] * 100 for r in front],
        color=COL["black"],
        linestyle=(0, (4, 2)),
        linewidth=1.05,
        label="Pareto front",
    )
    for row in selected:
        ax_a.scatter(
            row["latency_ms"],
            row["macro_f1"] * 100,
            s=105,
            marker="*",
            color=ROLE_COLORS.get(board_role[row["arch_id"]], COL["orange"]),
            edgecolor="black",
            linewidth=0.75,
            zorder=5,
        )
    ax_a.set_title("Search macro-F1 landscape")
    ax_a.set_xlabel("Estimated latency (ms)")
    ax_a.set_ylabel("Macro-F1 (%)")
    ax_a.set_xlim(5.15, 7.42)
    ax_a.set_ylim(42, 64)
    ax_a.legend(frameon=False, loc="lower right", handlelength=1.5)
    clean(ax_a)
    panel(ax_a, "A")

    ax_b.scatter(
        [r["latency_ms"] for r in others],
        [r["top1"] * 100 for r in others],
        s=16,
        color=COL["sky"],
        alpha=0.55,
        edgecolors="none",
        label=f"Candidates (n={len(search)})",
    )
    ax_b.plot(
        [r["latency_ms"] for r in front],
        [r["top1"] * 100 for r in front],
        color=COL["black"],
        linestyle=(0, (4, 2)),
        linewidth=1.05,
        label="Pareto front",
    )
    for row in selected:
        ax_b.scatter(
            row["latency_ms"],
            row["top1"] * 100,
            s=105,
            marker="*",
            color=ROLE_COLORS.get(board_role[row["arch_id"]], COL["orange"]),
            edgecolor="black",
            linewidth=0.75,
            zorder=5,
        )
    ax_b.set_title("Search top-1 landscape")
    ax_b.set_xlabel("Estimated latency (ms)")
    ax_b.set_ylabel("Top-1 (%)")
    ax_b.set_xlim(5.15, 7.42)
    ax_b.set_ylim(66, 81)
    ax_b.legend(frameon=False, loc="lower right", handlelength=1.5)
    clean(ax_b)
    panel(ax_b, "B")

    by_order = sorted(search, key=lambda r: r["candidate_index"])
    x = np.array([r["candidate_index"] for r in by_order])
    y = np.array([r["macro_f1"] * 100 for r in by_order])
    best = np.maximum.accumulate(y)
    ax_c.plot(x, best, color=COL["blue"], linewidth=1.6, drawstyle="steps-post", label="Best so far")
    for row in selected:
        ax_c.scatter(
            row["candidate_index"],
            row["macro_f1"] * 100,
            s=38,
            color=ROLE_COLORS.get(board_role[row["arch_id"]], COL["orange"]),
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
    ax_c.set_title("Closed-loop search progress")
    ax_c.set_xlabel("Evaluated candidates")
    ax_c.set_ylabel("Best Macro-F1 (%)")
    ax_c.set_xlim(0, max(x) + 5)
    ax_c.set_ylim(50, 64)
    ax_c.legend(frameon=False, loc="lower right")
    clean(ax_c)
    panel(ax_c, "C")

    groups: dict[float, list[float]] = defaultdict(list)
    for row in search:
        groups[round(row["latency_ms"], 2)].append(row["macro_f1"] * 100)
    latencies = sorted(groups)
    data = [groups[v] for v in latencies]
    positions = np.arange(len(latencies))
    bp = ax_d.boxplot(
        data,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=0.9),
        capprops=dict(color="black", linewidth=0.9),
    )
    for patch in bp["boxes"]:
        patch.set(facecolor=COL["sky"], alpha=0.34, edgecolor=COL["gray"], linewidth=0.85)
    rng = np.random.default_rng(20260816)
    for i, vals in enumerate(data):
        jitter = rng.normal(0, 0.045, len(vals))
        ax_d.scatter(np.full(len(vals), positions[i]) + jitter, vals, s=11, color=COL["sky"], alpha=0.45, edgecolors="none")
    lat_to_pos = {v: i for i, v in enumerate(latencies)}
    for row in selected:
        pos = lat_to_pos[round(row["latency_ms"], 2)]
        ax_d.scatter(
            pos,
            row["macro_f1"] * 100,
            s=95,
            marker="*",
            color=ROLE_COLORS.get(board_role[row["arch_id"]], COL["orange"]),
            edgecolor="black",
            linewidth=0.75,
            zorder=5,
        )
    ax_d.set_xticks(positions, [f"{v:.2f}" for v in latencies])
    ax_d.set_title("Quality by latency group")
    ax_d.set_xlabel("Estimated latency group (ms)")
    ax_d.set_ylabel("Macro-F1 (%)")
    ax_d.set_ylim(42, 64)
    clean(ax_d)
    panel(ax_d, "D")

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.12, top=0.94, wspace=0.34, hspace=0.46)
    fig.text(
        0.085,
        0.035,
        "Search part of the board-closed-loop batch: all panels use candidate-generation and hardware-estimation results before COM5 measurement.",
        fontsize=6.5,
        color=COL["gray"],
    )
    out = OUT / "fig07_v1_closed_loop_search_part_scivis"
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    save_data(search, board)


if __name__ == "__main__":
    draw()
