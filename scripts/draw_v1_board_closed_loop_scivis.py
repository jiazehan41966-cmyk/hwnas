from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARETO = ROOT / "results" / "fnas_style_pareto_outputs_20260816"
OUT = ROOT / "results" / "v1_polished_figures_20260816"

OKABE = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#111111",
    "gray": "#7A7A7A",
    "grid": "#E6E6E6",
}

ROLE_COLORS = {
    "精度优先": OKABE["blue"],
    "延迟优先": OKABE["orange"],
    "资源优先": OKABE["green"],
    "额外候选": OKABE["purple"],
}

ROLE_LABELS = {
    "精度优先": "Acc.-first",
    "延迟优先": "Lat.-first",
    "资源优先": "Res.-first",
    "额外候选": "Extra",
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, Any], keys: list[str]) -> None:
    for key in keys:
        row[key] = float(row[key])


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 7.5,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.4,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.1,
            "figure.dpi": 180,
            "savefig.dpi": 360,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "grid.color": OKABE["grid"],
            "grid.linewidth": 0.7,
        }
    )


def load_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    search = read_csv(PARETO / "table08_search_pareto_ranked_candidates.csv")
    board = read_csv(PARETO / "table09_board_pareto_candidates.csv")
    for row in search:
        as_float(row, ["candidate_index", "macro_f1", "top1", "latency_ms", "lut", "dsp", "bram"])
        row["pareto_front"] = str(row["pareto_front"]).lower() == "true"
    for row in board:
        as_float(row, ["macro_f1", "top1", "latency_ms", "nas_latency_ms", "lut", "dsp", "bram"])
    return search, board


def label(arch_id: str) -> str:
    return arch_id.replace("rl_arch_", "#")


def pareto(rows: list[dict[str, Any]], x_key: str, y_key: str) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    best_y = -1.0
    for row in sorted(rows, key=lambda r: (r[x_key], -r[y_key])):
        if row[y_key] > best_y:
            front.append(row)
            best_y = row[y_key]
    return front


def panel(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.14,
        1.07,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="black",
    )


def clean(ax: plt.Axes) -> None:
    ax.grid(axis="y")
    ax.set_axisbelow(True)


def save_data(board: list[dict[str, Any]]) -> None:
    with (OUT / "fig06_v1_board_closed_loop_scivis_data.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "arch_id",
                "role",
                "search_estimated_latency_ms",
                "board_measured_latency_ms",
                "macro_f1_percent",
                "top1_percent",
                "lut",
                "dsp",
                "bram",
            ],
        )
        writer.writeheader()
        for row in board:
            writer.writerow(
                {
                    "arch_id": row["arch_id"],
                    "role": row["role"],
                    "search_estimated_latency_ms": f"{row['nas_latency_ms']:.5f}",
                    "board_measured_latency_ms": f"{row['latency_ms']:.5f}",
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
    selected_search = [r for r in search if r["arch_id"] in board_ids]
    other_search = [r for r in search if r["arch_id"] not in board_ids]
    search_front = sorted([r for r in search if r["pareto_front"]], key=lambda r: r["latency_ms"])
    board_front = pareto(board, "latency_ms", "top1")
    board_lookup = {r["arch_id"]: r for r in board}

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.05))
    ax_a, ax_b, ax_c, ax_d = axs.ravel()

    ax_a.scatter(
        [r["latency_ms"] for r in other_search],
        [r["macro_f1"] * 100 for r in other_search],
        s=17,
        color=OKABE["sky"],
        alpha=0.5,
        edgecolors="none",
        label=f"Search candidates (n={len(search)})",
    )
    if search_front:
        ax_a.plot(
            [r["latency_ms"] for r in search_front],
            [r["macro_f1"] * 100 for r in search_front],
            color=OKABE["black"],
            linestyle=(0, (4, 2)),
            linewidth=1.15,
            label="Pareto front",
            zorder=3,
        )
    for row in selected_search:
        role = board_lookup[row["arch_id"]]["role"]
        ax_a.scatter(
            row["latency_ms"],
            row["macro_f1"] * 100,
            s=115,
            marker="*",
            color=ROLE_COLORS.get(role, OKABE["orange"]),
            edgecolor="black",
            linewidth=0.75,
            zorder=4,
        )
    ax_a.set_title("Search-stage candidate quality")
    ax_a.set_xlabel("Estimated latency (ms)")
    ax_a.set_ylabel("Macro-F1 (%)")
    ax_a.set_xlim(5.15, 7.42)
    ax_a.set_ylim(42, 64)
    ax_a.legend(frameon=False, loc="lower right", handlelength=1.6)
    clean(ax_a)
    panel(ax_a, "A")

    for row in board:
        ax_b.scatter(
            row["latency_ms"],
            row["top1"] * 100,
            s=120,
            marker="*",
            color=ROLE_COLORS.get(row["role"], OKABE["orange"]),
            edgecolor="black",
            linewidth=0.75,
            label=ROLE_LABELS.get(row["role"], row["role"]),
            zorder=4,
        )
    board_front = sorted(board_front, key=lambda r: r["latency_ms"])
    ax_b.plot(
        [r["latency_ms"] for r in board_front],
        [r["top1"] * 100 for r in board_front],
        color=OKABE["black"],
        linestyle=(0, (4, 2)),
        linewidth=1.15,
        label="Board Pareto",
        zorder=2,
    )
    best = max(board, key=lambda r: r["top1"])
    ax_b.annotate(
        f"Ours\n{best['top1'] * 100:.2f}%, {best['latency_ms']:.2f} ms",
        xy=(best["latency_ms"], best["top1"] * 100),
        xytext=(44.2, 79.85),
        fontsize=7.3,
        color=OKABE["blue"],
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", lw=0.8, color=OKABE["blue"]),
    )
    ax_b.set_title("Board-measured trade-off")
    ax_b.set_xlabel("Measured latency (ms)")
    ax_b.set_ylabel("Top-1 (%)")
    ax_b.set_xlim(23.0, 51.5)
    ax_b.set_ylim(76.0, 80.1)
    handles, labels = ax_b.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax_b.legend(unique.values(), unique.keys(), frameon=False, loc="lower right", handlelength=1.5)
    clean(ax_b)
    panel(ax_b, "B")

    order = sorted(board, key=lambda r: (r["latency_ms"], -r["macro_f1"]))
    xs = np.arange(1, len(order) + 1)
    ys = np.array([r["macro_f1"] * 100 for r in order])
    best_so_far = np.maximum.accumulate(ys)
    ax_c.plot(xs, best_so_far, color=OKABE["blue"], drawstyle="steps-post", label="Best so far")
    ax_c.scatter(
        xs,
        ys,
        s=32,
        color=[ROLE_COLORS.get(r["role"], OKABE["gray"]) for r in order],
        edgecolors="white",
        linewidth=0.4,
        zorder=3,
    )
    ax_c.set_xticks(xs, [label(r["arch_id"]) for r in order])
    ax_c.set_title("Closed-loop validation")
    ax_c.set_xlabel("Board-tested candidate")
    ax_c.set_ylabel("Best Macro-F1 (%)")
    ax_c.set_xlim(0.75, len(order) + 0.25)
    ax_c.set_ylim(60.0, 63.3)
    ax_c.legend(frameon=False, loc="lower right")
    clean(ax_c)
    panel(ax_c, "C")

    groups = [
        ("Search\nall", [r["macro_f1"] * 100 for r in search], OKABE["sky"]),
        ("Selected\nestimate", [r["macro_f1"] * 100 for r in selected_search], OKABE["orange"]),
        ("Board\nmeasured", [r["macro_f1"] * 100 for r in board], OKABE["green"]),
    ]
    positions = np.arange(1, 4)
    bp = ax_d.boxplot(
        [g[1] for g in groups],
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=0.9),
        capprops=dict(color="black", linewidth=0.9),
    )
    for patch, (_, _, color) in zip(bp["boxes"], groups):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(OKABE["gray"])
        patch.set_linewidth(0.85)
    rng = np.random.default_rng(20260816)
    for pos, (_, vals, color) in zip(positions, groups):
        size = 12 if len(vals) > 20 else 24
        jitter = rng.normal(0, 0.045, len(vals))
        ax_d.scatter(np.full(len(vals), pos) + jitter, vals, s=size, color=color, alpha=0.55, edgecolors="none")
    ax_d.set_xticks(positions, [g[0] for g in groups])
    ax_d.set_title("Candidate quality distribution")
    ax_d.set_ylabel("Macro-F1 (%)")
    ax_d.set_ylim(42, 64)
    clean(ax_d)
    panel(ax_d, "D")

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.12, top=0.94, wspace=0.34, hspace=0.46)
    fig.text(
        0.085,
        0.035,
        "A/D use search-stage estimates; B/C and the board-measured group in D use AV7K325 COM5 closed-loop results.",
        fontsize=6.7,
        color=OKABE["gray"],
    )

    out = OUT / "fig06_v1_board_closed_loop_scivis"
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    save_data(board)


if __name__ == "__main__":
    draw()
