from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARETO = ROOT / "results" / "fnas_style_pareto_outputs_20260816"
OUT = ROOT / "results" / "v1_polished_figures_20260816"

COL = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "gray": "#6B7280",
    "light_grid": "#E5E7EB",
    "black": "#111827",
}

ROLE_COLOR = {
    "精度优先": COL["blue"],
    "延迟优先": "#E69F00",
    "资源优先": COL["green"],
    "额外候选": COL["purple"],
}

TOTAL = {
    "lut": 230_400.0,
    "dsp": 1_728.0,
    "bram": 312.0,
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def f(value: Any) -> float:
    return float(value)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "font.family": "sans-serif",
            "axes.unicode_minus": False,
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "figure.dpi": 160,
            "savefig.dpi": 360,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.9,
            "grid.color": COL["light_grid"],
            "grid.linewidth": 0.9,
            "grid.alpha": 1.0,
            "savefig.bbox": "tight",
        }
    )


def load_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    search = read_csv(PARETO / "table08_search_pareto_ranked_candidates.csv")
    board = read_csv(PARETO / "table09_board_pareto_candidates.csv")
    for row in search:
        for key in ["macro_f1", "top1", "latency_ms", "lut", "dsp", "bram", "candidate_index"]:
            row[key] = f(row[key])
        row["pareto_front"] = str(row["pareto_front"]).lower() == "true"
        row["selected"] = str(row["selected"]).lower() == "true"
    for row in board:
        for key in ["macro_f1", "top1", "latency_ms", "nas_latency_ms", "lut", "dsp", "bram"]:
            row[key] = f(row[key])
    return search, board


def name(arch_id: str) -> str:
    return arch_id.replace("rl_arch_", "候选")


def pareto_front(rows: list[dict[str, Any]], x_key: str, y_key: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: (r[x_key], -r[y_key]))
    front: list[dict[str, Any]] = []
    best_y = -np.inf
    for row in ordered:
        if row[y_key] > best_y:
            front.append(row)
            best_y = row[y_key]
    return front


def panel(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.14,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="black",
    )


def draw() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    search, board = load_rows()
    board_ids = {r["arch_id"] for r in board}
    selected_search = [r for r in search if r["arch_id"] in board_ids]
    other_search = [r for r in search if r["arch_id"] not in board_ids]
    search_front = sorted([r for r in search if r["pareto_front"]], key=lambda r: r["latency_ms"])
    board_front = pareto_front(board, "latency_ms", "top1")
    board_by_id = {r["arch_id"]: r for r in board}

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    ax_a.scatter(
        [r["latency_ms"] for r in other_search],
        [r["macro_f1"] * 100 for r in other_search],
        s=18,
        color=COL["sky"],
        alpha=0.52,
        edgecolors="none",
        label=f"搜索候选 (n={len(search)})",
    )
    if search_front:
        ax_a.plot(
            [r["latency_ms"] for r in search_front],
            [r["macro_f1"] * 100 for r in search_front],
            color=COL["black"],
            linestyle=(0, (4, 2)),
            linewidth=1.4,
            label="筛选前沿",
            zorder=3,
        )
    for row in selected_search:
        b = board_by_id[row["arch_id"]]
        ax_a.scatter(
            row["latency_ms"],
            row["macro_f1"] * 100,
            s=170,
            marker="*",
            color=ROLE_COLOR.get(b["role"], COL["orange"]),
            edgecolor="black",
            linewidth=1.0,
            zorder=5,
        )
    ax_a.set_xlabel("搜索估计延迟 (ms)")
    ax_a.set_ylabel("Macro-F1 (%)")
    ax_a.set_title("搜索候选质量")
    ax_a.set_xlim(5.15, 7.42)
    ax_a.set_ylim(43, 64)
    ax_a.grid(axis="y")
    ax_a.legend(frameon=False, loc="lower right", handlelength=1.8)
    panel(ax_a, "A")

    for row in board:
        size = 70 + 240 * (row["lut"] / TOTAL["lut"])
        ax_b.scatter(
            row["latency_ms"],
            row["top1"] * 100,
            s=size,
            marker="*",
            color=ROLE_COLOR.get(row["role"], COL["orange"]),
            edgecolor="black",
            linewidth=1.0,
            zorder=5,
            label=row["role"],
        )
    board_front = sorted(board_front, key=lambda r: r["latency_ms"])
    ax_b.plot(
        [r["latency_ms"] for r in board_front],
        [r["top1"] * 100 for r in board_front],
        color=COL["black"],
        linestyle=(0, (4, 2)),
        linewidth=1.3,
        label="板端前沿",
        zorder=2,
    )
    ax_b.annotate(
        "本工作 (AV7K325)\n79.23%, 49.06 ms",
        xy=(49.062, 79.23),
        xytext=(42.6, 79.82),
        textcoords="data",
        fontsize=8.4,
        color=COL["blue"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=COL["blue"], lw=1.1),
    )
    ax_b.annotate(
        "24.87 ms",
        xy=(24.873, 78.65),
        xytext=(26.8, 78.95),
        textcoords="data",
        fontsize=8.0,
        color="#B45309",
        arrowprops=dict(arrowstyle="-", color="#B45309", lw=1.0),
    )
    ax_b.set_xlabel("COM5 实测延迟 (ms)")
    ax_b.set_ylabel("Top-1 Accuracy (%)")
    ax_b.set_title("板端实测权衡")
    ax_b.set_xlim(22.5, 52.0)
    ax_b.set_ylim(76.0, 80.15)
    ax_b.grid(axis="y")
    handles, labels = ax_b.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax_b.legend(unique.values(), unique.keys(), frameon=False, loc="lower right", handlelength=1.2)
    panel(ax_b, "B")

    validation = list(board)
    x = np.arange(1, len(validation) + 1)
    y = np.array([r["macro_f1"] * 100 for r in validation])
    best = np.maximum.accumulate(y)
    ax_c.plot(x, best, color=COL["blue"], drawstyle="steps-post", label="已验证最好结果")
    ax_c.scatter(x, y, color=[ROLE_COLOR.get(r["role"], COL["gray"]) for r in validation], s=52, zorder=3)
    for xi, row in zip(x, validation):
        ax_c.text(xi, row["macro_f1"] * 100 + 0.18, name(row["arch_id"]), ha="center", va="bottom", fontsize=7.5)
    ax_c.set_xticks(x)
    ax_c.set_xlabel("COM5 验证顺序")
    ax_c.set_ylabel("Best Macro-F1 so far (%)")
    ax_c.set_title("闭环验证过程")
    ax_c.set_xlim(0.75, len(validation) + 0.25)
    ax_c.set_ylim(60.0, 63.4)
    ax_c.grid(axis="y")
    ax_c.legend(frameon=False, loc="lower right")
    panel(ax_c, "C")

    groups = [
        ("全部搜索", [r["macro_f1"] * 100 for r in search], COL["sky"]),
        ("上板候选\n(搜索估计)", [r["macro_f1"] * 100 for r in selected_search], "#E69F00"),
        ("COM5实测", [r["macro_f1"] * 100 for r in board], COL["green"]),
    ]
    box = ax_d.boxplot(
        [g[1] for g in groups],
        positions=np.arange(1, len(groups) + 1),
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
    )
    for patch, (_, _, color) in zip(box["boxes"], groups):
        patch.set(facecolor=color, alpha=0.42, edgecolor=COL["gray"], linewidth=1.0)
    rng = np.random.default_rng(16)
    for i, (_, vals, color) in enumerate(groups, start=1):
        jitter = rng.normal(0, 0.055, len(vals))
        ax_d.scatter(np.full(len(vals), i) + jitter, vals, s=16 if len(vals) > 20 else 34, color=color, alpha=0.55, edgecolors="none")
    ax_d.set_xticks(np.arange(1, len(groups) + 1), [g[0] for g in groups])
    ax_d.set_ylabel("Macro-F1 (%)")
    ax_d.set_title("候选质量分布")
    ax_d.set_ylim(42, 64)
    ax_d.grid(axis="y")
    panel(ax_d, "D")

    fig.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.12, wspace=0.34, hspace=0.44)
    note = "说明：A/D 中“搜索估计”来自候选生成阶段；B/C 与 D 中“COM5实测”来自 AV7K325 完整网络板端闭环。"
    fig.text(0.08, 0.035, note, fontsize=7.4, color=COL["gray"])

    out_base = OUT / "fig05_v1_board_closed_loop_academic"
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)

    with (OUT / "fig05_v1_board_closed_loop_academic_data.csv").open("w", encoding="utf-8-sig", newline="") as fcsv:
        writer = csv.DictWriter(
            fcsv,
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


if __name__ == "__main__":
    draw()
