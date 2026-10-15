from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARETO = ROOT / "results" / "fnas_style_pareto_outputs_20260816"
OUT = ROOT / "results" / "v1_skill_figure_gallery_20260816"

OKABE = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#111111",
    "gray": "#777777",
    "grid": "#E8E8E8",
}

OCEAN = {
    "ink": "#263238",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "coral": "#E76F51",
    "blue": "#264653",
    "sky": "#7CB7D8",
    "grid": "#E7ECEF",
}

ROLE_COLORS = {
    "精度优先": OKABE["blue"],
    "延迟优先": OKABE["orange"],
    "资源优先": OKABE["green"],
    "额外候选": OKABE["purple"],
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def cast(row: dict[str, Any], keys: list[str]) -> None:
    for key in keys:
        row[key] = float(row[key])


def load_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    search = read_csv(PARETO / "table08_search_pareto_ranked_candidates.csv")
    board = read_csv(PARETO / "table09_board_pareto_candidates.csv")
    for row in search:
        cast(row, ["candidate_index", "macro_f1", "top1", "latency_ms", "lut", "dsp", "bram"])
        row["pareto_front"] = str(row["pareto_front"]).lower() == "true"
    for row in board:
        cast(row, ["macro_f1", "top1", "latency_ms", "nas_latency_ms", "lut", "dsp", "bram"])
    return search, board


def base_data(search: list[dict[str, Any]], board: list[dict[str, Any]]) -> dict[str, Any]:
    board_ids = {r["arch_id"] for r in board}
    role = {r["arch_id"]: r["role"] for r in board}
    selected = [r for r in search if r["arch_id"] in board_ids]
    others = [r for r in search if r["arch_id"] not in board_ids]
    front = sorted([r for r in search if r["pareto_front"]], key=lambda r: r["latency_ms"])
    ordered = sorted(search, key=lambda r: r["candidate_index"])
    groups: dict[float, list[float]] = defaultdict(list)
    for row in search:
        groups[round(row["latency_ms"], 2)].append(row["macro_f1"] * 100)
    return {
        "board_ids": board_ids,
        "role": role,
        "selected": selected,
        "others": others,
        "front": front,
        "ordered": ordered,
        "lat_groups": {k: groups[k] for k in sorted(groups)},
    }


def save(fig: plt.Figure, stem: str, *, tiff: bool = False, svg: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png")
    fig.savefig(OUT / f"{stem}.pdf")
    if svg:
        fig.savefig(OUT / f"{stem}.svg")
    if tiff:
        fig.savefig(OUT / f"{stem}.tiff", dpi=600)
    plt.close(fig)


def reset_style() -> None:
    plt.rcdefaults()


def common_panel(ax: plt.Axes, letter: str, *, size: float = 10.0) -> None:
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=size, fontweight="bold", ha="left", va="bottom")


def scivis_style() -> None:
    reset_style()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei", "SimHei"],
            "font.size": 7.0,
            "axes.labelsize": 7.6,
            "axes.titlesize": 7.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.7,
            "figure.dpi": 180,
            "savefig.dpi": 360,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "grid.color": OKABE["grid"],
            "grid.linewidth": 0.65,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot_four_panel(
    search: list[dict[str, Any]],
    board: list[dict[str, Any]],
    stem: str,
    *,
    style: str,
    colors: dict[str, str],
    titles: bool,
    serif: bool = False,
    hero: bool = False,
    tiff: bool = False,
    svg: bool = False,
) -> None:
    d = base_data(search, board)
    if serif:
        plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 5.0) if not hero else (7.2, 5.25))
    ax_a, ax_b, ax_c, ax_d = axs.ravel()

    dot = colors.get("dot", OKABE["sky"])
    line = colors.get("line", OKABE["black"])
    accent = colors.get("accent", OKABE["blue"])

    for ax, metric, ylabel, title, ylim in [
        (ax_a, "macro_f1", "Macro-F1 (%)", "Macro-F1 landscape", (42, 64)),
        (ax_b, "top1", "Top-1 (%)", "Top-1 landscape", (66, 81)),
    ]:
        ax.scatter(
            [r["latency_ms"] for r in d["others"]],
            [r[metric] * 100 for r in d["others"]],
            s=14 if style != "academic" else 16,
            color=dot,
            alpha=0.52 if style != "nature" else 0.42,
            edgecolors="none",
            label=f"Candidates (n={len(search)})",
        )
        ax.plot(
            [r["latency_ms"] for r in d["front"]],
            [r[metric] * 100 for r in d["front"]],
            color=line,
            linestyle=(0, (4, 2)),
            linewidth=1.0,
            label="Pareto front",
        )
        for row in d["selected"]:
            ax.scatter(
                row["latency_ms"],
                row[metric] * 100,
                s=86 if style == "nature" else 100,
                marker="*" if style != "paper" else "D",
                color=ROLE_COLORS.get(d["role"][row["arch_id"]], colors.get("selected", OKABE["orange"])),
                edgecolor="black" if style != "nature" else "white",
                linewidth=0.6,
                zorder=5,
            )
        if titles:
            ax.set_title(title)
        ax.set_xlabel("Estimated latency (ms)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(5.15, 7.42)
        ax.set_ylim(*ylim)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    ax_a.legend(frameon=False, loc="lower right", handlelength=1.5)
    ax_b.legend(frameon=False, loc="lower right", handlelength=1.5)
    common_panel(ax_a, "A")
    common_panel(ax_b, "B")

    ordered = d["ordered"]
    x = np.array([r["candidate_index"] for r in ordered])
    y = np.array([r["macro_f1"] * 100 for r in ordered])
    ax_c.plot(x, np.maximum.accumulate(y), color=accent, linewidth=1.55, drawstyle="steps-post", label="Best so far")
    for row in d["selected"]:
        ax_c.scatter(
            row["candidate_index"],
            row["macro_f1"] * 100,
            s=32,
            color=ROLE_COLORS.get(d["role"][row["arch_id"]], colors.get("selected", OKABE["orange"])),
            edgecolor="white",
            linewidth=0.4,
            zorder=4,
        )
    if titles:
        ax_c.set_title("Search progress")
    ax_c.set_xlabel("Evaluated candidates")
    ax_c.set_ylabel("Best Macro-F1 (%)")
    ax_c.set_xlim(0, 305)
    ax_c.set_ylim(50, 64)
    ax_c.grid(axis="y")
    ax_c.legend(frameon=False, loc="lower right")
    common_panel(ax_c, "C")

    lat_groups = d["lat_groups"]
    latencies = list(lat_groups)
    data = [lat_groups[k] for k in latencies]
    positions = np.arange(len(latencies))
    if style == "academic":
        violin = ax_d.violinplot(data, positions=positions, widths=0.75, showmedians=True, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(dot)
            body.set_alpha(0.32)
            body.set_edgecolor("none")
        violin["cmedians"].set_color(line)
        violin["cmedians"].set_linewidth(1.1)
    else:
        bp = ax_d.boxplot(
            data,
            positions=positions,
            widths=0.56,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.1),
            whiskerprops=dict(color="black", linewidth=0.85),
            capprops=dict(color="black", linewidth=0.85),
        )
        for patch in bp["boxes"]:
            patch.set(facecolor=dot, alpha=0.30 if style == "nature" else 0.36, edgecolor=colors.get("edge", "#999999"), linewidth=0.8)
    rng = np.random.default_rng(16)
    for i, vals in enumerate(data):
        ax_d.scatter(np.full(len(vals), i) + rng.normal(0, 0.04, len(vals)), vals, s=10, color=dot, alpha=0.38, edgecolors="none")
    for row in d["selected"]:
        pos = latencies.index(round(row["latency_ms"], 2))
        ax_d.scatter(
            pos,
            row["macro_f1"] * 100,
            s=72 if style == "nature" else 86,
            marker="*" if style != "paper" else "D",
            color=ROLE_COLORS.get(d["role"][row["arch_id"]], colors.get("selected", OKABE["orange"])),
            edgecolor="black" if style != "nature" else "white",
            linewidth=0.6,
            zorder=5,
        )
    if titles:
        ax_d.set_title("Quality by latency group")
    ax_d.set_xticks(positions, [f"{v:.2f}" for v in latencies])
    ax_d.set_xlabel("Estimated latency group (ms)")
    ax_d.set_ylabel("Macro-F1 (%)")
    ax_d.set_ylim(42, 64)
    ax_d.grid(axis="y")
    common_panel(ax_d, "D")

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.10, top=0.94, wspace=0.34, hspace=0.42)
    save(fig, stem, tiff=tiff, svg=svg)


def draw_scientific_visualization(search: list[dict[str, Any]], board: list[dict[str, Any]]) -> None:
    scivis_style()
    plot_four_panel(
        search,
        board,
        "fig01_scientific_visualization",
        style="scivis",
        colors={"dot": OKABE["sky"], "line": OKABE["black"], "accent": OKABE["blue"], "edge": OKABE["gray"]},
        titles=True,
        svg=True,
    )


def draw_nature(search: list[dict[str, Any]], board: list[dict[str, Any]]) -> None:
    scivis_style()
    plt.rcParams.update({"font.size": 6.8, "axes.titlesize": 7.2, "legend.fontsize": 6.4})
    plot_four_panel(
        search,
        board,
        "fig02_nature_figure",
        style="nature",
        colors={"dot": "#8EC6DF", "line": "#222222", "accent": "#2E6F9E", "edge": "#888888"},
        titles=False,
        tiff=True,
        svg=True,
    )


def draw_paper_figure(search: list[dict[str, Any]], board: list[dict[str, Any]]) -> None:
    scivis_style()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "axes.grid": False})
    plot_four_panel(
        search,
        board,
        "fig03_paper_figure",
        style="paper",
        colors={"dot": "#9ECAE1", "line": "#252525", "accent": "#08519C", "edge": "#969696", "selected": "#E6550D"},
        titles=False,
        serif=True,
    )


def draw_academic_plotting(search: list[dict[str, Any]], board: list[dict[str, Any]]) -> None:
    reset_style()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 7.8,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.0,
            "figure.dpi": 180,
            "savefig.dpi": 360,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": OCEAN["grid"],
            "grid.linewidth": 0.75,
            "pdf.fonttype": 42,
        }
    )
    plot_four_panel(
        search,
        board,
        "fig04_academic_plotting",
        style="academic",
        colors={"dot": OCEAN["sky"], "line": OCEAN["ink"], "accent": OCEAN["coral"], "edge": "#8796A0", "selected": OCEAN["coral"]},
        titles=True,
        serif=True,
    )


def write_index() -> None:
    lines = [
        "# 闭环搜索部分结果图 skill gallery",
        "",
        "数据口径：全部版本均使用 `table08_search_pareto_ranked_candidates.csv` 的 300 个候选；星形/菱形为最终进入板端闭环的候选。",
        "",
        "| 文件 | 对应 skill | 适合用途 |",
        "|---|---|---|",
        "| `fig01_scientific_visualization.png/pdf/svg` | scientific-visualization | 通用论文图，信息完整，适合论文/PPT初稿 |",
        "| `fig02_nature_figure.png/pdf/svg/tiff` | nature-figure | 高水平期刊风格，少标题、低墨量、适合投稿排版 |",
        "| `fig03_paper_figure.png/pdf` | paper-figure | 论文正文风格，Times字体、少装饰，适合LaTeX |",
        "| `fig04_academic_plotting.png/pdf` | academic-plotting | ML会议风格，视觉略活跃，适合NeurIPS/ICML式汇报 |",
        "",
        "不直接画同一张数值图的技能：",
        "",
        "- `figure-spec`：更适合闭环流程/系统 pipeline 的 SVG，不适合散点/箱线数值图。",
        "- `paper-illustration`：用于 AI 方法插图，不适合真实数值结果图；若用它画散点会破坏数据可复现。",
        "- `paper-poster-html`：用于整张会议海报，不是单张结果图。",
        "- `research-paper-writing`：用于图题、caption、claim-evidence 检查，不直接绘图。",
    ]
    (OUT / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    search, board = load_data()
    draw_scientific_visualization(search, board)
    draw_nature(search, board)
    draw_paper_figure(search, board)
    draw_academic_plotting(search, board)
    write_index()


if __name__ == "__main__":
    main()
