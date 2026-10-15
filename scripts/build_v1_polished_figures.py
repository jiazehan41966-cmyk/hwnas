from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v1_polished_figures_20260816"
TABLES = ROOT / "results" / "fnas_style_outputs_20260816"
PARETO = ROOT / "results" / "fnas_style_pareto_outputs_20260816"

COL = {
    "ink": "#1F2933",
    "muted": "#7B8794",
    "grid": "#E4E7EB",
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "bg": "#FFFFFF",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "axes.edgecolor": COL["ink"],
            "axes.labelcolor": COL["ink"],
            "xtick.color": COL["ink"],
            "ytick.color": COL["ink"],
            "figure.facecolor": COL["bg"],
            "axes.facecolor": COL["bg"],
        }
    )


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    text = str(value)
    if text.lower() == "inf":
        return float("inf")
    return float(text)


def clean(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(True, axis="y", color=COL["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def panel(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.13,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        color=COL["ink"],
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def format_arch(arch_id: str) -> str:
    return arch_id.replace("rl_arch_", "候选")


def load_search_rows() -> list[dict[str, Any]]:
    rows = read_csv(PARETO / "table08_search_pareto_ranked_candidates.csv")
    for row in rows:
        for key in ["macro_f1", "top1", "latency_ms", "lut", "dsp", "bram", "rank"]:
            row[key] = to_float(row[key])
        row["pareto_front"] = str(row["pareto_front"]).lower() == "true"
        row["selected"] = str(row["selected"]).lower() == "true"
    return rows


def load_board_rows() -> list[dict[str, Any]]:
    rows = read_csv(PARETO / "table09_board_pareto_candidates.csv")
    for row in rows:
        for key in ["macro_f1", "top1", "latency_ms", "nas_latency_ms", "lut", "dsp", "bram"]:
            row[key] = to_float(row[key])
    return rows


def draw_search_to_board() -> None:
    rows = load_search_rows()
    board = load_board_rows()
    front = [r for r in rows if r["pareto_front"]]
    others = [r for r in rows if not r["pareto_front"]]
    board_ids = {r["arch_id"] for r in board}
    selected_rows = [r for r in rows if r["arch_id"] in board_ids]

    fig = plt.figure(figsize=(10.5, 7.0))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    ax_a.scatter(
        [r["latency_ms"] for r in others],
        [r["macro_f1"] for r in others],
        s=18,
        color=COL["sky"],
        alpha=0.22,
        edgecolors="none",
        label="搜索候选",
    )
    ax_a.scatter(
        [r["latency_ms"] for r in front],
        [r["macro_f1"] for r in front],
        s=80,
        color=COL["orange"],
        edgecolor="white",
        linewidth=0.8,
        label="前沿候选",
        zorder=4,
    )
    for row in front:
        ax_a.annotate(
            format_arch(row["arch_id"]),
            (row["latency_ms"], row["macro_f1"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax_a.set_title("搜索阶段：识别效果与估计延迟")
    ax_a.set_xlabel("估计延迟（毫秒，越低越好）")
    ax_a.set_ylabel("各类别平均识别得分")
    ax_a.legend(frameon=False, loc="lower right")
    clean(ax_a)
    panel(ax_a, "A")

    all_scores = [r["macro_f1"] for r in rows]
    front_scores = [r["macro_f1"] for r in front]
    selected_scores = [r["macro_f1"] for r in selected_rows]
    parts = ax_b.violinplot([all_scores, front_scores, selected_scores], showmeans=False, showmedians=True)
    for body, color in zip(parts["bodies"], [COL["sky"], COL["orange"], COL["green"]]):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.55)
    for key in ["cmedians", "cmins", "cmaxes", "cbars"]:
        parts[key].set_color(COL["ink"])
        parts[key].set_linewidth(1.1)
    rng = np.random.default_rng(7)
    for i, vals in enumerate([all_scores, front_scores, selected_scores], start=1):
        jitter = rng.normal(0, 0.025, len(vals))
        ax_b.scatter(np.full(len(vals), i) + jitter, vals, s=10, color=COL["ink"], alpha=0.16, edgecolors="none")
    ax_b.set_xticks([1, 2, 3], ["全部候选", "前沿候选", "上板候选"])
    ax_b.set_title("候选结构质量分布")
    ax_b.set_ylabel("各类别平均识别得分")
    clean(ax_b)
    panel(ax_b, "B")

    role_colors = {"精度优先": COL["blue"], "延迟优先": COL["orange"], "资源优先": COL["green"], "额外候选": COL["purple"]}
    for row in board:
        ax_c.scatter(
            row["latency_ms"],
            row["macro_f1"],
            s=90 + row["dsp"] * 0.06,
            color=role_colors.get(row["role"], COL["muted"]),
            edgecolor="white",
            linewidth=0.8,
            label=row["role"],
            zorder=4,
        )
        ax_c.annotate(format_arch(row["arch_id"]), (row["latency_ms"], row["macro_f1"]), xytext=(6, 4), textcoords="offset points", fontsize=7)
    ax_c.set_title("板级实测：识别效果与真实延迟")
    ax_c.set_xlabel("COM5 实测延迟（毫秒，越低越好）")
    ax_c.set_ylabel("各类别平均识别得分")
    handles, labels = ax_c.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax_c.legend(unique.values(), unique.keys(), frameon=False, loc="lower right")
    clean(ax_c)
    panel(ax_c, "C")

    ordered = sorted(board, key=lambda r: r["role"])
    x = np.arange(len(ordered))
    width = 0.35
    ax_d.bar(x - width / 2, [r["nas_latency_ms"] for r in ordered], width, color=COL["sky"], label="搜索估计")
    ax_d.bar(x + width / 2, [r["latency_ms"] for r in ordered], width, color=COL["red"], label="板端实测")
    ax_d.set_xticks(x, [format_arch(r["arch_id"]) for r in ordered], rotation=12)
    ax_d.set_ylabel("延迟（毫秒）")
    ax_d.set_title("反馈校正：估计值与实测值")
    ax_d.legend(frameon=False)
    clean(ax_d)
    panel(ax_d, "D")

    fig.suptitle("版本1闭环结果：从候选搜索到 AV7K325 板级验证", fontsize=13, fontweight="bold", y=0.995)
    fig.text(
        0.01,
        0.005,
        "说明：A/B 使用搜索阶段估计结果；C/D 使用 Vivado 完整实现和 COM5 板级测试结果，二者不混写。",
        fontsize=7.5,
        color=COL["muted"],
    )
    save(fig, "fig01_v1_search_to_board_overview")


def draw_board_closed_loop_candidate_quality() -> None:
    rows = load_search_rows()
    board = load_board_rows()
    board_ids = {r["arch_id"] for r in board}
    selected_rows = [r for r in rows if r["arch_id"] in board_ids]
    others = [r for r in rows if r["arch_id"] not in board_ids]

    role_colors = {
        "精度优先": COL["blue"],
        "延迟优先": COL["orange"],
        "资源优先": COL["green"],
        "额外候选": COL["purple"],
    }
    arch_to_role = {r["arch_id"]: r["role"] for r in board}
    arch_to_color = {r["arch_id"]: role_colors.get(r["role"], COL["muted"]) for r in board}

    fig = plt.figure(figsize=(9.0, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    def small_panel(ax: plt.Axes, letter: str) -> None:
        ax.text(
            -0.1,
            1.04,
            letter,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            va="bottom",
            color=COL["ink"],
        )

    ax_a.scatter(
        [r["latency_ms"] for r in others],
        [r["macro_f1"] * 100 for r in others],
        s=13,
        color="#A7D8F0",
        alpha=0.36,
        edgecolors="none",
        label="搜索候选",
    )
    for row in selected_rows:
        ax_a.scatter(
            row["latency_ms"],
            row["macro_f1"] * 100,
            s=70,
            marker="*",
            color=arch_to_color.get(row["arch_id"], COL["orange"]),
            edgecolor=COL["ink"],
            linewidth=0.7,
            zorder=5,
            label=arch_to_role.get(row["arch_id"], "上板候选"),
        )
        ax_a.annotate(
            format_arch(row["arch_id"]),
            (row["latency_ms"], row["macro_f1"] * 100),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
            color=COL["ink"],
        )
    ax_a.set_title("搜索结果：筛出上板候选")
    ax_a.set_xlabel("搜索估计延迟（毫秒）")
    ax_a.set_ylabel("各类别平均识别得分（%）")
    ax_a.set_xlim(5.15, 7.45)
    ax_a.set_ylim(42, 64)
    handles, labels = ax_a.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax_a.legend(uniq.values(), uniq.keys(), frameon=False, loc="lower right", handlelength=1.2)
    clean(ax_a)
    small_panel(ax_a, "A")

    b_offsets = {
        "rl_arch_186": (7, 12),
        "rl_arch_185": (7, -24),
        "rl_arch_242": (5, 7),
        "rl_arch_276": (5, 7),
    }
    for row in board:
        ax_b.scatter(
            row["latency_ms"],
            row["top1"] * 100,
            s=78,
            color=role_colors.get(row["role"], COL["muted"]),
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
            label=row["role"],
        )
        offset = b_offsets.get(row["arch_id"], (6, 3))
        ax_b.annotate(
            f"{format_arch(row['arch_id'])}\n{row['top1'] * 100:.2f}%",
            (row["latency_ms"], row["top1"] * 100),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
            linespacing=1.05,
        )
    ax_b.set_title("COM5 实测：总体识别正确率")
    ax_b.set_xlabel("板端实测延迟（毫秒）")
    ax_b.set_ylabel("总体识别正确率（%）")
    ax_b.set_xlim(22.5, 51.5)
    ax_b.set_ylim(75.7, 80.1)
    clean(ax_b)
    small_panel(ax_b, "B")

    ax_c.axis("off")
    ax_c.set_title("闭环数量：从搜索到板端通过", pad=8)
    steps = [
        ("搜索候选", len(rows), COL["sky"]),
        ("前沿筛选", len(selected_rows), COL["orange"]),
        ("完整实现", len(board), COL["green"]),
        ("COM5通过", sum(1 for r in board if str(r.get("status", "")).upper() == "PASS"), COL["blue"]),
    ]
    x0, y0 = 0.03, 0.5
    box_w, box_h, gap = 0.2, 0.28, 0.045
    for i, (name, count, color) in enumerate(steps):
        x_pos = x0 + i * (box_w + gap)
        box = plt.Rectangle((x_pos, y0), box_w, box_h, transform=ax_c.transAxes, facecolor=color, alpha=0.18, edgecolor=color, linewidth=1.2)
        ax_c.add_patch(box)
        ax_c.text(x_pos + box_w / 2, y0 + box_h * 0.62, name, transform=ax_c.transAxes, ha="center", va="center", fontsize=8.5, color=COL["ink"])
        ax_c.text(x_pos + box_w / 2, y0 + box_h * 0.28, f"{count} 个", transform=ax_c.transAxes, ha="center", va="center", fontsize=11, fontweight="bold", color=color)
        if i < len(steps) - 1:
            ax_c.annotate(
                "",
                xy=(x_pos + box_w + gap * 0.72, y0 + box_h / 2),
                xytext=(x_pos + box_w + gap * 0.2, y0 + box_h / 2),
                xycoords=ax_c.transAxes,
                textcoords=ax_c.transAxes,
                arrowprops=dict(arrowstyle="->", color=COL["muted"], lw=1.0),
            )
    ax_c.text(
        0.03,
        0.26,
        "含义：这一批闭环证据重点证明“筛出来的候选能完整落到板上并完成实测”。",
        transform=ax_c.transAxes,
        fontsize=7.6,
        color=COL["muted"],
    )
    small_panel(ax_c, "C")

    ordered = sorted(board, key=lambda r: (r["latency_ms"], -r["macro_f1"]))
    x = np.arange(len(ordered))
    width = 0.34
    macro_vals = [r["macro_f1"] * 100 for r in ordered]
    top1_vals = [r["top1"] * 100 for r in ordered]
    ax_d.bar(x - width / 2, macro_vals, width, color=COL["blue"], label="各类别平均识别得分")
    ax_d.bar(x + width / 2, top1_vals, width, color=COL["orange"], label="总体识别正确率")
    for xi, row in zip(x, ordered):
        ax_d.text(xi - width / 2, row["macro_f1"] * 100 + 0.35, f"{row['macro_f1'] * 100:.1f}", ha="center", va="bottom", fontsize=6.8, color=COL["ink"])
        ax_d.text(xi + width / 2, row["top1"] * 100 + 0.35, f"{row['top1'] * 100:.1f}", ha="center", va="bottom", fontsize=6.8, color=COL["ink"])
    ax_d.set_xticks(x, [format_arch(r["arch_id"]) for r in ordered], rotation=0)
    ax_d.set_ylabel("得分（%）")
    ax_d.set_title("完整网络上板后的识别指标")
    ax_d.set_ylim(58, 83.5)
    ax_d.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, handlelength=1.4)
    clean(ax_d)
    small_panel(ax_d, "D")

    fig.suptitle("板端闭环候选质量：搜索筛选到 COM5 实测", fontsize=11.5, fontweight="bold", y=0.985)
    fig.text(
        0.01,
        0.006,
        "说明：A 是搜索阶段估计结果；B-D 是 AV7K325 完整网络实现后的 COM5 板端实测结果。",
        fontsize=7.2,
        color=COL["muted"],
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.085, top=0.9)
    save(fig, "fig04_v1_board_closed_loop_candidate_quality")

    data_path = OUT / "fig04_v1_board_closed_loop_candidate_quality_data.csv"
    with data_path.open("w", encoding="utf-8-sig", newline="") as f:
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
                "status",
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
                    "status": row["status"],
                }
            )


def draw_complete_network_metrics() -> None:
    ablation = read_csv(TABLES / "table03_v1_complete_network_ablation.csv")
    network = read_csv(TABLES / "table04_three_dataset_clean_strong_network.csv")
    quality = read_csv(TABLES / "table05_external_psnr_ssim.csv")

    fig = plt.figure(figsize=(10.8, 7.2))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    names = [r["完整网络"] for r in ablation]
    x = np.arange(len(names))
    width = 0.36
    macro = [to_float(r["macro_f1_mean"]) for r in ablation]
    macro_std = [to_float(r["macro_f1_std"]) for r in ablation]
    top1 = [to_float(r["top1_mean"]) for r in ablation]
    top1_std = [to_float(r["top1_std"]) for r in ablation]
    ax_a.bar(x - width / 2, macro, width, yerr=macro_std, color=COL["blue"], capsize=3, label="各类别平均识别得分")
    ax_a.bar(x + width / 2, top1, width, yerr=top1_std, color=COL["orange"], capsize=3, label="总体识别正确率")
    ax_a.set_xticks(x, names, rotation=15, ha="right")
    ax_a.set_ylim(0.55, 0.9)
    ax_a.set_title("完整网络消融：处理方式是否真的有效")
    ax_a.set_ylabel("得分")
    ax_a.legend(frameon=False, loc="upper right")
    clean(ax_a)
    panel(ax_a, "A")

    datasets = ["NKSID", "Roboflow", "Figshare"]
    nets = ["普通网络", "新去噪算子网络", "新边缘算子网络", "新去噪+新边缘网络"]
    matrix = np.full((len(datasets), len(nets)), np.nan)
    for i, ds in enumerate(datasets):
        for j, net in enumerate(nets):
            vals = [
                to_float(r["macro_f1_mean"])
                for r in network
                if r["数据集"] == ds and r["网络"] == net and r["条件"] == "强噪声"
            ]
            if vals:
                matrix[i, j] = vals[0]
    im = ax_b.imshow(matrix, cmap="YlGnBu", vmin=0.1, vmax=0.75, aspect="auto")
    ax_b.set_xticks(np.arange(len(nets)), ["普通", "新去噪", "新边缘", "新去噪+新边缘"], rotation=20, ha="right")
    ax_b.set_yticks(np.arange(len(datasets)), datasets)
    for i in range(len(datasets)):
        for j in range(len(nets)):
            if np.isfinite(matrix[i, j]):
                ax_b.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8, color=COL["ink"])
            else:
                ax_b.text(j, i, "无", ha="center", va="center", fontsize=8, color=COL["muted"])
    ax_b.set_title("三套声呐数据：强噪声下完整网络得分")
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.03, label="各类别平均识别得分")
    panel(ax_b, "B")

    qrows = [r for r in quality if r["处理"] != "原图"]
    proc = ["去噪", "边缘", "去噪+边缘"]
    ds2 = ["Roboflow", "Figshare"]
    x = np.arange(len(proc))
    width = 0.35
    for offset, ds, color in [(-width / 2, "Roboflow", COL["blue"]), (width / 2, "Figshare", COL["green"])]:
        vals = [to_float(next(r["PSNR均值"] for r in qrows if r["数据集"] == ds and r["处理"] == p)) for p in proc]
        errs = [to_float(next(r["PSNR标准差"] for r in qrows if r["数据集"] == ds and r["处理"] == p)) for p in proc]
        ax_c.bar(x + offset, vals, width, yerr=errs, capsize=3, color=color, label=ds)
    ax_c.set_xticks(x, proc)
    ax_c.set_title("图像质量：峰值信噪比（PSNR）")
    ax_c.set_ylabel("峰值信噪比（越高越接近原图）")
    ax_c.legend(frameon=False)
    clean(ax_c)
    panel(ax_c, "C")

    for offset, ds, color in [(-width / 2, "Roboflow", COL["orange"]), (width / 2, "Figshare", COL["purple"])]:
        vals = [to_float(next(r["SSIM均值"] for r in qrows if r["数据集"] == ds and r["处理"] == p)) for p in proc]
        errs = [to_float(next(r["SSIM标准差"] for r in qrows if r["数据集"] == ds and r["处理"] == p)) for p in proc]
        ax_d.bar(x + offset, vals, width, yerr=errs, capsize=3, color=color, label=ds)
    ax_d.set_xticks(x, proc)
    ax_d.set_ylim(0.0, 1.0)
    ax_d.set_title("图像质量：结构相似度（SSIM）")
    ax_d.set_ylabel("结构相似度（越高越接近原图）")
    ax_d.legend(frameon=False)
    clean(ax_d)
    panel(ax_d, "D")

    fig.suptitle("版本1完整网络结果：识别指标与图像质量指标", fontsize=13, fontweight="bold", y=0.995)
    fig.text(
        0.01,
        0.005,
        "说明：峰值信噪比和结构相似度以原图为参照；原图自身为参考值，因此图中只展示三种处理方式。",
        fontsize=7.5,
        color=COL["muted"],
    )
    save(fig, "fig02_v1_complete_network_and_quality")


def pareto_mask(rows: list[dict[str, Any]], x_key: str, y_key: str) -> list[bool]:
    mask = []
    for row in rows:
        dominated = False
        for other in rows:
            no_worse = other[x_key] <= row[x_key] and other[y_key] >= row[y_key]
            strictly_better = other[x_key] < row[x_key] or other[y_key] > row[y_key]
            if no_worse and strictly_better:
                dominated = True
                break
        mask.append(not dominated)
    return mask


def draw_latest_pareto_front() -> None:
    rows = load_search_rows()
    board = load_board_rows()
    search_front = [r for r in rows if r["pareto_front"]]
    search_other = [r for r in rows if not r["pareto_front"]]
    board_mask = pareto_mask(board, "latency_ms", "macro_f1")
    board_front = [r for r, keep in zip(board, board_mask) if keep]
    board_other = [r for r, keep in zip(board, board_mask) if not keep]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    ax.scatter(
        [r["latency_ms"] for r in search_other],
        [r["macro_f1"] for r in search_other],
        s=20,
        color=COL["sky"],
        alpha=0.22,
        edgecolors="none",
        label="其他候选",
    )
    search_front_sorted = sorted(search_front, key=lambda r: r["latency_ms"])
    ax.plot(
        [r["latency_ms"] for r in search_front_sorted],
        [r["macro_f1"] for r in search_front_sorted],
        color=COL["ink"],
        linewidth=1.6,
        linestyle="--",
        label="最新版前沿",
        zorder=3,
    )
    ax.scatter(
        [r["latency_ms"] for r in search_front],
        [r["macro_f1"] for r in search_front],
        s=95,
        color=COL["orange"],
        edgecolor="white",
        linewidth=0.9,
        zorder=4,
        label="前沿候选",
    )
    for row in search_front:
        ax.annotate(format_arch(row["arch_id"]), (row["latency_ms"], row["macro_f1"]), xytext=(6, 5), textcoords="offset points", fontsize=7.5)
    ax.set_title("搜索阶段最新版帕累托图")
    ax.set_xlabel("估计延迟（毫秒，越低越好）")
    ax.set_ylabel("各类别平均识别得分（越高越好）")
    ax.legend(frameon=False, loc="lower right")
    clean(ax)
    panel(ax, "A")

    ax = axes[1]
    if board_other:
        ax.scatter(
            [r["latency_ms"] for r in board_other],
            [r["macro_f1"] for r in board_other],
            s=90,
            color=COL["muted"],
            alpha=0.55,
            edgecolor="white",
            linewidth=0.8,
            label="非前沿上板候选",
            zorder=3,
        )
    board_front_sorted = sorted(board_front, key=lambda r: r["latency_ms"])
    ax.plot(
        [r["latency_ms"] for r in board_front_sorted],
        [r["macro_f1"] for r in board_front_sorted],
        color=COL["ink"],
        linewidth=1.8,
        linestyle="--",
        label="实测前沿",
        zorder=3,
    )
    role_colors = {"精度优先": COL["blue"], "延迟优先": COL["orange"], "资源优先": COL["green"], "额外候选": COL["purple"]}
    for row in board_front:
        ax.scatter(
            row["latency_ms"],
            row["macro_f1"],
            s=120,
            color=role_colors.get(row["role"], COL["orange"]),
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
            label=row["role"],
        )
    for row in board:
        ax.annotate(format_arch(row["arch_id"]), (row["latency_ms"], row["macro_f1"]), xytext=(6, 5), textcoords="offset points", fontsize=7.5)
    ax.set_title("COM5 实测最新版帕累托图")
    ax.set_xlabel("COM5 实测延迟（毫秒，越低越好）")
    ax.set_ylabel("各类别平均识别得分（越高越好）")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, loc="lower right")
    clean(ax)
    panel(ax, "B")

    fig.suptitle("版本1最新版帕累托结果：估计前沿与板端实测前沿分开报告", fontsize=12.5, fontweight="bold", y=1.02)
    fig.text(
        0.01,
        -0.01,
        "说明：左图是搜索阶段估计结果；右图是通过完整实现和 COM5 板级测试后的实测结果。旧版帕累托图没有体现这层证据分离。",
        fontsize=7.5,
        color=COL["muted"],
    )
    fig.tight_layout()
    save(fig, "fig03_v1_latest_pareto_front")


def draw_latest_pareto_front_clean() -> None:
    rows = load_search_rows()
    board = load_board_rows()
    search_mask = pareto_mask(rows, "latency_ms", "macro_f1")
    search_front = [r for r, keep in zip(rows, search_mask) if keep]
    search_other = [r for r, keep in zip(rows, search_mask) if not keep]
    board_mask = pareto_mask(board, "latency_ms", "macro_f1")
    board_front = [r for r, keep in zip(board, board_mask) if keep]
    board_other = [r for r, keep in zip(board, board_mask) if not keep]

    fig = plt.figure(figsize=(8.6, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    front_sorted = sorted(search_front, key=lambda r: r["latency_ms"])
    ax_a.scatter(
        [r["latency_ms"] for r in search_other],
        [r["macro_f1"] for r in search_other],
        s=13,
        color="#B8DFF2",
        alpha=0.55,
        edgecolors="none",
        label="其他候选",
    )
    ax_a.plot(
        [r["latency_ms"] for r in front_sorted],
        [r["macro_f1"] for r in front_sorted],
        color=COL["ink"],
        linewidth=1.2,
        label="二维前沿",
        zorder=3,
    )
    ax_a.scatter(
        [r["latency_ms"] for r in front_sorted],
        [r["macro_f1"] for r in front_sorted],
        s=42,
        color=COL["orange"],
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label="前沿点",
    )
    ax_a.set_title("搜索候选空间")
    ax_a.set_xlabel("估计延迟（毫秒）")
    ax_a.set_ylabel("平均识别得分")
    ax_a.set_xlim(5.15, 7.42)
    ax_a.set_ylim(0.43, 0.64)
    ax_a.legend(frameon=False, loc="lower right")
    clean(ax_a)
    panel(ax_a, "A")

    zoom = [r for r in rows if r["macro_f1"] >= 0.59]
    zoom_front = [r for r in search_front if r["macro_f1"] >= 0.59]
    zoom_front_sorted = sorted(zoom_front, key=lambda r: r["latency_ms"])
    ax_b.scatter(
        [r["latency_ms"] for r in zoom],
        [r["macro_f1"] for r in zoom],
        s=20,
        color="#B8DFF2",
        alpha=0.45,
        edgecolors="none",
    )
    ax_b.plot(
        [r["latency_ms"] for r in zoom_front_sorted],
        [r["macro_f1"] for r in zoom_front_sorted],
        color=COL["ink"],
        linewidth=1.2,
        zorder=3,
    )
    ax_b.scatter(
        [r["latency_ms"] for r in zoom_front_sorted],
        [r["macro_f1"] for r in zoom_front_sorted],
        s=52,
        color=COL["orange"],
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    for row in zoom_front_sorted:
        ax_b.annotate(
            format_arch(row["arch_id"]),
            (row["latency_ms"], row["macro_f1"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.8,
        )
    ax_b.set_title("搜索前沿放大")
    ax_b.set_xlabel("估计延迟（毫秒）")
    ax_b.set_ylabel("平均识别得分")
    ax_b.set_xlim(5.15, 7.42)
    ax_b.set_ylim(0.595, 0.635)
    clean(ax_b)
    panel(ax_b, "B")

    board_front_sorted = sorted(board_front, key=lambda r: r["latency_ms"])
    if board_other:
        ax_c.scatter(
            [r["latency_ms"] for r in board_other],
            [r["macro_f1"] for r in board_other],
            s=58,
            color="#BFC7D1",
            alpha=0.9,
            edgecolor="white",
            linewidth=0.7,
            label="非前沿",
            zorder=3,
        )
    ax_c.plot(
        [r["latency_ms"] for r in board_front_sorted],
        [r["macro_f1"] for r in board_front_sorted],
        color=COL["ink"],
        linewidth=1.25,
        label="实测前沿",
        zorder=3,
    )
    role_colors = {"精度优先": COL["blue"], "延迟优先": COL["orange"], "资源优先": COL["green"], "额外候选": COL["purple"]}
    for row in board_front_sorted:
        ax_c.scatter(
            row["latency_ms"],
            row["macro_f1"],
            s=68,
            color=role_colors.get(row["role"], COL["orange"]),
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        ax_c.annotate(format_arch(row["arch_id"]), (row["latency_ms"], row["macro_f1"]), xytext=(4, 4), textcoords="offset points", fontsize=6.8)
    for row in board_other:
        ax_c.annotate(format_arch(row["arch_id"]), (row["latency_ms"], row["macro_f1"]), xytext=(4, -10), textcoords="offset points", fontsize=6.8, color=COL["muted"])
    ax_c.set_title("COM5 实测前沿")
    ax_c.set_xlabel("实测延迟（毫秒）")
    ax_c.set_ylabel("平均识别得分")
    ax_c.set_xlim(23.5, 50.5)
    ax_c.set_ylim(0.603, 0.631)
    ax_c.legend(frameon=False, loc="lower right")
    clean(ax_c)
    panel(ax_c, "C")

    ordered = sorted(board, key=lambda r: r["latency_ms"])
    y = np.arange(len(ordered))
    ax_d.hlines(y, [r["nas_latency_ms"] for r in ordered], [r["latency_ms"] for r in ordered], color=COL["grid"], linewidth=3)
    ax_d.scatter([r["nas_latency_ms"] for r in ordered], y, s=36, color=COL["sky"], label="搜索估计", zorder=3)
    ax_d.scatter([r["latency_ms"] for r in ordered], y, s=42, color=COL["red"], label="板端实测", zorder=3)
    ax_d.set_yticks(y, [format_arch(r["arch_id"]) for r in ordered])
    ax_d.set_xlabel("延迟（毫秒）")
    ax_d.set_title("估计到实测的变化")
    ax_d.set_xlim(0, 52)
    ax_d.legend(frameon=False, loc="lower right")
    clean(ax_d)
    panel(ax_d, "D")

    fig.suptitle("版本1帕累托结果", fontsize=12, fontweight="bold", y=0.99)
    fig.text(
        0.01,
        0.005,
        "说明：A/B 为搜索阶段估计结果；C/D 为 AV7K325 上完整实现后的 COM5 实测结果。",
        fontsize=7,
        color=COL["muted"],
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    save(fig, "fig03_v1_latest_pareto_clean")


def draw_latest_pareto_front_compact() -> None:
    rows = load_search_rows()
    board = load_board_rows()
    search_mask = pareto_mask(rows, "latency_ms", "macro_f1")
    search_front = [r for r, keep in zip(rows, search_mask) if keep]
    search_other = [r for r, keep in zip(rows, search_mask) if not keep]
    board_mask = pareto_mask(board, "latency_ms", "macro_f1")
    board_front = [r for r, keep in zip(board, board_mask) if keep]
    board_other = [r for r, keep in zip(board, board_mask) if not keep]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for letter, ax in zip(["A", "B"], axes):
        ax.text(
            -0.1,
            1.03,
            letter,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            va="bottom",
            color=COL["ink"],
        )

    ax = axes[0]
    front_sorted = sorted(search_front, key=lambda r: r["latency_ms"])
    ax.scatter(
        [r["latency_ms"] for r in search_other],
        [r["macro_f1"] for r in search_other],
        s=12,
        color="#A7D8F0",
        alpha=0.42,
        edgecolors="none",
        label="其他候选",
    )
    ax.plot(
        [r["latency_ms"] for r in front_sorted],
        [r["macro_f1"] for r in front_sorted],
        color=COL["ink"],
        linewidth=1.25,
        zorder=3,
        label="前沿",
    )
    ax.scatter(
        [r["latency_ms"] for r in front_sorted],
        [r["macro_f1"] for r in front_sorted],
        s=42,
        color=COL["orange"],
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label="前沿候选",
    )
    for row in front_sorted:
        ax.annotate(
            format_arch(row["arch_id"]),
            (row["latency_ms"], row["macro_f1"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.8,
            color=COL["ink"],
        )
    ax.set_title("搜索阶段估计前沿", fontsize=9.5)
    ax.set_xlabel("估计延迟（毫秒）")
    ax.set_ylabel("平均识别得分")
    ax.set_xlim(5.15, 7.42)
    ax.set_ylim(0.43, 0.635)
    ax.legend(frameon=False, loc="lower right", handlelength=1.5)
    clean(ax)

    ax = axes[1]
    if board_other:
        ax.scatter(
            [r["latency_ms"] for r in board_other],
            [r["macro_f1"] for r in board_other],
            s=42,
            color="#BFC7D1",
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
            label="非前沿",
            zorder=2,
        )
    board_front_sorted = sorted(board_front, key=lambda r: r["latency_ms"])
    ax.plot(
        [r["latency_ms"] for r in board_front_sorted],
        [r["macro_f1"] for r in board_front_sorted],
        color=COL["ink"],
        linewidth=1.25,
        zorder=3,
        label="前沿",
    )
    role_colors = {"精度优先": COL["blue"], "延迟优先": COL["orange"], "资源优先": COL["green"], "额外候选": COL["purple"]}
    for row in board_front_sorted:
        ax.scatter(
            row["latency_ms"],
            row["macro_f1"],
            s=54,
            color=role_colors.get(row["role"], COL["orange"]),
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        ax.annotate(format_arch(row["arch_id"]), (row["latency_ms"], row["macro_f1"]), xytext=(4, 4), textcoords="offset points", fontsize=6.8)
    for row in board_other:
        ax.annotate(format_arch(row["arch_id"]), (row["latency_ms"], row["macro_f1"]), xytext=(4, -9), textcoords="offset points", fontsize=6.8, color=COL["muted"])
    ax.set_title("COM5 板端实测前沿", fontsize=9.5)
    ax.set_xlabel("实测延迟（毫秒）")
    ax.set_ylabel("平均识别得分")
    ax.set_xlim(23.5, 50.5)
    ax.set_ylim(0.603, 0.631)
    ax.legend(frameon=False, loc="lower right", handlelength=1.5)
    clean(ax)

    fig.suptitle("版本1帕累托结果", fontsize=10.5, fontweight="bold", y=1.02)
    fig.text(
        0.01,
        -0.01,
        "左：搜索阶段估计；右：AV7K325 完整实现后的 COM5 实测。",
        fontsize=6.8,
        color=COL["muted"],
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.22, top=0.84, wspace=0.32)
    save(fig, "fig03_v1_latest_pareto_compact")


def write_index() -> None:
    lines = [
        "# 版本1 polished 图表输出",
        "",
        "生成日期：2026-08-16",
        "",
        "## 图",
        "",
        "- `fig01_v1_search_to_board_overview.png/pdf`：候选搜索、候选分布、板级实测、反馈校正。",
        "- `fig02_v1_complete_network_and_quality.png/pdf`：完整网络消融、三数据集强噪声结果、Roboflow/Figshare 图像质量指标。",
        "- `fig03_v1_latest_pareto_front.png/pdf`：最新版帕累托图，分开展示搜索估计前沿和 COM5 实测前沿。",
        "- `fig03_v1_latest_pareto_clean.png/pdf`：更适合放论文/PPT的精简版帕累托图。",
        "- `fig03_v1_latest_pareto_compact.png/pdf`：最简洁版帕累托图，建议优先使用。",
        "- `fig04_v1_board_closed_loop_candidate_quality.png/pdf`：板端闭环候选质量图，展示搜索筛选到 COM5 实测通过的完整链条。",
        "- `fig04_v1_board_closed_loop_candidate_quality_data.csv`：板端闭环图的数据版本。",
        "",
        "## 对应表",
        "",
        "- `../fnas_style_outputs_20260816/table03_v1_complete_network_ablation.csv`：完整网络消融结果。",
        "- `../fnas_style_outputs_20260816/table04_three_dataset_clean_strong_network.csv`：三套声呐数据完整网络结果。",
        "- `../fnas_style_outputs_20260816/table05_external_psnr_ssim.csv`：Roboflow/Figshare 图像质量结果。",
        "- `../fnas_style_pareto_outputs_20260816/table08_search_pareto_ranked_candidates.csv`：300个搜索候选与前沿标记。",
        "- `../fnas_style_pareto_outputs_20260816/table09_board_pareto_candidates.csv`：4个板级候选结果。",
        "",
        "## 使用边界",
        "",
        "- 搜索阶段的延迟是估计值，不能写成板端实测。",
        "- 板级延迟来自 COM5 测试结果。",
        "- 图像质量指标只覆盖 Roboflow 和 Figshare，因为这两套外部数据已有原图处理结果。",
        "- 搜索耗时没有正式日志，因此本轮不补搜索耗时图。",
        "",
        "## 指标中文说明",
        "",
        "- 各类别平均识别得分：先分别计算每个类别的识别表现，再取平均，适合类别不均衡的声呐数据。",
        "- 总体识别正确率：全部样本中识别正确的比例。",
        "- 峰值信噪比：处理后图像与原图的差异，越高说明越接近原图。",
        "- 结构相似度：处理后图像与原图结构的一致程度，越高说明结构保持越好。",
    ]
    (OUT / "index_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    draw_search_to_board()
    draw_board_closed_loop_candidate_quality()
    draw_complete_network_metrics()
    draw_latest_pareto_front()
    draw_latest_pareto_front_clean()
    draw_latest_pareto_front_compact()
    write_index()


if __name__ == "__main__":
    main()
