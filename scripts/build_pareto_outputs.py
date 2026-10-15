from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fnas_style_pareto_outputs_20260816"
SEARCH_ROOT = (
    ROOT
    / "results"
    / "remote_phase0_v3b_lowdsp_prune6_rl300_search"
    / "results"
    / "nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_v3b_lowdsp_prune6"
    / "results"
)
BOARD_JSON = (
    ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "pareto_route_gate_phase0_v3b_lowdsp_prune6"
    / "reports"
    / "phase0_v3_board_candidate_comparison.json"
)


COL = {
    "ink": "#202124",
    "muted": "#8A8F98",
    "grid": "#E7E9EE",
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
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
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "axes.edgecolor": COL["ink"],
            "axes.labelcolor": COL["ink"],
            "xtick.color": COL["ink"],
            "ytick.color": COL["ink"],
        }
    )


def read_search_rows() -> list[dict[str, Any]]:
    path = SEARCH_ROOT / "pareto_ranked_candidates.csv"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out = dict(row)
            for key in [
                "candidate_index",
                "rank",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "top1",
                "latency_ms",
                "energy_mj",
                "dsp",
                "bram",
                "lut",
                "power_w",
                "physical_risk",
                "early_expand_pressure",
            ]:
                if out.get(key) not in (None, ""):
                    out[key] = float(out[key])
            out["selected"] = str(out["selected"]).lower() == "true"
            out["pareto_front"] = str(out["pareto_front"]).lower() == "true"
            rows.append(out)
    return rows


def read_board_rows() -> list[dict[str, Any]]:
    data = json.loads(BOARD_JSON.read_text(encoding="utf-8"))
    rows = []
    roles = {
        data["selections"].get("accuracy_first"): "精度优先",
        data["selections"].get("latency_balanced"): "延迟优先",
        data["selections"].get("resource_min"): "资源优先",
    }
    for item in data["candidates"]:
        rows.append(
            {
                "arch_id": item["arch_id"],
                "role": roles.get(item["arch_id"], "额外候选"),
                "macro_f1": float(item["macro_f1"]),
                "top1": float(item["top1"]),
                "latency_ms": float(item["real_e2e_latency_ms"]),
                "nas_latency_ms": float(item["nas_lut_latency_ms_estimate"]),
                "lut": float(item["actual_lut"]),
                "dsp": float(item["actual_dsp"]),
                "bram": float(item["actual_bram"]),
                "wns_ns": float(item["wns_ns"]),
                "status": item["route_gate_status"],
                "com5_status": item["status_code"],
            }
        )
    return rows


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def clean_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=COL["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def label_point(ax: plt.Axes, x: float, y: float, text: str, dx: float = 0.02, dy: float = 0.002) -> None:
    ax.annotate(
        text.replace("rl_arch_", "arch "),
        (x, y),
        xytext=(x + dx, y + dy),
        textcoords="data",
        fontsize=7.5,
        color=COL["ink"],
    )


def pareto_front_xy(rows: list[dict[str, Any]], x_key: str) -> tuple[list[float], list[float]]:
    front = [r for r in rows if r["pareto_front"]]
    front = sorted(front, key=lambda r: float(r[x_key]))
    return [float(r[x_key]) for r in front], [float(r["macro_f1"]) for r in front]


def plot_search_pareto(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    other = [r for r in rows if not r["pareto_front"]]
    front = [r for r in rows if r["pareto_front"]]
    ax.scatter(
        [r["latency_ms"] for r in other],
        [r["macro_f1"] for r in other],
        s=20,
        c=COL["muted"],
        alpha=0.22,
        edgecolors="none",
        label="搜索候选",
    )
    x, y = pareto_front_xy(rows, "latency_ms")
    ax.plot(x, y, color=COL["ink"], linewidth=1.4, alpha=0.85, linestyle="--", label="多目标前沿投影")
    role_color = {"rl_arch_186": COL["blue"], "rl_arch_242": COL["orange"], "rl_arch_276": COL["green"], "rl_arch_185": COL["purple"]}
    for r in front:
        ax.scatter(r["latency_ms"], r["macro_f1"], s=85, color=role_color.get(r["arch_id"], COL["vermillion"]), edgecolor="white", linewidth=0.8, zorder=4)
        label_point(ax, r["latency_ms"], r["macro_f1"], r["arch_id"], dx=0.025, dy=0.002)
    ax.set_xlabel("搜索阶段估计延迟（ms，越低越好）")
    ax.set_ylabel("各类别平均识别得分（越高越好）")
    ax.set_title("搜索阶段：精度-延迟取舍（多目标前沿投影）")
    clean_axes(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.text(0.02, 0.02, "注：灰点为300个搜索候选；彩色点为多目标前沿投影，估计硬件代价不等同COM5实测。", fontsize=7, color=COL["muted"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    savefig(fig, "fig06_search_pareto_latency_macro")


def plot_search_resource(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7), sharey=True)
    panels = [("dsp", "估计DSP"), ("lut", "估计LUT")]
    for ax, (key, label) in zip(axes, panels):
        other = [r for r in rows if not r["pareto_front"]]
        front = [r for r in rows if r["pareto_front"]]
        ax.scatter([r[key] for r in other], [r["macro_f1"] for r in other], s=18, c=COL["muted"], alpha=0.22, edgecolors="none")
        ax.scatter([r[key] for r in front], [r["macro_f1"] for r in front], s=78, c=COL["orange"], edgecolor="white", linewidth=0.8, zorder=3)
        for r in front:
            label_point(ax, r[key], r["macro_f1"], r["arch_id"], dx=(8 if key == "dsp" else 450), dy=0.001)
        ax.set_xlabel(f"{label}（越低越好）")
        ax.set_title(f"精度-{label}取舍")
        clean_axes(ax)
    axes[0].set_ylabel("各类别平均识别得分")
    fig.suptitle("搜索阶段：资源约束下的帕累托候选", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig07_search_pareto_resources")


def plot_board_pareto(rows: list[dict[str, Any]]) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["latency_ms"])
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    role_color = {"精度优先": COL["blue"], "延迟优先": COL["orange"], "资源优先": COL["green"], "额外候选": COL["purple"]}
    sizes = [60 + (r["dsp"] - min(x["dsp"] for x in rows)) * 0.5 for r in rows]
    for r, s in zip(rows, sizes):
        ax.scatter(r["latency_ms"], r["macro_f1"], s=s, color=role_color[r["role"]], edgecolor="white", linewidth=0.8, label=r["role"], zorder=4)
        offsets = {"rl_arch_186": (-2.4, 0.004), "rl_arch_242": (0.45, 0.001), "rl_arch_276": (0.45, -0.001), "rl_arch_185": (-2.2, -0.006)}
        dx, dy = offsets.get(r["arch_id"], (0.4, 0.0))
        label_point(ax, r["latency_ms"], r["macro_f1"], r["arch_id"], dx=dx, dy=dy)
    ax.plot([r["latency_ms"] for r in rows_sorted], [r["macro_f1"] for r in rows_sorted], color=COL["ink"], linewidth=1.2, alpha=0.55)
    ax.set_xlabel("COM5板端实测延迟（ms，越低越好）")
    ax.set_ylabel("各类别平均识别得分（越高越好）")
    ax.set_title("板级闭环：实测延迟-精度取舍")
    clean_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), frameon=False, loc="lower right")
    fig.text(0.02, 0.02, "注：点大小随实际DSP略增大；这里为已通过全实现和COM5测试的4个候选。", fontsize=7, color=COL["muted"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    savefig(fig, "fig08_board_pareto_latency_macro")


def plot_feedback(rows: list[dict[str, Any]]) -> None:
    labels = [r["arch_id"].replace("rl_arch_", "arch ") for r in rows]
    x = np.arange(len(rows))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    axes[0].bar(x - width / 2, [r["nas_latency_ms"] for r in rows], width, color=COL["sky"], label="搜索估计")
    axes[0].bar(x + width / 2, [r["latency_ms"] for r in rows], width, color=COL["orange"], label="COM5实测")
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].set_ylabel("延迟（ms）")
    axes[0].set_title("估计延迟与板端实测")
    clean_axes(axes[0])
    axes[0].legend(frameon=False)
    axes[1].bar(x - width / 2, [r["dsp"] for r in rows], width, color=COL["green"], label="实际DSP")
    axes[1].bar(x + width / 2, [r["lut"] / 100 for r in rows], width, color=COL["purple"], label="实际LUT/100")
    axes[1].set_xticks(x, labels, rotation=15)
    axes[1].set_ylabel("资源")
    axes[1].set_title("实际资源占用")
    clean_axes(axes[1])
    axes[1].legend(frameon=False)
    fig.suptitle("结果反馈校正：搜索估计到板端实测", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig09_feedback_estimate_vs_board")


def plot_pareto_summary(rows: list[dict[str, Any]], board: list[dict[str, Any]]) -> None:
    front_count = sum(1 for r in rows if r["pareto_front"])
    total = len(rows)
    selected = sum(1 for r in rows if r["selected"])
    claimable = len(board)
    labels = ["搜索候选", "前沿候选", "选中候选", "可上板候选"]
    vals = [total, front_count, selected, claimable]
    colors = [COL["muted"], COL["orange"], COL["blue"], COL["green"]]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    bars = ax.bar(labels, vals, color=colors, width=0.56)
    ax.set_ylabel("数量")
    ax.set_title("从搜索候选到板级候选的筛选漏斗")
    clean_axes(ax)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.025, str(v), ha="center", va="bottom")
    fig.tight_layout()
    savefig(fig, "fig10_pareto_candidate_funnel")


def write_index(search_rows: list[dict[str, Any]], board_rows: list[dict[str, Any]]) -> None:
    front = sum(1 for r in search_rows if r["pareto_front"])
    lines = [
        "# 帕累托图表输出清单",
        "",
        "生成日期：2026-08-16",
        "",
        "## 已生成图",
        "",
        "- `fig06_search_pareto_latency_macro.png/pdf`：搜索阶段精度-估计延迟取舍（多目标前沿投影）",
        "- `fig07_search_pareto_resources.png/pdf`：搜索阶段精度-估计资源取舍",
        "- `fig08_board_pareto_latency_macro.png/pdf`：板级COM5实测延迟-精度取舍",
        "- `fig09_feedback_estimate_vs_board.png/pdf`：搜索估计到板端实测反馈校正",
        "- `fig10_pareto_candidate_funnel.png/pdf`：候选筛选漏斗",
        "",
        "## 已生成表",
        "",
        "- `table08_search_pareto_ranked_candidates.csv`：300个搜索候选及帕累托标记",
        "- `table09_board_pareto_candidates.csv`：4个板级可声明候选",
        "",
        "## 核心数字",
        "",
        f"- 搜索候选：{len(search_rows)}",
        f"- 帕累托前沿候选：{front}",
        f"- 板级可声明候选：{len(board_rows)}",
        "",
        "## 边界",
        "",
        "- 搜索图里的延迟和资源是搜索阶段估计口径。",
        "- 板级图里的延迟和资源来自 Vivado 全实现与 COM5 实测。",
        "- 不把搜索估计延迟写成板端实测延迟。",
    ]
    (OUT / "index_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    search_rows = read_search_rows()
    board_rows = read_board_rows()
    write_csv(OUT / "table08_search_pareto_ranked_candidates.csv", search_rows)
    write_csv(OUT / "table09_board_pareto_candidates.csv", board_rows)
    plot_search_pareto(search_rows)
    plot_search_resource(search_rows)
    plot_board_pareto(board_rows)
    plot_feedback(board_rows)
    plot_pareto_summary(search_rows, board_rows)
    write_index(search_rows, board_rows)


if __name__ == "__main__":
    main()
