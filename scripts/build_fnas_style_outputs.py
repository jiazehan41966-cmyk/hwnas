from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fnas_style_outputs_20260816"


COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#6E6E6E",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def savefig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fmt(value: float | str | None, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def table_dataset_settings() -> list[dict[str, Any]]:
    return [
        {
            "数据集": "NKSID",
            "数据口径": "闭集声呐目标分类",
            "样本规模": "2617张；每折约2094训练/523验证",
            "类别数": 8,
            "输入": "224×224灰度图",
            "训练设置": "搜索10轮；充分重训练150轮",
            "重复次数": "5折×3种子",
            "公平性说明": "主实验口径；不与外部40轮结果直接排榜",
        },
        {
            "数据集": "Roboflow Cylinder2",
            "数据口径": "按原始标注框裁剪后的声呐识别",
            "样本规模": "478张；330训练/99验证/49测试",
            "类别数": 3,
            "输入": "标注框外扩25%，再缩放到224×224灰度图",
            "训练设置": "外部数据完整网络40轮",
            "重复次数": "3种子",
            "公平性说明": "只和同数据集内方法比较",
        },
        {
            "数据集": "Figshare Mine Detection",
            "数据口径": "按原始标注框裁剪后的水雷/非水雷识别",
            "样本规模": "668个目标框；460训练/110验证/98测试",
            "类别数": 2,
            "输入": "标注框外扩25%，再缩放到224×224灰度图",
            "训练设置": "外部数据完整网络40轮",
            "重复次数": "3种子",
            "公平性说明": "只和同数据集内方法比较",
        },
    ]


def table_search_space() -> list[dict[str, Any]]:
    return [
        {"设置项": "板卡", "取值": "ALINX AV7K325，Xilinx Kintex-7 325T", "说明": "本阶段板级验证平台"},
        {"设置项": "目标频率", "取值": "200 MHz", "说明": "对应5 ns时钟周期"},
        {"设置项": "输入", "取值": "1×224×224灰度图", "说明": "与主分类网络一致"},
        {"设置项": "固定主干", "取值": "4阶段，通道16/24/32/32", "说明": "固定Stem、部分阶段和分类头"},
        {"设置项": "可搜索位置", "取值": "主要第2阶段和第4阶段", "说明": "版本1为轻量候选空间"},
        {"设置项": "正式候选", "取值": "2×2×2，共8个候选", "说明": "8候选×5折×3种子=120个正式单元"},
        {"设置项": "搜索代理", "取值": "10轮短训练", "说明": "用于候选筛选，不等同最终精度"},
        {"设置项": "充分重训练", "取值": "150轮", "说明": "用于最终软件精度"},
        {"设置项": "硬件约束", "取值": "200 MHz时序通过；DSP≤700", "说明": "保留布线和控制余量"},
        {"设置项": "板级验证", "取值": "Vivado全实现+COM5测试", "说明": "COM5延迟为板端实测"},
    ]


def table_v1_ablation() -> list[dict[str, Any]]:
    return [
        {"完整网络": "普通网络", "macro_f1_mean": 0.7341, "macro_f1_std": 0.0276, "top1_mean": 0.8365, "top1_std": 0.0273, "相对普通网络变化": 0.0000},
        {"完整网络": "去噪网络", "macro_f1_mean": 0.7223, "macro_f1_std": 0.0378, "top1_mean": 0.8241, "top1_std": 0.0402, "相对普通网络变化": -0.0118},
        {"完整网络": "边缘网络", "macro_f1_mean": 0.6424, "macro_f1_std": 0.0270, "top1_mean": 0.7300, "top1_std": 0.0287, "相对普通网络变化": -0.0917},
        {"完整网络": "去噪+边缘网络", "macro_f1_mean": 0.6368, "macro_f1_std": 0.0301, "top1_mean": 0.7188, "top1_std": 0.0338, "相对普通网络变化": -0.0974},
    ]


def table_board_candidates() -> list[dict[str, Any]]:
    return [
        {"候选": "rl_arch_186", "角色": "精度优先", "精度口径": "150轮重训练", "macro_f1": 0.854482, "top1": 0.915385, "latency_ms": 49.06201, "LUT": 18598, "DSP": 612, "BRAM": 127.5},
        {"候选": "rl_arch_242", "角色": "延迟优先", "精度口径": "150轮重训练", "macro_f1": 0.839023, "top1": 0.913462, "latency_ms": 24.87291, "LUT": 18553, "DSP": 612, "BRAM": 127.5},
        {"候选": "rl_arch_276", "角色": "资源优先", "精度口径": "搜索阶段短训练", "macro_f1": 0.605372, "top1": 0.767308, "latency_ms": 24.83615, "LUT": 15354, "DSP": 524, "BRAM": 101.0},
        {"候选": "rl_arch_185", "角色": "额外可上板候选", "精度口径": "搜索阶段短训练", "macro_f1": 0.619166, "top1": 0.790385, "latency_ms": 49.02574, "LUT": 15380, "DSP": 524, "BRAM": 101.0},
    ]


def table_psnr_ssim() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = [
        ("Roboflow", OUT.parent / "sonar_image_quality_external_v1ops_20260816" / "roboflow_cylider2_v6" / "sonar_image_quality_summary.json"),
        ("Figshare", OUT.parent / "sonar_image_quality_external_v1ops_20260816" / "figshare_mine_detection_v2" / "sonar_image_quality_summary.json"),
    ]
    name_map = {"identity": "原图", "denoise": "去噪", "edge": "边缘", "denoise_edge": "去噪+边缘"}
    for dataset, path in sources:
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data["overall"]:
            rows.append(
                {
                    "数据集": dataset,
                    "处理": name_map.get(item["transform"], item["transform"]),
                    "样本数": item["sample_count"],
                    "PSNR均值": item["psnr_mean"],
                    "PSNR标准差": item["psnr_std"],
                    "SSIM均值": item["ssim_mean"],
                    "SSIM标准差": item["ssim_std"],
                }
            )
    order = {"原图": 0, "去噪": 1, "边缘": 2, "去噪+边缘": 3}
    return sorted(rows, key=lambda r: (r["数据集"], order.get(str(r["处理"]), 9)))


def table_three_dataset_noise() -> list[dict[str, Any]]:
    path = OUT.parent / "external_sonar_noise_analysis_20260816" / "combined_network_summary.csv"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["condition"] not in {"干净", "强噪声"}:
                continue
            rows.append(
                {
                    "数据集": row["dataset"],
                    "网络": row["method"],
                    "条件": row["condition"],
                    "次数": int(row["run_count"]),
                    "macro_f1_mean": float(row["macro_f1_mean"]),
                    "macro_f1_std": float(row["macro_f1_std"]),
                    "top1_mean": float(row["top1_mean"]),
                    "top1_std": float(row["top1_std"]),
                }
            )
    return rows


def table_search_time_audit() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "results").rglob("search_efficiency.json")):
        rel = path.relative_to(ROOT).as_posix()
        data = json.loads(path.read_text(encoding="utf-8"))
        count = data.get("candidate_count")
        method = data.get("search_method", "")
        wall = data.get("wall_clock_seconds")
        lower = rel.lower()
        usable = not any(k in lower for k in ["smoke", "dummy", "local_full_cycle_dummy"]) and (count or 0) >= 100
        rows.append(
            {
                "路径": rel,
                "方法": method,
                "候选数": count,
                "墙钟秒": wall,
                "GPU小时": data.get("gpu_reserved_hours"),
                "是否可作正式搜索耗时": "是" if usable else "否",
                "原因": "候选数足够且路径未标记smoke/dummy" if usable else "smoke/dummy或候选数不足；不用于正式图",
            }
        )
    return rows


def plot_dataset_samples(rows: list[dict[str, Any]]) -> None:
    labels = ["NKSID", "Roboflow", "Figshare"]
    totals = [2617, 478, 668]
    classes = [8, 3, 2]
    fig, ax1 = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(len(labels))
    ax1.bar(x, totals, color=[COLORS["blue"], COLORS["orange"], COLORS["green"]], width=0.55)
    ax1.set_ylabel("样本数")
    ax1.set_xticks(x, labels)
    ax1.set_title("三套声呐数据规模")
    ax1.spines[["top", "right"]].set_visible(False)
    for xi, yi in zip(x, totals):
        ax1.text(xi, yi + max(totals) * 0.025, str(yi), ha="center", va="bottom")
    ax2 = ax1.twinx()
    ax2.plot(x, classes, color=COLORS["red"], marker="o", linewidth=2)
    ax2.set_ylabel("类别数")
    ax2.set_ylim(0, max(classes) + 2)
    ax2.spines["top"].set_visible(False)
    savefig(fig, "fig01_dataset_samples")


def plot_v1_ablation(rows: list[dict[str, Any]]) -> None:
    labels = [r["完整网络"] for r in rows]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    macro = [r["macro_f1_mean"] for r in rows]
    macro_std = [r["macro_f1_std"] for r in rows]
    top1 = [r["top1_mean"] for r in rows]
    top1_std = [r["top1_std"] for r in rows]
    ax.bar(x - width / 2, macro, width, yerr=macro_std, capsize=3, label="各类别平均识别得分", color=COLORS["blue"])
    ax.bar(x + width / 2, top1, width, yerr=top1_std, capsize=3, label="总体正确率", color=COLORS["orange"])
    ax.set_ylim(0.55, 0.90)
    ax.set_ylabel("分数")
    ax.set_title("版本1完整网络消融结果（NKSID）")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.legend(frameon=False, ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "fig02_v1_complete_network_ablation")


def plot_three_dataset_noise(rows: list[dict[str, Any]]) -> None:
    datasets = ["NKSID", "Roboflow", "Figshare"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=False)
    for ax, dataset in zip(axes, datasets):
        sub = [r for r in rows if r["数据集"] == dataset]
        methods = []
        for r in sub:
            if r["网络"] not in methods:
                methods.append(r["网络"])
        clean = {r["网络"]: r for r in sub if r["条件"] == "干净"}
        strong = {r["网络"]: r for r in sub if r["条件"] == "强噪声"}
        x = np.arange(len(methods))
        width = 0.35
        ax.bar(x - width / 2, [clean[m]["macro_f1_mean"] for m in methods], width, label="干净", color=COLORS["blue"])
        ax.bar(x + width / 2, [strong[m]["macro_f1_mean"] for m in methods], width, label="强噪声", color=COLORS["red"])
        ax.set_title(dataset)
        ax.set_xticks(x, [m.replace("网络", "") for m in methods], rotation=25, ha="right")
        ax.set_ylabel("各类别平均识别得分")
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("三数据集完整网络干净/强噪声识别结果", y=1.02)
    fig.subplots_adjust(bottom=0.25, top=0.84, wspace=0.30)
    savefig(fig, "fig03_three_dataset_clean_vs_strong_noise")


def plot_psnr_ssim(rows: list[dict[str, Any]]) -> None:
    rows = [r for r in rows if r["处理"] != "原图"]
    transforms = ["去噪", "边缘", "去噪+边缘"]
    datasets = ["Roboflow", "Figshare"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    x = np.arange(len(transforms))
    width = 0.35
    for metric, ax, ylabel in [
        ("PSNR均值", axes[0], "PSNR"),
        ("SSIM均值", axes[1], "SSIM"),
    ]:
        for i, dataset in enumerate(datasets):
            values = []
            err = []
            for t in transforms:
                r = next(item for item in rows if item["数据集"] == dataset and item["处理"] == t)
                values.append(float(r[metric]))
                err.append(float(r["PSNR标准差"] if metric == "PSNR均值" else r["SSIM标准差"]))
            ax.bar(x + (i - 0.5) * width, values, width, yerr=err, capsize=3, label=dataset, color=[COLORS["blue"], COLORS["orange"]][i])
        ax.set_title(ylabel)
        ax.set_xticks(x, transforms)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)
    fig.suptitle("外部声呐数据图像质量指标", y=1.03)
    savefig(fig, "fig04_external_psnr_ssim")


def plot_board(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    labels = [r["候选"] for r in rows]
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["gray"]]
    axes[0].bar(np.arange(len(rows)), [r["latency_ms"] for r in rows], color=colors)
    axes[0].set_xticks(np.arange(len(rows)), labels, rotation=20, ha="right")
    axes[0].set_ylabel("板端实测延迟（ms）")
    axes[0].set_title("COM5板端延迟")
    axes[0].spines[["top", "right"]].set_visible(False)
    for ax_x, r in enumerate(rows):
        axes[0].text(ax_x, r["latency_ms"] + 1.2, f'{r["latency_ms"]:.1f}', ha="center", va="bottom", fontsize=8)
    offsets = {
        "rl_arch_186": (-2.6, 0.006),
        "rl_arch_242": (0.4, 0.000),
        "rl_arch_276": (0.4, 0.000),
        "rl_arch_185": (-2.6, -0.010),
    }
    for r, c in zip(rows, colors):
        axes[1].scatter(r["latency_ms"], r["macro_f1"], s=80, color=c, label=f'{r["候选"]}（{r["角色"]}）')
        dx, dy = offsets[r["候选"]]
        axes[1].text(r["latency_ms"] + dx, r["macro_f1"] + dy, r["候选"].replace("rl_arch_", "arch "), fontsize=7.5, va="center")
    axes[1].set_xlabel("板端实测延迟（ms）")
    axes[1].set_ylabel("各类别平均识别得分")
    axes[1].set_title("延迟-精度关系")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].set_xlim(22.5, 52.0)
    axes[1].legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.35), ncols=2)
    fig.subplots_adjust(bottom=0.30, wspace=0.30)
    savefig(fig, "fig05_board_latency_accuracy")


def write_report(outputs: list[str], search_rows: list[dict[str, Any]]) -> None:
    usable = [r for r in search_rows if r["是否可作正式搜索耗时"] == "是"]
    lines = [
        "# FNAS 对应图表输出清单",
        "",
        "生成日期：2026-08-16",
        "",
        "## 搜索耗时日志结论",
        "",
        "已扫描现有 `search_efficiency.json`。当前没有可直接用于正式论文图的同协议、等预算搜索耗时结果；已有条目主要是 smoke/dummy 或候选数太少。因此本轮不输出搜索耗时加速图，只输出日志审计表。",
        "",
        f"- 可作正式搜索耗时的日志数量：{len(usable)}",
        "- 搜索耗时审计表：`table_search_time_log_audit.csv`",
        "",
        "## 输出文件",
        "",
    ]
    for item in outputs:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "- NKSID 是主实验口径；Roboflow 和 Figshare 是外部声呐补充验证。",
            "- Roboflow/Figshare 的目标框裁剪来自原始标注框，按框宽高外扩 25%，再统一缩放到 224×224。",
            "- PSNR/SSIM 是图像质量指标，不写成完整分类网络最终指标。",
            "- COM5 延迟是板端实测；功耗本轮不写成实测结果。",
        ]
    )
    (OUT / "index_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()

    dataset_rows = table_dataset_settings()
    search_space_rows = table_search_space()
    v1_rows = table_v1_ablation()
    board_rows = table_board_candidates()
    psnr_rows = table_psnr_ssim()
    noise_rows = table_three_dataset_noise()
    search_rows = table_search_time_audit()

    tables = {
        "table01_dataset_settings.csv": dataset_rows,
        "table02_search_space_hardware_constraints.csv": search_space_rows,
        "table03_v1_complete_network_ablation.csv": v1_rows,
        "table04_three_dataset_clean_strong_network.csv": noise_rows,
        "table05_external_psnr_ssim.csv": psnr_rows,
        "table06_board_latency_accuracy.csv": board_rows,
        "table07_search_time_log_audit.csv": search_rows,
    }
    for name, rows in tables.items():
        write_csv(OUT / name, rows)

    plot_dataset_samples(dataset_rows)
    plot_v1_ablation(v1_rows)
    plot_three_dataset_noise(noise_rows)
    plot_psnr_ssim(psnr_rows)
    plot_board(board_rows)

    outputs = sorted(p.name for p in OUT.iterdir() if p.is_file())
    write_report(outputs, search_rows)


if __name__ == "__main__":
    main()
