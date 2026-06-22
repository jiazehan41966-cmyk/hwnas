"""Draw a concise HW-NAS sonar workflow figure.

The figure is intentionally minimal: five clean stages on one horizontal flow,
with a separate bottom feedback loop for board-measurement calibration.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures"
BASE_NAME = "hardware_aware_sonar_nas_flow"


def configure_font() -> font_manager.FontProperties:
    for font_path in (Path("C:/Windows/Fonts/simhei.ttf"), Path("C:/Windows/Fonts/simsun.ttc")):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            prop = font_manager.FontProperties(fname=str(font_path))
            plt.rcParams["font.family"] = prop.get_name()
            break
    else:
        prop = font_manager.FontProperties(family="sans-serif")

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )
    return prop


FONT = configure_font()


def text(ax, x, y, s, *, size=9, color="#222222", weight="normal", ha="center", va="center", ls=1.2):
    ax.text(
        x,
        y,
        s,
        ha=ha,
        va=va,
        fontsize=size,
        color=color,
        fontweight=weight,
        fontproperties=FONT,
        linespacing=ls,
        zorder=10,
    )


def rounded(ax, x, y, w, h, *, fc, ec, lw=1.5, r=0.15, z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.025,rounding_size={r}",
        linewidth=lw,
        facecolor=fc,
        edgecolor=ec,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color="#555555", lw=1.8, rad=0.0, ms=14, alpha=1.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
            zorder=4,
        )
    )


def stage(ax, x, y, w, h, tag, title, keywords, color, fill):
    rounded(ax, x, y, w, h, fc=fill, ec=color, lw=1.7, r=0.18)
    rounded(ax, x + 0.20, y + h - 0.62, 0.56, 0.42, fc=color, ec=color, lw=0, r=0.10, z=3)
    text(ax, x + 0.48, y + h - 0.41, tag, size=9.2, color="white", weight="bold")
    text(ax, x + 0.92, y + h - 0.41, title, size=10.8, color=color, weight="bold", ha="left")
    text(ax, x + w / 2, y + 0.88, keywords, size=9.0, color="#333333", weight="normal", ls=1.45)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15.5, 6.0))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 6.0)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    blue = "#0072B2"
    green = "#009E73"
    purple = "#6B5FB5"
    orange = "#E69F00"
    vermillion = "#D55E00"
    gray = "#5A5A5A"

    text(ax, 7.75, 5.58, "硬件感知的声呐架构搜索总体流程", size=15.5, weight="bold")
    text(ax, 7.75, 5.24, "P1-P5 五阶段闭环", size=9.2, color="#666666")

    y = 2.55
    w = 2.38
    h = 1.86
    xs = [0.55, 3.45, 6.35, 9.25, 12.15]

    stage(ax, xs[0], y, w, h, "P1", "硬件建模", "算子 LUT\nFPGA 约束\nLat / Res", blue, "#EAF4FC")
    stage(ax, xs[1], y, w, h, "P2", "搜索空间", "算子削减\n预算剪枝\nS-FPGA", green, "#EAF7F1")
    stage(ax, xs[2], y, w, h, "P3", "RL 搜索", "Controller\nReward\nPareto Top-K", purple, "#F0EEFC")
    stage(ax, xs[3], y, w, h, "P4", "重训练", "充分训练\nINT8 量化\n最终精度", orange, "#FFF7E6")
    stage(ax, xs[4], y, w, h, "P5", "部署验证", "Bitstream\nCOM5 实测\n偏差分析", vermillion, "#FFF1EA")

    for i, color in enumerate([blue, green, purple, orange]):
        arrow(ax, (xs[i] + w + 0.08, y + h / 2), (xs[i + 1] - 0.10, y + h / 2), color=color, lw=1.7)

    # Bottom feedback loop: it stays below the cards, so it never crosses the main flow.
    feedback_y = 1.25
    p5_bottom = (xs[4] + w / 2, y - 0.02)
    p1_bottom = (xs[0] + w / 2, y - 0.02)
    arrow(ax, p5_bottom, (xs[4] + w / 2, feedback_y), color=vermillion, lw=1.7)
    arrow(ax, (xs[4] + w / 2 - 0.08, feedback_y), (xs[0] + w / 2 + 0.08, feedback_y), color=vermillion, lw=1.7)
    arrow(ax, (xs[0] + w / 2, feedback_y), p1_bottom, color=vermillion, lw=1.7)

    rounded(ax, 5.45, 0.56, 4.60, 0.48, fc="#FFFFFF", ec="#DDC2B5", lw=1.0, r=0.10, z=1)
    text(ax, 7.75, 0.80, "实测偏差回灌：校准 LUT 与代价模型", size=8.8, color=vermillion, weight="bold")

    rounded(ax, 2.20, 0.10, 11.10, 0.32, fc="#F7F7F7", ec="#D9D9D9", lw=0.8, r=0.08, z=1)
    text(
        ax,
        7.75,
        0.26,
        "证据分层：搜索代理指标  |  PyTorch 重训练指标  |  FPGA 板卡实测指标",
        size=7.8,
        color=gray,
    )

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"{BASE_NAME}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
