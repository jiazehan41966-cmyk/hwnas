#!/usr/bin/env python3
"""HW-NAS 实验结果可视化."""

import json
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ── 全局样式 ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
})

COLORS = {
    "random": "#4C9BE8",
    "rl":     "#E8784C",
    "best":   "#2ECC71",
    "accent": "#9B59B6",
}

RESULTS = Path("results")
RL_DIR  = RESULTS / "hwnas_run_search_rl_nksid_20260314_143749/results/candidates"
RND_DIR = RESULTS / "hwnas_run_search_random_nksid_20260314_134512/results/candidates"
RETRAIN = RESULTS / "retrain/arch_2_retrain.json"


# ── 数据加载 ──────────────────────────────────────────────
def load_candidates(cand_dir: Path):
    records = []
    for fp in sorted(cand_dir.glob("*.json")):
        with open(fp) as f:
            d = json.load(f)
        m = d["candidate"]["metrics"]
        ce = d.get("cost_estimate", {})
        records.append({
            "arch_id":  d["candidate"]["arch_id"],
            "accuracy": m["accuracy"],
            "latency":  m["latency_ms"],
            "lut":      m["lut"],
            "params":   ce.get("params", 0),
            "energy":   m["energy_mj"],
        })
    return records


rnd_cands = load_candidates(RND_DIR)
rl_cands  = load_candidates(RL_DIR)

with open(RETRAIN) as f:
    retrain_data = json.load(f)

retrain_hist    = retrain_data["history"]
retrain_results = retrain_data["results"]


# ═══════════════════════════════════════════════════════════
#  Figure 1: 搜索阶段综合对比（2×2）
# ═══════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(13, 9))
fig1.suptitle("HW-NAS Search Phase Comparison: Random vs RL",
              fontsize=15, fontweight="bold", y=0.98)

# ── 1-A: 精度分布散点图 ──
ax = axes[0, 0]
rnd_accs = [c["accuracy"] * 100 for c in rnd_cands]
rl_accs  = [c["accuracy"] * 100 for c in rl_cands]
ax.scatter(range(len(rnd_cands)), rnd_accs,
           color=COLORS["random"], s=90, zorder=3, label="Random Search")
ax.scatter(range(len(rl_cands)),  rl_accs,
           color=COLORS["rl"],     s=40, alpha=0.7, zorder=2, label="RL Search")
ax.axhline(max(rnd_accs), color=COLORS["random"], ls="--", lw=1.2, alpha=0.6)
ax.axhline(max(rl_accs),  color=COLORS["rl"],     ls="--", lw=1.2, alpha=0.6)
ax.set_xlabel("Candidate Index")
ax.set_ylabel("Validation Accuracy (%)")
ax.set_title("(A) Accuracy per Candidate")
ax.legend(fontsize=9)

# ── 1-B: Accuracy vs Latency Pareto ──
ax = axes[0, 1]
for c in rnd_cands:
    ax.scatter(c["latency"], c["accuracy"] * 100,
               color=COLORS["random"], s=90, zorder=3)
for c in rl_cands:
    ax.scatter(c["latency"], c["accuracy"] * 100,
               color=COLORS["rl"], s=40, alpha=0.6, zorder=2)

# 标注 arch_2
best = max(rnd_cands, key=lambda x: x["accuracy"])
ax.scatter(best["latency"], best["accuracy"] * 100,
           color=COLORS["best"], s=200, marker="*", zorder=5, label="arch_2 (best)")
ax.annotate(f"arch_2\n{best['accuracy']*100:.1f}%",
            xy=(best["latency"], best["accuracy"] * 100),
            xytext=(best["latency"] + 0.05, best["accuracy"] * 100 - 3),
            fontsize=8, color=COLORS["best"])

rnd_patch = mpatches.Patch(color=COLORS["random"], label="Random Search")
rl_patch  = mpatches.Patch(color=COLORS["rl"],     label="RL Search")
ax.legend(handles=[rnd_patch, rl_patch,
          mpatches.Patch(color=COLORS["best"], label="Best (arch_2)")],
          fontsize=8)
ax.set_xlabel("Latency (ms)")
ax.set_ylabel("Validation Accuracy (%)")
ax.set_title("(B) Accuracy vs Latency (Pareto View)")

# ── 1-C: 准确率箱线图 ──
ax = axes[1, 0]
bp = ax.boxplot([rnd_accs, rl_accs],
                labels=["Random\n(n=5)", "RL\n(n=30)"],
                patch_artist=True, widths=0.4,
                medianprops=dict(color="white", linewidth=2))
bp["boxes"][0].set_facecolor(COLORS["random"])
bp["boxes"][1].set_facecolor(COLORS["rl"])
ax.set_ylabel("Validation Accuracy (%)")
ax.set_title("(C) Accuracy Distribution")
for i, (accs, color) in enumerate([(rnd_accs, COLORS["random"]),
                                    (rl_accs,  COLORS["rl"])], 1):
    ax.scatter([i] * len(accs), accs,
               color="white", edgecolors=color, s=40, zorder=3, alpha=0.8)

# ── 1-D: 硬件资源对比（LUT）──
ax = axes[1, 1]
rnd_luts = [c["lut"] for c in rnd_cands]
rl_luts  = [c["lut"] for c in rl_cands]
rnd_accs_all = [c["accuracy"] * 100 for c in rnd_cands]
rl_accs_all  = [c["accuracy"] * 100 for c in rl_cands]
sc1 = ax.scatter(rnd_luts, rnd_accs_all,
                 c=COLORS["random"], s=90, zorder=3, label="Random")
sc2 = ax.scatter(rl_luts,  rl_accs_all,
                 c=COLORS["rl"],     s=40, alpha=0.6, zorder=2, label="RL")
ax.scatter(best["lut"], best["accuracy"] * 100,
           color=COLORS["best"], s=200, marker="*", zorder=5)
ax.set_xlabel("LUT Usage")
ax.set_ylabel("Validation Accuracy (%)")
ax.set_title("(D) Accuracy vs FPGA LUT Usage")
ax.legend(fontsize=9)

fig1.tight_layout()
out1 = "results/fig1_search_comparison.png"
fig1.savefig(out1, bbox_inches="tight")
print(f"Saved: {out1}")


# ═══════════════════════════════════════════════════════════
#  Figure 2: arch_2 完整训练曲线
# ═══════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
fig2.suptitle("arch_2 Full Training (30 epochs, 224×224, NKSID)",
              fontsize=13, fontweight="bold")

epochs = range(1, len(retrain_hist["val_acc"]) + 1)

# Loss
ax = axes2[0]
ax.plot(epochs, retrain_hist["train_loss"],
        color=COLORS["random"], lw=2, label="Train Loss")
ax.plot(epochs, retrain_hist["val_loss"],
        color=COLORS["rl"], lw=2, ls="--", label="Val Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training & Validation Loss")
ax.legend()

# Accuracy
ax = axes2[1]
train_acc_pct = [a * 100 for a in retrain_hist["train_acc"]]
val_acc_pct   = [a * 100 for a in retrain_hist["val_acc"]]
ax.plot(epochs, train_acc_pct,
        color=COLORS["random"], lw=2, label="Train Acc")
ax.plot(epochs, val_acc_pct,
        color=COLORS["rl"], lw=2, ls="--", label="Val Acc")

best_ep  = retrain_results["best_epoch"]
best_acc = retrain_results["best_val_acc"] * 100
ax.scatter([best_ep], [best_acc], color=COLORS["best"],
           s=150, zorder=5, marker="*")
ax.annotate(f"Best: {best_acc:.1f}%\n(epoch {best_ep})",
            xy=(best_ep, best_acc),
            xytext=(best_ep + 0.5, best_acc - 4),
            fontsize=9, color=COLORS["best"])
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Training & Validation Accuracy")
ax.legend()

fig2.tight_layout()
out2 = "results/fig2_arch2_training_curve.png"
fig2.savefig(out2, bbox_inches="tight")
print(f"Saved: {out2}")


# ═══════════════════════════════════════════════════════════
#  Figure 3: 最终模型对比雷达图 + 柱状图
# ═══════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle("Final Model Comparison: arch_2 vs rl_arch_18",
              fontsize=13, fontweight="bold")

# ── 3-A: 关键指标柱状图 ──
ax = axes3[0]
arch2_meta = next(c for c in rnd_cands if c["arch_id"] == "arch_2")
rl18_meta  = next(c for c in rl_cands  if c["arch_id"] == "rl_arch_18")

metrics_labels = ["Val Acc\n(search, %)", "Params\n(×1000)", "Latency\n(ms)", "LUT\n(×1000)"]
arch2_vals = [
    arch2_meta["accuracy"] * 100,
    arch2_meta["params"] / 1000,
    arch2_meta["latency"],
    arch2_meta["lut"] / 1000,
]
rl18_vals = [
    rl18_meta["accuracy"] * 100,
    rl18_meta["params"] / 1000,
    rl18_meta["latency"],
    rl18_meta["lut"] / 1000,
]

x = np.arange(len(metrics_labels))
w = 0.35
bars1 = ax.bar(x - w/2, arch2_vals, w, color=COLORS["random"],
               label="arch_2 (Random)", alpha=0.85)
bars2 = ax.bar(x + w/2, rl18_vals,  w, color=COLORS["rl"],
               label="rl_arch_18 (RL)", alpha=0.85)

def autolabel(bars, vals):
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

autolabel(bars1, arch2_vals)
autolabel(bars2, rl18_vals)
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels, fontsize=9)
ax.set_title("(A) Key Metrics Comparison")
ax.legend(fontsize=9)

# ── 3-B: 雷达图（归一化多维度）──
ax = axes3[1]
ax.remove()
ax = fig3.add_subplot(1, 2, 2, projection="polar")

categories = ["Accuracy", "Lightweight\n(1/Params)", "Speed\n(1/Latency)",
              "LUT Eff.\n(1/LUT)", "Energy Eff.\n(1/Energy)"]
N = len(categories)

# 归一化（越高越好）
def norm(a2, rl, higher_better=True):
    mx = max(a2, rl)
    if mx == 0:
        return 0, 0
    if higher_better:
        return a2/mx, rl/mx
    else:
        mn = min(a2, rl)
        return mn/a2, mn/rl  # 倒数归一化

pairs = [
    norm(arch2_meta["accuracy"],       rl18_meta["accuracy"]),
    norm(1/arch2_meta["params"],       1/rl18_meta["params"]),
    norm(1/arch2_meta["latency"],      1/rl18_meta["latency"]),
    norm(1/arch2_meta["lut"],          1/rl18_meta["lut"]),
    norm(1/arch2_meta["energy"],       1/rl18_meta["energy"]),
]
vals_a2 = [p[0] for p in pairs]
vals_rl = [p[1] for p in pairs]

angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
vals_a2 += [vals_a2[0]]
vals_rl += [vals_rl[0]]
angles  += [angles[0]]

ax.plot(angles, vals_a2, color=COLORS["random"], lw=2,  label="arch_2")
ax.fill(angles, vals_a2, color=COLORS["random"], alpha=0.2)
ax.plot(angles, vals_rl, color=COLORS["rl"],     lw=2,  label="rl_arch_18")
ax.fill(angles, vals_rl, color=COLORS["rl"],     alpha=0.2)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=8)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
ax.set_title("(B) Multi-Dim Radar\n(normalized, higher=better)",
             pad=15, fontsize=10)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=9)

fig3.tight_layout()
out3 = "results/fig3_model_comparison.png"
fig3.savefig(out3, bbox_inches="tight")
print(f"Saved: {out3}")


# ═══════════════════════════════════════════════════════════
#  Figure 4: RL 搜索 reward 学习曲线
# ═══════════════════════════════════════════════════════════
rl_episodes, rl_rewards, rl_baselines, rl_accs_ep = [], [], [], []
for fp in sorted(RL_DIR.glob("*.json"),
                 key=lambda p: int(p.stem.split("_")[-1])):
    with open(fp) as f:
        d = json.load(f)
    extra = d.get("extra", {})
    rl_episodes.append(extra.get("episode", 0))
    rl_rewards.append(extra.get("reward", 0))
    rl_baselines.append(extra.get("baseline", 0))
    rl_accs_ep.append(d["candidate"]["metrics"]["accuracy"] * 100)

fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4))
fig4.suptitle("RL Search Learning Dynamics",
              fontsize=13, fontweight="bold")

ax = axes4[0]
ax.plot(rl_episodes, rl_rewards,   color=COLORS["rl"],
        lw=2, label="Episode Reward")
ax.plot(rl_episodes, rl_baselines, color=COLORS["accent"],
        lw=1.5, ls="--", label="Baseline (EMA)")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.set_title("(A) REINFORCE Reward Curve")
ax.legend()

ax = axes4[1]
# 滑动平均
window = 5
ma = np.convolve(rl_accs_ep, np.ones(window)/window, mode="valid")
ax.scatter(rl_episodes, rl_accs_ep,
           color=COLORS["rl"], s=40, alpha=0.6, label="Episode Acc")
ax.plot(rl_episodes[window-1:], ma,
        color=COLORS["accent"], lw=2, label=f"Moving Avg (w={window})")
ax.axhline(max(rnd_accs), color=COLORS["random"], ls="--", lw=1.5,
           label=f"Random Best ({max(rnd_accs):.1f}%)")
ax.set_xlabel("Episode")
ax.set_ylabel("Val Accuracy (%)")
ax.set_title("(B) RL Accuracy vs Random Baseline")
ax.legend(fontsize=8)

fig4.tight_layout()
out4 = "results/fig4_rl_learning_curve.png"
fig4.savefig(out4, bbox_inches="tight")
print(f"Saved: {out4}")

print("\nAll figures saved to results/")
print("  fig1_search_comparison.png  — 搜索阶段综合对比")
print("  fig2_arch2_training_curve.png — arch_2 完整训练曲线")
print("  fig3_model_comparison.png   — 最终模型多维对比")
print("  fig4_rl_learning_curve.png  — RL 搜索学习动态")
