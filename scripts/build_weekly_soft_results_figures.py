from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "weekly_group_meeting_20260908" / "soft_results_visualization"

FROZEN_JSON = (
    ROOT
    / "artifacts"
    / "zcu104_full_campaign_v1"
    / "frozen_candidates"
    / "mobilenetA_rl200_20260908_155713"
    / "rl_frozen_mobilenetA_pareto_candidates_20260908_155713.json"
)
RETRAIN_CSV = (
    ROOT
    / "artifacts"
    / "zcu104_full_campaign_v1"
    / "retrain_protocol_v1_summaries"
    / "zcu104_mobilenetA_retrain_protocol_v1_summary.csv"
)
PTQ_CSV = (
    ROOT
    / "artifacts"
    / "zcu104_full_campaign_v1"
    / "int8_complexity"
    / "mobilenetA_rl200_20260908_1640"
    / "zcu104_mobilenetA_ptq_complexity_summary.csv"
)
QUALITY_CSV = ROOT / "results" / "v1_polished_figures_20260816" / "sonar_task_quality_metrics_retrain_aligned.csv"
BASELINE_DENOISE_CSV = ROOT / "results" / "v1_polished_figures_20260816" / "latest_baseline_vs_denoise_task_metrics.csv"
M1_PACKAGE = (
    ROOT
    / "results"
    / "zcu104_full_campaign_v1"
    / "search"
    / "m1_mobilenetA_dmdc_cfem_rl_20260907_2205"
    / "m1_completion_package_20260908_061514"
)
M1_SUMMARY_JSON = M1_PACKAGE / "m1_completion_summary.json"
M1_JOB_CSV = M1_PACKAGE / "job_summary.csv"
M1_RECORDS_JSONL = M1_PACKAGE / "candidate_records_canonical.jsonl"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def set_style() -> None:
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = next((font for font in preferred_fonts if font in available_fonts), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": font_family,
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d7dee8",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )


def load_frozen_rows() -> tuple[dict[str, object], list[dict[str, object]]]:
    frozen = json.loads(FROZEN_JSON.read_text(encoding="utf-8"))
    rows = []
    for role, payload in frozen["roles"].items():
        rows.append(
            {
                "role": role,
                "arch_id": payload["arch_id"],
                "macro_f1_proxy": float(payload["macro_f1"]),
                "top1_proxy": float(payload["top1"]),
                "latency_ms_proxy": float(payload["latency_ms"]),
                "energy_mj_proxy": float(payload["energy_mj"]),
                "lut_proxy": float(payload["lut"]),
                "dsp_proxy": float(payload["dsp"]),
                "bram_proxy": float(payload["bram"]),
                "power_w_proxy": float(payload["power_w_proxy"]),
            }
        )
    return frozen, rows


def merge_retrain_ptq(frozen_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    retrain_by_arch = {r["arch_id"]: r for r in read_csv(RETRAIN_CSV)}
    ptq_by_arch = {r["arch_id"]: r for r in read_csv(PTQ_CSV)}
    merged = []
    for row in frozen_rows:
        arch = str(row["arch_id"])
        retrain = retrain_by_arch[arch]
        ptq = ptq_by_arch[arch]
        merged.append(
            {
                **row,
                "unit_count": int(retrain["units"]),
                "fp32_macro_f1_mean": float(retrain["macro_f1_mean"]),
                "fp32_macro_f1_std": float(retrain["macro_f1_std"]),
                "fp32_top1_mean": float(retrain["top1_mean"]),
                "fp32_top1_std": float(retrain["top1_std"]),
                "tail_class_f1_mean": float(retrain["tail_class_f1_mean"]),
                "fishing_net_f1_mean": float(retrain["fishing_net_f1_mean"]),
                "small_propeller_f1_mean": float(retrain["small_propeller_f1_mean"]),
                "claimable": retrain["claimable"],
                "source_freeze_verified": retrain["source_freeze_verified"],
                "params": int(ptq["params"]),
                "macs": int(ptq["macs"]),
                "ptq_macro_f1_mean": float(ptq["int8_macro_f1_mean"]),
                "ptq_macro_f1_std": float(ptq["int8_macro_f1_std"]),
                "ptq_top1_mean": float(ptq["int8_top1_mean"]),
                "ptq_top1_std": float(ptq["int8_top1_std"]),
                "ptq_macro_f1_drop_mean": float(ptq["int8_macro_f1_drop_mean"]),
                "ptq_top1_drop_mean": float(ptq["int8_top1_drop_mean"]),
                "int8_parity_pass": ptq["int8_parity_pass"],
            }
        )
    return merged


def plot_candidate_quality(rows: list[dict[str, object]]) -> None:
    colors = {
        "accuracy_first": "#0072B2",
        "knee_point": "#009E73",
        "latency_energy_first": "#D55E00",
        "pareto_top2": "#56B4E9",
        "pareto_top4": "#E69F00",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    for row in rows:
        role = str(row["role"])
        ax.scatter(
            row["latency_ms_proxy"],
            row["macro_f1_proxy"],
            s=70,
            color=colors.get(role, "#333333"),
            edgecolor="black",
            linewidth=0.5,
            label=f"{role} ({row['arch_id']})",
        )
        ax.annotate(str(row["arch_id"]).replace("rl_arch_", "#"), (row["latency_ms_proxy"], row["macro_f1_proxy"]), xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Search proxy latency (ms)")
    ax.set_ylabel("Search proxy Macro-F1")
    ax.set_title("A. Frozen candidates: accuracy-latency")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    sorted_rows = sorted(rows, key=lambda r: float(r["energy_mj_proxy"]))
    labels = [str(r["arch_id"]).replace("rl_arch_", "#") for r in sorted_rows]
    x = np.arange(len(sorted_rows))
    ax.bar(x - 0.18, [r["energy_mj_proxy"] for r in sorted_rows], width=0.36, label="Energy proxy (mJ)", color="#009E73")
    ax2 = ax.twinx()
    ax2.plot(x + 0.18, [r["lut_proxy"] / 1000 for r in sorted_rows], marker="o", color="#D55E00", label="LUT proxy (k)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy proxy (mJ)")
    ax2.set_ylabel("LUT proxy (k)")
    ax.set_title("B. Cost profile of selected candidates")
    handles, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels1 + labels2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig01_rl_candidate_quality.png")
    fig.savefig(OUT / "fig01_rl_candidate_quality.pdf")
    plt.close(fig)


def plot_ptq_retrain(rows: list[dict[str, object]]) -> None:
    rows = sorted(rows, key=lambda r: float(r["fp32_macro_f1_mean"]), reverse=True)
    labels = [str(r["arch_id"]).replace("rl_arch_", "#") for r in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    ax.bar(x - width / 2, [r["fp32_macro_f1_mean"] for r in rows], width, yerr=[r["fp32_macro_f1_std"] for r in rows], capsize=3, label="FP32", color="#0072B2")
    ax.bar(x + width / 2, [r["ptq_macro_f1_mean"] for r in rows], width, yerr=[r["ptq_macro_f1_std"] for r in rows], capsize=3, label="PTQ fake-quant", color="#E69F00")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.75, 0.94)
    ax.set_ylabel("Macro-F1 (mean +/- std, n=15)")
    ax.set_title("A. Frozen-candidate classification")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.axhline(0, color="#333333", linewidth=0.8)
    bars = ax.bar(x, [r["ptq_macro_f1_drop_mean"] for r in rows], color="#009E73")
    for bar, row in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{row['ptq_macro_f1_drop_mean']:+.4f}", ha="center", va="bottom" if row["ptq_macro_f1_drop_mean"] >= 0 else "top", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("FP32 - PTQ Macro-F1")
    ax.set_title("B. PTQ change")
    fig.tight_layout()
    fig.savefig(OUT / "fig02_retrain_ptq_macro_f1.png")
    fig.savefig(OUT / "fig02_retrain_ptq_macro_f1.pdf")
    plt.close(fig)


def plot_per_class_rows(rows: list[dict[str, object]]) -> None:
    rows = sorted(rows, key=lambda r: float(r["fp32_macro_f1_mean"]), reverse=True)
    labels = [str(r["arch_id"]).replace("rl_arch_", "#") for r in rows]
    metrics = [
        ("tail_class_f1_mean", "Tail class"),
        ("fishing_net_f1_mean", "Fishing net"),
        ("small_propeller_f1_mean", "Small propeller"),
    ]
    data = np.array([[float(r[key]) for key, _ in metrics] for r in rows])
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    im = ax.imshow(data, cmap="viridis", vmin=0.25, vmax=0.9, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([label for _, label in metrics], rotation=15, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", color="white" if data[i, j] < 0.55 else "black", fontsize=8)
    ax.set_title("Per-class weak spots among frozen candidates")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("F1")
    fig.tight_layout()
    fig.savefig(OUT / "fig03_per_class_soft_weakspots.png")
    fig.savefig(OUT / "fig03_per_class_soft_weakspots.pdf")
    plt.close(fig)


def plot_sonar_quality() -> None:
    rows = [r for r in read_csv(QUALITY_CSV) if r["数据集"] == "NKSID"]
    denoise_rows = [r for r in rows if r["图像处理"] == "去噪"]
    baseline_rows = [r for r in read_csv(BASELINE_DENOISE_CSV) if r["数据集"] == "NKSID"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    ax = axes[0]
    labels = [r["测试条件"] for r in denoise_rows]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, [float(r["Macro-F1 %"]) for r in denoise_rows], width=0.36, color="#0072B2", label="Macro-F1 (%)")
    ax.bar(x + 0.18, [float(r["Top-1 %"]) for r in denoise_rows], width=0.36, color="#E69F00", label="Top-1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(55, 85)
    ax.set_ylabel("Classification metric (%)")
    ax.set_title("A. Denoise network under clean/noisy inputs")
    ax.legend(frameon=False)

    ax = axes[1]
    labels = [r["条件"] for r in baseline_rows]
    x = np.arange(len(labels))
    macro_delta = [float(r["Macro-F1变化"]) for r in baseline_rows]
    top1_delta = [float(r["Top-1变化"]) for r in baseline_rows]
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.bar(x - 0.18, macro_delta, width=0.36, color="#009E73", label="Macro-F1 delta")
    ax.bar(x + 0.18, top1_delta, width=0.36, color="#56B4E9", label="Top-1 delta")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Denoise - baseline")
    ax.set_title("B. Denoise gain on NKSID")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig04_sonar_quality_and_robustness.png")
    fig.savefig(OUT / "fig04_sonar_quality_and_robustness.pdf")
    plt.close(fig)


def plot_slide_overview(frozen: dict[str, object], rows: list[dict[str, object]]) -> None:
    role_alias = {
        "accuracy_first": "Acc-first",
        "knee_point": "Knee",
        "latency_energy_first": "Lat/E",
        "pareto_top2": "Pareto-2",
        "pareto_top4": "Pareto-4",
    }
    rows = sorted(rows, key=lambda r: float(r["fp32_macro_f1_mean"]), reverse=True)
    fig = plt.figure(figsize=(13.33, 7.5))
    gs = fig.add_gridspec(12, 18, left=0.04, right=0.98, top=0.90, bottom=0.08, wspace=1.55, hspace=1.35)
    fig.suptitle("ZCU104 software-side results: RL search, retraining, and PTQ", fontsize=18, fontweight="bold", color="#003b8e", x=0.04, ha="left")

    ax_table1 = fig.add_subplot(gs[0:5, 0:8])
    ax_table1.axis("off")
    ax_table1.set_title("Search-stage frozen candidates", loc="left", fontsize=12, fontweight="bold", pad=8)
    search_table = [
        [
            role_alias.get(str(r["role"]), str(r["role"])),
            str(r["arch_id"]).replace("rl_arch_", "#"),
            f"{r['macro_f1_proxy']:.3f}",
            f"{r['top1_proxy']:.3f}",
            f"{r['latency_ms_proxy']:.1f}",
            f"{int(r['lut_proxy'])}",
            f"{int(r['dsp_proxy'])}",
            f"{int(r['bram_proxy'])}",
        ]
        for r in rows
    ]
    table = ax_table1.table(
        cellText=search_table,
        colLabels=["Role", "Arch", "M-F1", "Top-1", "Lat(ms)", "LUT", "DSP", "BRAM"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(0.96, 1.52)
    for (r, _), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#0b73b9")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_edgecolor("#6aa6d7")
            if search_table[r - 1][0] == "Acc-first":
                cell.set_facecolor("#fff7b3")

    ax_table2 = fig.add_subplot(gs[5:9, 0:8])
    ax_table2.axis("off")
    ax_table2.set_title("15-unit retrain/PTQ summary", loc="left", fontsize=12, fontweight="bold", pad=8)
    ptq_table = [
        [
            role_alias.get(str(r["role"]), str(r["role"])),
            str(r["arch_id"]).replace("rl_arch_", "#"),
            f"{r['fp32_macro_f1_mean']:.3f}",
            f"{r['ptq_macro_f1_mean']:.3f}",
            f"{r['fp32_top1_mean']:.3f}",
            f"{r['ptq_top1_mean']:.3f}",
            f"{r['ptq_macro_f1_drop_mean']:+.4f}",
        ]
        for r in rows
    ]
    table = ax_table2.table(
        cellText=ptq_table,
        colLabels=["Role", "Arch", "FP32 M-F1", "PTQ M-F1", "FP32 Top-1", "PTQ Top-1", "M-F1 drop"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.9)
    table.scale(0.96, 1.46)
    for (r, _), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#0b73b9")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_edgecolor("#6aa6d7")
            if ptq_table[r - 1][0] == "Acc-first":
                cell.set_facecolor("#fff7b3")

    ax_note = fig.add_subplot(gs[9:12, 0:8])
    ax_note.axis("off")
    note = (
        f"RL run: 200 evaluations, 180 records, Pareto size 26, HV={float(frozen['search_protocol']['hypervolume']):.6f}\n"
        "Claim boundary: search/classification/PTQ evidence only.\n"
        "Next gates: integer INT8 parity, candidate HLS, board runtime, and power."
    )
    ax_note.text(
        0,
        0.65,
        note,
        fontsize=9.4,
        color="#003b8e",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#eaf3ff", edgecolor="#5c9bd5", linewidth=1.0),
        va="center",
    )

    ax = fig.add_subplot(gs[0:5, 9:13])
    color_map = ["#0072B2", "#E69F00", "#009E73", "#56B4E9", "#D55E00"]
    ax.scatter([r["latency_ms_proxy"] for r in rows], [r["macro_f1_proxy"] for r in rows], s=65, color=color_map, edgecolor="black", linewidth=0.4)
    for r in rows:
        ax.annotate(str(r["arch_id"]).replace("rl_arch_", "#"), (r["latency_ms_proxy"], r["macro_f1_proxy"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_title("A. Macro-F1 vs latency")
    ax.set_xlabel("Latency proxy (ms)")
    ax.set_ylabel("M-F1 proxy", labelpad=2)

    ax = fig.add_subplot(gs[0:5, 14:18])
    labels = [str(r["arch_id"]).replace("rl_arch_", "#") for r in rows]
    x = np.arange(len(rows))
    ax.bar(x, [r["ptq_macro_f1_drop_mean"] for r in rows], color="#009E73")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("B. PTQ Macro-F1 drop")
    ax.set_ylabel("FP32 - PTQ")

    ax = fig.add_subplot(gs[6:12, 9:18])
    width = 0.36
    ax.bar(x - width / 2, [r["fp32_macro_f1_mean"] for r in rows], width, yerr=[r["fp32_macro_f1_std"] for r in rows], capsize=2.5, color="#0072B2", label="FP32")
    ax.bar(x + width / 2, [r["ptq_macro_f1_mean"] for r in rows], width, yerr=[r["ptq_macro_f1_std"] for r in rows], capsize=2.5, color="#E69F00", label="PTQ fake-quant")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.75, 0.94)
    ax.set_title("C. 15-unit classification after retraining")
    ax.set_ylabel("Macro-F1", labelpad=2)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    fig.savefig(OUT / "slide01_soft_results_overview.png")
    fig.savefig(OUT / "slide01_soft_results_overview.pdf")
    plt.close(fig)


def summarize_m1_scope(scope: str, jobs: list[dict[str, str]], records: list[dict[str, object]]) -> dict[str, object]:
    wall_sec = sum(float(job["wall_time_seconds_sum"]) for job in jobs)
    gpu_hours = sum(float(job["gpu_hours_sum"]) for job in jobs)
    starts = [datetime.fromisoformat(job["started_at_utc"].replace("Z", "+00:00")) for job in jobs if job.get("started_at_utc")]
    ends = [datetime.fromisoformat(job["ended_at_utc"].replace("Z", "+00:00")) for job in jobs if job.get("ended_at_utc")]
    calendar_elapsed_sec = (max(ends) - min(starts)).total_seconds() if starts and ends else None
    evaluated = len(records)
    feasible = sum(1 for row in records if bool(row.get("hardware_feasible")))
    best = max(records, key=lambda row: float(row["macro_f1"])) if records else {}
    first_95_values = [int(job["first_candidate_index_reaching_95pct_final_best"]) for job in jobs if job.get("first_candidate_index_reaching_95pct_final_best") not in ("", None)]
    return {
        "scope": scope,
        "source": "M1 completion package",
        "search_time_seconds": f"{wall_sec:.3f}",
        "search_time_minutes": f"{wall_sec / 60:.3f}",
        "calendar_elapsed_minutes": f"{calendar_elapsed_sec / 60:.3f}" if calendar_elapsed_sec is not None else "MISSING",
        "gpu_hours": f"{gpu_hours:.6f}",
        "evaluated_architectures": evaluated,
        "hardware_feasible_count": feasible,
        "hardware_feasible_rate": f"{(feasible / evaluated) if evaluated else 0:.6f}",
        "best_so_far_macro_f1_final": f"{float(best.get('macro_f1', 0)):.6f}" if best else "MISSING",
        "best_top1": f"{float(best.get('top1', 0)):.6f}" if best else "MISSING",
        "hypervolume": "MISSING",
        "non_dominated_solutions": "MISSING",
        "sampling_efficiency": f"median first index to 95pct final best = {float(np.median(first_95_values)):.1f}" if first_95_values else "MISSING",
        "claim_boundary": "provisional_proxy_only_not_route_not_board_not_power",
    }


def build_search_quality_summary(frozen: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "scope": "RL200 freeze run, fold0 seed42",
            "source": "frozen candidate summary",
            "search_time_seconds": "MISSING_LOCAL_ARTIFACT",
            "search_time_minutes": "MISSING_LOCAL_ARTIFACT",
            "calendar_elapsed_minutes": "MISSING_LOCAL_ARTIFACT",
            "gpu_hours": "MISSING_LOCAL_ARTIFACT",
            "evaluated_architectures": frozen["search_protocol"]["candidate_evaluations"],
            "hardware_feasible_count": "MISSING_LOCAL_ARTIFACT",
            "hardware_feasible_rate": "MISSING_LOCAL_ARTIFACT",
            "best_so_far_macro_f1_final": "MISSING_LOCAL_ARTIFACT",
            "best_top1": "MISSING_LOCAL_ARTIFACT",
            "hypervolume": f"{float(frozen['search_protocol']['hypervolume']):.9f}",
            "non_dominated_solutions": frozen["search_protocol"]["pareto_front_size"],
            "sampling_efficiency": "NOT_DEFINED",
            "claim_boundary": "RL search proxy summary; referenced curve/metrics files are not present locally",
        }
    ]

    if M1_JOB_CSV.exists() and M1_RECORDS_JSONL.exists():
        jobs = read_csv(M1_JOB_CSV)
        records = read_jsonl(M1_RECORDS_JSONL)
        rows.append(summarize_m1_scope("M1 all datasets/spaces", jobs, records))
        for dataset in sorted({job["dataset_display"] for job in jobs}):
            dataset_jobs = [job for job in jobs if job["dataset_display"] == dataset]
            dataset_records = [row for row in records if row.get("dataset_display") == dataset]
            rows.append(summarize_m1_scope(f"M1 {dataset}", dataset_jobs, dataset_records))
    return rows


def plot_m1_search_quality(search_rows: list[dict[str, object]]) -> None:
    if not M1_JOB_CSV.exists() or not M1_RECORDS_JSONL.exists():
        return
    jobs = read_csv(M1_JOB_CSV)
    records = read_jsonl(M1_RECORDS_JSONL)
    nksid_records = [row for row in records if row.get("dataset_display") == "NKSID"]
    nksid_jobs = [job for job in jobs if job["dataset_display"] == "NKSID"]

    fig = plt.figure(figsize=(10.5, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], hspace=0.42, wspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    for space in sorted({str(row["search_space"]) for row in nksid_records}):
        subset = sorted([row for row in nksid_records if row["search_space"] == space], key=lambda row: int(row["candidate_index"]))
        ax.plot(
            [int(row["candidate_index"]) + 1 for row in subset],
            [float(row["best_so_far_macro_f1"]) for row in subset],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=space,
        )
    ax.set_title("A. NKSID best-so-far Macro-F1")
    ax.set_xlabel("Evaluated architectures")
    ax.set_ylabel("Best-so-far Macro-F1")
    ax.set_ylim(0.88, 0.975)
    ax.legend(title="Search space", frameon=False, ncol=2)

    ax = fig.add_subplot(gs[0, 1])
    labels = [job["search_space"] for job in nksid_jobs]
    x = np.arange(len(labels))
    ax.bar(x, [float(job["wall_time_seconds_sum"]) / 60 for job in nksid_jobs], color="#0072B2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Minutes")
    ax.set_title("B. Search time per NKSID cell")

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(x, [float(job["gpu_hours_sum"]) for job in nksid_jobs], color="#E69F00")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("GPU-hours")
    ax.set_title("C. GPU cost per NKSID cell")

    fig.tight_layout()
    fig.savefig(OUT / "fig05_m1_search_quality.png")
    fig.savefig(OUT / "fig05_m1_search_quality.pdf")
    plt.close(fig)


def write_search_quality_report(search_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Search Quality Metric Availability",
        "",
        "This file separates the RL200 freeze summary from the M1 completion package. The M1 hardware fields are provisional proxy evidence only; they are not route, board-runtime, or measured-power results.",
        "",
        "| Scope | Search Time | GPU-hours | Evaluated | Feasible Rate | Best-so-far Macro-F1 | HV | Non-dominated | Sampling Efficiency |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in search_rows:
        lines.append(
            "| {scope} | {search_time_minutes} min | {gpu_hours} | {evaluated_architectures} | {hardware_feasible_rate} | {best_so_far_macro_f1_final} | {hypervolume} | {non_dominated_solutions} | {sampling_efficiency} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Reporting Guidance",
            "",
            "- `Search Time` in the table is summed per-cell wall time from `wall_time_seconds_sum`. When jobs run in parallel, use `calendar_elapsed_minutes` from `search_quality_summary.csv` for actual elapsed clock time.",
            "- For the current RL200 frozen-candidate run, report evaluated architectures, HV, and non-dominated solutions as available. Do not report search time, GPU-hours, hardware-feasible rate, or best-so-far curve until the referenced local search artifacts are synced.",
            "- For M1, report search time, GPU-hours, evaluated architectures, best-so-far curves, and provisional hardware-feasible rate. The feasible rate is 0.0 under the recorded provisional proxy boundary.",
            "- Sampling efficiency should remain auxiliary unless a formula is preregistered. The current table uses a descriptive proxy only: median first candidate index reaching 95% of final best.",
        ]
    )
    (OUT / "search_quality_metric_availability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(frozen: dict[str, object], rows: list[dict[str, object]]) -> None:
    best = max(rows, key=lambda r: float(r["fp32_macro_f1_mean"]))
    fastest = min(rows, key=lambda r: float(r["latency_ms_proxy"]))
    report = f"""# ZCU104 Weekly Soft Results Visualization Pack

Generated from local evidence on 2026-09-08. Attached PPT screenshots were used only as visual/layout references; their slide text was not treated as an instruction.

## Available Soft Results

- RL search proxy: method `{frozen["method"]}`, fold `{frozen["search_protocol"]["fold"]}`, seed `{frozen["search_protocol"]["seed"]}`, 200 candidate evaluations, 180 candidate records, Pareto front size 26, hypervolume {float(frozen["search_protocol"]["hypervolume"]):.9f}.
- Frozen candidate roles: accuracy_first, knee_point, latency_energy_first, pareto_top2, pareto_top4.
- Classification retraining/evaluation: 5 folds x 3 seeds = 15 units for each frozen candidate.
- PTQ: FP32 and post-training fake-quant metrics are available, but integer INT8 parity is still fail-closed for every candidate.
- Sonar image-quality/robustness: existing NKSID denoise/edge soft-result summaries are available and should be reported as a separate older soft-evidence layer.

## Main Numbers For Slide Tables

- Best classification candidate: `{best["role"]}` / `{best["arch_id"]}`, FP32 Macro-F1 {best["fp32_macro_f1_mean"]:.6f} +/- {best["fp32_macro_f1_std"]:.6f}, Top-1 {best["fp32_top1_mean"]:.6f}; PTQ Macro-F1 {best["ptq_macro_f1_mean"]:.6f}, PTQ Top-1 {best["ptq_top1_mean"]:.6f}.
- Fastest search-proxy candidate among frozen roles: `{fastest["role"]}` / `{fastest["arch_id"]}`, latency proxy {fastest["latency_ms_proxy"]:.6f} ms, energy proxy {fastest["energy_mj_proxy"]:.6f} mJ.
- Do not label these as ZCU104 board runtime or board power. The soft pack is search/classification/PTQ/image-quality evidence only.

## Generated Figures

- `fig01_rl_candidate_quality.png`: frozen candidate Macro-F1 vs latency and proxy cost profile.
- `fig02_retrain_ptq_macro_f1.png`: FP32 vs PTQ Macro-F1, with mean +/- std over 15 units.
- `fig03_per_class_soft_weakspots.png`: per-class weak spots for tail, fishing-net, and small-propeller classes.
- `fig04_sonar_quality_and_robustness.png`: NKSID denoise classification and strong-noise gain, matching the PPT-style left-panel story.
- `slide01_soft_results_overview.png`: one-slide overview that follows the attached PPT's table-plus-chart layout.
- `fig05_m1_search_quality.png`: M1 search time/GPU cost and NKSID best-so-far curves.
- `search_quality_metric_availability.md`: exact availability table for Search Quality metrics.

## Missing For Better PPT Figures

- The frozen JSON references full search artifacts (`rl_search_curve.csv`, `pareto_selection.json`, `rl_search_metrics.json`), but they are not present at the referenced local path. Once synced, draw the PPT-style three-panel figure: Macro-F1 vs latency, Top-1 vs latency, and best-so-far Macro-F1.
- Candidate HLS evidence for the MobileNetA frozen shortlist is incomplete in the current dry-run summary, so do not draw HLS resource bars as final results for these five candidates yet.
"""
    (OUT / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    set_style()
    frozen, frozen_rows = load_frozen_rows()
    rows = merge_retrain_ptq(frozen_rows)
    search_rows = build_search_quality_summary(frozen)
    fieldnames = list(rows[0].keys())
    write_csv(OUT / "soft_results_summary.csv", rows, fieldnames)
    search_fieldnames = list(search_rows[0].keys())
    write_csv(OUT / "search_quality_summary.csv", search_rows, search_fieldnames)
    plot_candidate_quality(rows)
    plot_ptq_retrain(rows)
    plot_per_class_rows(rows)
    plot_sonar_quality()
    plot_m1_search_quality(search_rows)
    write_search_quality_report(search_rows)
    plot_slide_overview(frozen, rows)
    write_report(frozen, rows)


if __name__ == "__main__":
    main()
