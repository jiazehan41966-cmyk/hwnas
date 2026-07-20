#!/usr/bin/env python3
"""Build benchmark tables/figures from archived source data, fail closed on gaps."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwnas_fpga.benchmarks.archive import (
    CampaignPaths,
    FIGURE_TITLES,
    TABLE_TITLES,
    sha256_file,
    write_figure_metadata,
    write_json,
    write_table_bundle,
)


def _configure_plot_fonts() -> None:
    import matplotlib
    from matplotlib import font_manager

    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
        matplotlib.rcParams["font.family"] = family
    matplotlib.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default="ccf_ab_nksid_av7k325_v1")
    parser.add_argument("--source-audit", default=None)
    parser.add_argument("--search-comparison", default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _source_audit_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("papers") or payload.get("records") or []
    if not rows and isinstance(payload.get("audits"), list):
        rows = []
        for item in payload["audits"]:
            paper = item["paper"]
            rows.append(
                {
                    "paper_id": paper["paper_id"],
                    "registry_role": paper["registry_role"],
                    "direction": paper["direction"],
                    "venue": paper["venue"],
                    "class": paper["comparability_class"],
                    "commit": item.get("observed_commit") or "MISSING",
                    "pin_match": item["commit_matches_pin"],
                    "license": paper.get("license_spdx") or "UNVERIFIED",
                    "paper_code_correspondence": paper.get("paper_code_correspondence"),
                    "source_shape": item["source_shape"],
                    "source_audit_pass": item.get("source_audit_pass", False),
                    "formal_eligible": item["formal_eligible"],
                    "blockers": ";".join(item["blockers"]),
                }
            )
    if not isinstance(rows, list):
        raise ValueError("source audit must contain a paper/record list")
    return [dict(row) for row in rows]


def _write_f2(paths: CampaignPaths, comparison_path: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    _configure_plot_fonts()

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    exact = payload.get("exact_hypervolume") or {}
    if exact.get("available") is not True:
        raise ValueError("search comparison does not contain formal exact HV curves")
    curves = exact.get("curves") or {}
    if not curves:
        raise ValueError("exact HV curve set is empty")

    source_path = paths.artifact_root / "figures" / "f2_source.csv"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_label, curve in curves.items():
        for point in curve:
            rows.append({"run_label": run_label, **dict(point)})
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for run_label, curve in curves.items():
        axis.step(
            [int(point["evaluation"]) for point in curve],
            [float(point["exact_normalized_hypervolume"]) for point in curve],
            where="post",
            label=run_label,
        )
    axis.set_xlabel("Evaluator calls / 评估器调用次数")
    axis.set_ylabel("Exact normalized hypervolume / 精确归一化 HV")
    axis.set_xlim(left=1)
    axis.set_ylim(bottom=0.0, top=1.0)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    png_path = paths.artifact_root / "figures" / "f2.png"
    pdf_path = paths.artifact_root / "figures" / "f2.pdf"
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)
    meta_path = write_figure_metadata(
        paths,
        "F2",
        caption=(
            "Anytime exact normalized hypervolume under the frozen three-objective "
            "definition and equal evaluator-call budget."
        ),
        supported_claim="Search-proxy Pareto quality versus evaluator calls only.",
        limitations=[
            "Does not provide retrain, HLS/route, board, or measured-power evidence.",
            "Requires the same frozen latency limit and objective definitions for every run.",
        ],
        generator_sha=sha256_file(__file__),
        input_files=[comparison_path],
        source_data_path=source_path,
    )
    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
        "source_csv": str(source_path),
        "meta_json": str(meta_path),
    }


def _write_f1(paths: CampaignPaths) -> dict[str, str]:
    """Render the frozen evidence workflow; this figure contains no result data."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    _configure_plot_fonts()

    stages = [
        ("Search proxy\n搜索代理", "PENDING", "candidate logs, exact HV"),
        ("Retrain\n冻结协议重训练", "G1 PENDING", "15 fold-seed predictions"),
        ("HLS / Route\n综合与布线", "G3 FROZEN", "resources, WNS/TNS"),
        ("COM5 board\n板级推理", "G2 PENDING", "latency, FPS, errors"),
        ("Power meter\n外部功率仪", "NOT_MEASURED", "traces, dynamic mJ/inf"),
    ]
    source_path = paths.artifact_root / "figures" / "f1_source.csv"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("order", "stage", "status", "evidence"))
        writer.writeheader()
        for index, (stage, status, evidence) in enumerate(stages, start=1):
            writer.writerow(
                {"order": index, "stage": stage.replace("\n", " / "), "status": status, "evidence": evidence}
            )

    figure, axis = plt.subplots(figsize=(12.0, 3.6), constrained_layout=True)
    axis.set_xlim(0, len(stages) * 2.2)
    axis.set_ylim(0, 3.0)
    axis.axis("off")
    colors = ["#4477AA", "#66CCEE", "#CCBB44", "#EE6677", "#AA3377"]
    for index, ((stage, status, evidence), color) in enumerate(zip(stages, colors)):
        x = index * 2.2 + 0.15
        box = FancyBboxPatch(
            (x, 1.05),
            1.75,
            1.15,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor=color,
            edgecolor="#222222",
            linewidth=1.0,
            alpha=0.88,
        )
        axis.add_patch(box)
        axis.text(x + 0.875, 1.67, stage, ha="center", va="center", fontsize=10, color="white", weight="bold")
        axis.text(x + 0.875, 0.72, status, ha="center", va="center", fontsize=9, color="#8B0000", weight="bold")
        axis.text(x + 0.875, 0.30, evidence, ha="center", va="center", fontsize=8, color="#333333")
        if index < len(stages) - 1:
            axis.add_patch(
                FancyArrowPatch(
                    (x + 1.77, 1.62),
                    (x + 2.15, 1.62),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    linewidth=1.2,
                    color="#333333",
                )
            )
    axis.text(
        0.15,
        2.72,
        "Evidence layers are reported separately; a downstream claim requires every upstream provenance check.",
        fontsize=11,
        weight="bold",
        color="#222222",
    )
    png_path = paths.artifact_root / "figures" / "f1.png"
    pdf_path = paths.artifact_root / "figures" / "f1.pdf"
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)
    campaign_config = PROJECT_ROOT / "configs" / "benchmarks" / "ccf_ab_campaign_v1.yaml"
    meta_path = write_figure_metadata(
        paths,
        "F1",
        caption="Frozen evidence layers and current conservative Gate states.",
        supported_claim="Defines the benchmark evidence workflow and claim boundaries.",
        limitations=["Conceptual workflow only; contains no experimental performance result."],
        generator_sha=sha256_file(__file__),
        input_files=[campaign_config],
        source_data_path=source_path,
    )
    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
        "source_csv": str(source_path),
        "meta_json": str(meta_path),
    }


def build_status(paths: CampaignPaths) -> dict[str, Any]:
    tables = {}
    for table_id, title in TABLE_TITLES.items():
        base = paths.artifact_root / "tables" / table_id.lower()
        files = {suffix: str(base.with_suffix(f".{suffix}")) for suffix in ("csv", "md", "tex")}
        complete = all(Path(path).exists() for path in files.values())
        tables[table_id] = {
            "title": title,
            "status": "AVAILABLE" if complete else "PENDING",
            "files": files,
        }
    figures = {}
    for figure_id, title in FIGURE_TITLES.items():
        base = paths.artifact_root / "figures" / figure_id.lower()
        files = {
            "png_300dpi": str(base.with_suffix(".png")),
            "vector_pdf": str(base.with_suffix(".pdf")),
            "source_csv": str(base.parent / f"{base.name}_source.csv"),
            "meta_json": str(base.parent / f"{base.name}_meta.json"),
        }
        complete = all(Path(path).exists() for path in files.values())
        figures[figure_id] = {
            "title": title,
            "status": "AVAILABLE" if complete else "PENDING",
            "files": files,
        }
    return {
        "schema_version": 1,
        "campaign_id": paths.campaign_id,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "tables": tables,
        "figures": figures,
        "boundary": (
            "PENDING items are not emitted as empty placeholders. AVAILABLE means every "
            "required archival format exists, not that the scientific Gate has passed."
        ),
    }


def main() -> int:
    args = parse_args()
    paths = CampaignPaths.from_repo(PROJECT_ROOT, args.campaign_id).create()
    source_audit = Path(args.source_audit).resolve() if args.source_audit else (
        paths.artifact_root / "manifests" / "source_audit.json"
    )
    if source_audit.exists():
        rows = _source_audit_rows(json.loads(source_audit.read_text(encoding="utf-8")))
        if rows:
            write_table_bundle(paths, "T1", rows)
    _write_f1(paths)
    if args.search_comparison:
        comparison_path = Path(args.search_comparison).resolve()
        payload = json.loads(comparison_path.read_text(encoding="utf-8"))
        run_rows = [dict(row) for row in payload.get("runs") or []]
        if run_rows:
            write_table_bundle(paths, "T5", run_rows)
        _write_f2(paths, comparison_path)

    status = build_status(paths)
    status_path = write_json(paths.artifact_root / "manifests" / "artifact_status.json", status)
    pending = [
        item_id
        for family in (status["tables"], status["figures"])
        for item_id, item in family.items()
        if item["status"] != "AVAILABLE"
    ]
    print(json.dumps({"status": str(status_path), "pending": pending}, ensure_ascii=False, indent=2))
    if args.strict and pending:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
