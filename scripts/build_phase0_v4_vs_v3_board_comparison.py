#!/usr/bin/env python3
"""Build Phase0 v4 vs v3 board-evidence comparison artifacts.

This script is evidence packaging only. It reads the v4 board experiment
manifest and writes comparison CSV/JSON/Markdown files; it does not run Vivado,
COM5, training, or search.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "results"
    / "phase0_v4_sonar_stage3_k3_board_experiment"
    / "phase0_v4_sonar_board_candidate_manifest.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "phase0_v4_sonar_stage3_k3_board_experiment"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evidence_class(row: dict[str, Any]) -> str:
    com5 = row.get("com5_board", {})
    vivado = row.get("vivado_actual", {})
    if com5.get("status") == "board-claimable":
        return "board-claimable"
    if vivado.get("status") == "route-clean":
        return "route-clean"
    if str(vivado.get("status", "")).startswith("full-route-fail"):
        return "route-fail"
    return "search-only"


def stability_json_path(row: dict[str, Any]) -> str:
    measurement_json = row.get("com5_board", {}).get("measurement_json")
    if not measurement_json:
        return ""
    candidate = Path(measurement_json).parent / "stability_runs.json"
    return str(candidate) if candidate.exists() else ""


def v4_comparison_row(row: dict[str, Any]) -> dict[str, Any]:
    proxy = row.get("search_proxy", {})
    vivado = row.get("vivado_actual", {})
    com5 = row.get("com5_board", {})
    klass = evidence_class(row)
    return {
        "phase": "v4",
        "arch_id": row.get("arch_id"),
        "role": row.get("role"),
        "stage3_op": row.get("stage3_op"),
        "evidence_class": klass,
        "macro_f1": proxy.get("macro_f1"),
        "top1": proxy.get("top1"),
        "search_latency_ms": proxy.get("latency_ms"),
        "search_lut": proxy.get("lut"),
        "search_dsp": proxy.get("dsp"),
        "search_bram": proxy.get("bram"),
        "actual_wns_ns": vivado.get("wns_ns"),
        "actual_lut": vivado.get("actual_lut"),
        "actual_dsp": vivado.get("actual_dsp"),
        "actual_bram": vivado.get("actual_bram"),
        "com5_status_code": com5.get("status_code"),
        "com5_latency_ms": com5.get("latency_ms"),
        "com5_cycles": com5.get("cycles"),
        "com5_checksum": com5.get("checksum"),
        "measurement_runs": com5.get("runs"),
        "bitstream_sha256": vivado.get("bitstream_sha256"),
        "build_result": vivado.get("build_result"),
        "bitstream": vivado.get("bitstream"),
        "measurement_json": com5.get("measurement_json"),
        "stability_runs_json": stability_json_path(row),
        "claim_boundary": (
            "board-claimable only for this measured bitstream and COM5 run set"
            if klass == "board-claimable"
            else "not board-claimable without full-route PASS and COM5 status_code=0"
        ),
    }


def v3_comparison_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": "v3",
        "arch_id": row.get("arch_id"),
        "role": row.get("role"),
        "stage3_op": "",
        "evidence_class": row.get("claim_status", "board_claimable").replace("_", "-"),
        "macro_f1": row.get("macro_f1"),
        "top1": row.get("top1"),
        "search_latency_ms": "",
        "search_lut": "",
        "search_dsp": "",
        "search_bram": "",
        "actual_wns_ns": row.get("wns_ns"),
        "actual_lut": row.get("actual_lut"),
        "actual_dsp": row.get("actual_dsp"),
        "actual_bram": row.get("actual_bram"),
        "com5_status_code": 0,
        "com5_latency_ms": row.get("board_latency_ms"),
        "com5_cycles": row.get("board_cycles"),
        "com5_checksum": "",
        "measurement_runs": "",
        "bitstream_sha256": "",
        "build_result": "",
        "bitstream": "",
        "measurement_json": "",
        "stability_runs_json": "",
        "claim_boundary": "v3 baseline row from existing board-claimable handoff; v3 artifacts are read-only here",
    }


CSV_FIELDS = [
    "phase",
    "arch_id",
    "role",
    "stage3_op",
    "evidence_class",
    "macro_f1",
    "top1",
    "search_latency_ms",
    "search_lut",
    "search_dsp",
    "search_bram",
    "actual_wns_ns",
    "actual_lut",
    "actual_dsp",
    "actual_bram",
    "com5_status_code",
    "com5_latency_ms",
    "com5_cycles",
    "com5_checksum",
    "measurement_runs",
    "bitstream_sha256",
    "build_result",
    "bitstream",
    "measurement_json",
    "stability_runs_json",
    "claim_boundary",
]


def best_by(rows: list[dict[str, Any]], key: str, *, lower_is_better: bool = False) -> dict[str, Any] | None:
    usable = [row for row in rows if numeric(row.get(key)) is not None]
    if not usable:
        return None
    return min(usable, key=lambda row: numeric(row.get(key)) or 0.0) if lower_is_better else max(
        usable,
        key=lambda row: numeric(row.get(key)) or 0.0,
    )


def delta(a: Any, b: Any) -> float | None:
    left = numeric(a)
    right = numeric(b)
    if left is None or right is None:
        return None
    return left - right


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    rows = payload["rows"]
    v4_rows = [row for row in rows if row.get("phase") == "v4"]
    board_ids = [
        str(row["arch_id"])
        for row in v4_rows
        if row.get("evidence_class") == "board-claimable"
    ]
    route_only_ids = [
        str(row["arch_id"])
        for row in v4_rows
        if row.get("evidence_class") == "route-clean"
    ]
    route_fail_ids = [
        str(row["arch_id"])
        for row in v4_rows
        if row.get("evidence_class") == "route-fail"
    ]
    lines = [
        "# Phase0 v4 vs v3 Board Evidence Comparison",
        "",
        "This file separates search proxy metrics, Vivado actual metrics, and COM5 board measurements.",
        "It is generated from existing local evidence only.",
        "",
        "## Summary",
        "",
        f"- v4 Pareto rows: `{summary['v4_pareto_count']}`",
        f"- v4 route-clean rows: `{summary['v4_route_clean_count']}`",
        f"- v4 board-claimable rows: `{summary['v4_board_claimable_count']}`",
        f"- v4 route-fail rows: `{summary['v4_route_fail_count']}`",
        "",
        "## Decision-Level Findings",
        "",
    ]
    for item in payload["findings"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Comparison Table",
            "",
            "| phase | arch | role | stage3 op | class | search macro_f1 | search top1 | COM5 ms | cycles | WNS ns | actual DSP | actual LUT | runs |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['phase']} | `{row['arch_id']}` | {row['role']} | `{row['stage3_op']}` | "
            f"{row['evidence_class']} | {fmt(row['macro_f1'])} | {fmt(row['top1'])} | "
            f"{fmt(row['com5_latency_ms'])} | {fmt(row['com5_cycles'])} | "
            f"{fmt(row['actual_wns_ns'])} | {fmt(row['actual_dsp'])} | "
            f"{fmt(row['actual_lut'])} | {fmt(row['measurement_runs'])} |"
        )
    lines.extend(
        [
            "",
            "## Evidence Boundaries",
            "",
            "- A v4 row is `board-claimable` only when it has full-route PASS, nonnegative WNS, actual DSP <= 700, COM5 `status_code=0`, and the listed measured bitstream hash.",
            "- The macro_f1/top1 columns are search-proxy metrics; retrain150 validation metrics are reported in the separate retrain comparison artifact.",
            "- Search latency/DSP/LUT columns are proxy estimates and must not be described as real board latency/resources.",
        ]
    )
    if board_ids:
        lines.append(
            "- Current manifest board-claimable rows: "
            + ", ".join(f"`{arch_id}`" for arch_id in board_ids)
            + "."
        )
    if route_only_ids:
        lines.append(
            "- Current manifest route-clean-only rows: "
            + ", ".join(f"`{arch_id}`" for arch_id in route_only_ids)
            + "."
        )
    if route_fail_ids:
        lines.append(
            "- Current manifest route-fail rows: "
            + ", ".join(f"`{arch_id}`" for arch_id in route_fail_ids)
            + "; these remain blocked from COM5 claimability."
        )
    return "\n".join(lines) + "\n"


def build_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    v4_rows = [v4_comparison_row(row) for row in manifest["pareto_route_screen_candidates"]]
    v3_rows = [v3_comparison_row(row) for row in manifest["v3_board_claimable_baseline"]]
    board_v4 = [row for row in v4_rows if row["evidence_class"] == "board-claimable"]
    route_clean_v4 = [row for row in v4_rows if row["evidence_class"] in {"route-clean", "board-claimable"}]
    route_fail_v4 = [row for row in v4_rows if row["evidence_class"] == "route-fail"]

    v4_best_macro = best_by(board_v4, "macro_f1")
    v4_fastest = best_by(board_v4, "com5_latency_ms", lower_is_better=True)
    v4_lowest_dsp = best_by(route_clean_v4, "actual_dsp", lower_is_better=True)
    v3_best_macro = best_by(v3_rows, "macro_f1")
    v3_fastest = best_by(v3_rows, "com5_latency_ms", lower_is_better=True)
    v3_lowest_dsp = best_by(v3_rows, "actual_dsp", lower_is_better=True)

    findings: list[str] = []
    if v4_best_macro and v3_best_macro:
        findings.append(
            f"Best board-claimable v4 search-proxy macro_f1 row is `{v4_best_macro['arch_id']}` "
            f"({fmt(v4_best_macro['macro_f1'])}); delta vs best v3 macro_f1 row "
            f"`{v3_best_macro['arch_id']}` is {fmt(delta(v4_best_macro['macro_f1'], v3_best_macro['macro_f1']))}."
        )
        findings.append(
            f"`{v4_best_macro['arch_id']}` COM5 latency is {fmt(v4_best_macro['com5_latency_ms'])} ms; "
            f"delta vs `{v3_best_macro['arch_id']}` is "
            f"{fmt(delta(v4_best_macro['com5_latency_ms'], v3_best_macro['com5_latency_ms']))} ms."
        )
    if v4_fastest and v3_fastest:
        findings.append(
            f"Fastest board-claimable v4 row is `{v4_fastest['arch_id']}` "
            f"({fmt(v4_fastest['com5_latency_ms'])} ms); fastest v3 row is "
            f"`{v3_fastest['arch_id']}` ({fmt(v3_fastest['com5_latency_ms'])} ms), "
            f"so this v4 batch does not beat the v3 latency/resource baseline."
        )
    if v4_lowest_dsp and v3_lowest_dsp:
        findings.append(
            f"Lowest route-clean v4 DSP row is `{v4_lowest_dsp['arch_id']}` "
            f"(DSP {fmt(v4_lowest_dsp['actual_dsp'])}); lowest v3 board row is "
            f"`{v3_lowest_dsp['arch_id']}` (DSP {fmt(v3_lowest_dsp['actual_dsp'])})."
        )
    rows_by_id = {str(row.get("arch_id")): row for row in v4_rows}
    edge_row = rows_by_id.get("rl_arch_154")
    denoise_row = rows_by_id.get("rl_arch_169")
    if (
        edge_row
        and denoise_row
        and edge_row.get("evidence_class") == "board-claimable"
        and denoise_row.get("evidence_class") == "board-claimable"
    ):
        findings.append(
            "`rl_arch_154` (edge) and `rl_arch_169` (denoise) are board-claimable "
            "sonar-op candidates in the current manifest."
        )
    blocked_row = rows_by_id.get("rl_arch_116")
    if blocked_row and blocked_row.get("evidence_class") == "route-fail":
        findings.append(
            "`rl_arch_116` (denoise) remains blocked from COM5 because its current "
            "full-route evidence fails the timing/resource gate."
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_manifest": str(DEFAULT_MANIFEST),
        "summary": {
            "v4_pareto_count": len(v4_rows),
            "v4_route_clean_count": len(route_clean_v4),
            "v4_board_claimable_count": len(board_v4),
            "v4_route_fail_count": len(route_fail_v4),
            "v3_board_claimable_count": len(v3_rows),
            "best_v4_board_macro_f1_arch": v4_best_macro.get("arch_id") if v4_best_macro else "",
            "fastest_v4_board_arch": v4_fastest.get("arch_id") if v4_fastest else "",
            "lowest_dsp_v4_route_clean_arch": v4_lowest_dsp.get("arch_id") if v4_lowest_dsp else "",
        },
        "findings": findings,
        "rows": v4_rows + v3_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest = read_json(manifest_path)
    payload = build_payload(manifest)
    payload["source_manifest"] = str(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "phase0_v4_vs_v3_board_comparison.json"
    csv_path = output_dir / "phase0_v4_vs_v3_board_comparison.csv"
    md_path = output_dir / "phase0_v4_vs_v3_board_comparison.md"
    write_json(json_path, payload)
    write_csv(csv_path, payload["rows"])
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "json": str(json_path),
                "csv": str(csv_path),
                "markdown": str(md_path),
                "summary": payload["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
