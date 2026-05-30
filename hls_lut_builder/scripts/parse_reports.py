#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
BUILDER_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BUILDER_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common import build_cases, find_first_existing_report, load_config, resolve_workspace_path, select_pilot_cases
from hwnas_fpga.hardware import (
    parse_hls_report,
    parse_vivado_power_text,
    parse_vivado_timing_summary_text,
    parse_vivado_utilization_text,
)


CSV_FIELDS = [
    "op_id",
    "case_name",
    "op_type",
    "H",
    "W",
    "Cin",
    "Cout",
    "K",
    "stride",
    "groups",
    "expand",
    "bitwidth",
    "PI",
    "PO",
    "unroll",
    "pack_ch",
    "ch_block",
    "target_ii",
    "tile_order",
    "stream_order",
    "dsp_pack",
    "implementation_name",
    "target_part",
    "target_clock_ns",
    "target_clock_mhz",
    "deployable_at_200mhz",
    "estimated_clock_period_ns",
    "Fmax_est",
    "latency_cycles",
    "latency_ns_est",
    "latency_ms",
    "II",
    "depth",
    "LUT",
    "FF",
    "BRAM_18K",
    "BRAM_36K",
    "DSP",
    "report_format",
    "report_path",
    "power_w",
    "power_source",
    "synthesis_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse Vitis HLS reports into CSV and LUT manifest")
    parser.add_argument("--config", required=True, help="Path to builder config YAML")
    parser.add_argument("--project-root", default=None, help="Optional override for generated project root")
    parser.add_argument("--operator", action="append", default=None, help="Parse only specific operators")
    parser.add_argument("--case", action="append", default=None, help="Parse only specific cases")
    parser.add_argument("--pilot", action="store_true", help="Parse one representative case per enabled operator")
    parser.add_argument("--csv", default=None, help="Optional CSV output override")
    parser.add_argument("--manifest", default=None, help="Optional manifest output override")
    parser.add_argument("--archive-dir", default=None, help="Optional report archive override")
    parser.add_argument("--summary-json", default=None, help="Optional parse summary JSON override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    cases = build_cases(
        config,
        operator_filter=args.operator,
        case_filter=args.case,
        project_root_override=args.project_root,
        sort_cases=not args.pilot,
    )
    if args.pilot:
        cases = select_pilot_cases(cases)
    if not cases:
        raise SystemExit("No HLS cases matched the provided filters.")

    workspace = config.get("workspace", {})
    measurement_contract = _load_measurement_contract(config)
    csv_path = resolve_workspace_path(config, args.csv or workspace["csv_path"])
    manifest_path = resolve_workspace_path(config, args.manifest or workspace["manifest_path"])
    archive_dir = resolve_workspace_path(config, args.archive_dir or workspace["report_archive"])
    archive_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    missing_reports: list[str] = []

    for case in cases:
        report_path = find_first_existing_report(case.report_candidates)
        status_payload = _load_status_payload(case.project_dir / "synthesis_status.json")
        synthesis_status = str(status_payload.get("status", "missing_report" if report_path is None else "parsed"))

        if report_path is None:
            missing_reports.append(case.case_name)
            continue

        metrics = parse_hls_report(report_path)
        archived_path = archive_dir / f"{case.case_name}{report_path.suffix.lower()}"
        shutil.copy2(report_path, archived_path)

        height, width = case.op_spec.get("input_resolution", [0, 0])
        cycles = int(metrics.get("cycles") or 0)
        latency_ns = metrics.get("latency_ns")
        if latency_ns is None and cycles > 0:
            latency_ns = cycles * float(case.clock_period_ns)
        latency_ms = float(latency_ns) / 1_000_000.0 if latency_ns is not None else 0.0

        vivado_actual = _load_vivado_actual(case)
        post_route_metrics = _post_route_metrics(vivado_actual)
        power_w = post_route_metrics.get("power_w")
        power_source = post_route_metrics.get("power_source")

        row = {
            "op_id": case.op_id,
            "case_name": case.case_name,
            "op_type": case.op_spec.get("op", case.operator),
            "H": int(height),
            "W": int(width),
            "Cin": int(case.op_spec.get("in_channels", 0)),
            "Cout": int(case.op_spec.get("out_channels", 0)),
            "K": int(case.op_spec.get("kernel_size", 1)),
            "stride": int(case.op_spec.get("stride", 1)),
            "groups": int(case.op_spec.get("groups", 1)),
            "expand": int(case.op_spec.get("expand_ratio", 1)),
            "bitwidth": int(case.op_spec.get("bitwidth", 8)),
            "PI": int(case.op_spec.get("input_parallelism", 1)),
            "PO": int(case.op_spec.get("output_parallelism", 1)),
            "unroll": int(case.op_spec.get("unroll_factor", 1)),
            "pack_ch": case.op_spec.get("pack_ch", ""),
            "ch_block": case.op_spec.get("ch_block", ""),
            "target_ii": case.op_spec.get("target_ii", ""),
            "tile_order": case.op_spec.get("tile_order", ""),
            "stream_order": case.op_spec.get("stream_order", ""),
            "dsp_pack": case.op_spec.get("dsp_pack", ""),
            "implementation_name": case.implementation_name,
            "target_part": case.part,
            "target_clock_ns": float(case.clock_period_ns),
            "target_clock_mhz": float(case.target_clock_mhz),
            "deployable_at_200mhz": bool(float(case.target_clock_mhz) >= 199.9),
            "estimated_clock_period_ns": metrics.get("estimated_clock_period_ns"),
            "Fmax_est": metrics.get("fmax_est_mhz"),
            "latency_cycles": cycles,
            "latency_ns_est": latency_ns,
            "latency_ms": latency_ms,
            "II": int(metrics.get("ii") or 0),
            "depth": int(metrics.get("depth") or 0),
            "LUT": int(metrics.get("lut") or 0),
            "FF": int(metrics.get("ff") or 0),
            "BRAM_18K": int(metrics.get("bram_18k") or metrics.get("bram") or 0),
            "BRAM_36K": int(metrics.get("bram_36k") or 0),
            "DSP": int(metrics.get("dsp") or 0),
            "report_format": metrics.get("report_format", report_path.suffix.lower().lstrip(".")),
            "report_path": str(archived_path),
            "power_w": power_w if power_w is not None else "",
            "power_source": power_source if power_source else "missing",
            "synthesis_status": synthesis_status,
        }
        rows.append(row)

        manifest_entries.append(
            {
                "op_spec": case.op_spec,
                "clock_mhz": case.target_clock_mhz,
                "report": str(archived_path),
                "metrics": _build_preferred_metrics(
                    hls_metrics={
                        "cycles": row["latency_cycles"],
                        "latency_ns": row["latency_ns_est"],
                        "latency_ms": row["latency_ms"],
                        "ii": row["II"],
                        "depth": row["depth"],
                        "dsp": row["DSP"],
                        "bram": row["BRAM_18K"],
                        "bram_18k": row["BRAM_18K"],
                        "bram_36k": row["BRAM_36K"],
                        "lut": row["LUT"],
                        "ff": row["FF"],
                        "estimated_clock_period_ns": row["estimated_clock_period_ns"],
                        "fmax_est_mhz": row["Fmax_est"],
                    },
                    vivado_actual=vivado_actual,
                ),
                "hls_estimate": {
                    "report": str(archived_path),
                    "metrics": {
                        "cycles": row["latency_cycles"],
                        "latency_ns": row["latency_ns_est"],
                        "latency_ms": row["latency_ms"],
                        "ii": row["II"],
                        "depth": row["depth"],
                        "dsp": row["DSP"],
                        "bram": row["BRAM_18K"],
                        "bram_18k": row["BRAM_18K"],
                        "bram_36k": row["BRAM_36K"],
                        "lut": row["LUT"],
                        "ff": row["FF"],
                        "estimated_clock_period_ns": row["estimated_clock_period_ns"],
                        "fmax_est_mhz": row["Fmax_est"],
                    },
                    "status": synthesis_status,
                },
                "vivado_actual": vivado_actual,
                "metadata": {
                    "op_id": case.op_id,
                    "case_name": case.case_name,
                    "implementation_name": case.implementation_name,
                    "clock_profile_name": case.clock_profile_name,
                    "synthesis_status": synthesis_status,
                    "selected_metrics_source": "vivado_actual.post_route"
                    if vivado_actual is not None else "hls_estimate",
                    "measurement_contract_id": measurement_contract.get("contract_id"),
                    "measurement_contract_declaration": measurement_contract.get("declaration"),
                    "measurement_contract": copy.deepcopy(measurement_contract),
                },
            }
        )

    _write_csv(csv_path, rows)
    manifest_payload = {
        "clock_mhz": rows[0]["target_clock_mhz"] if rows else None,
        "measurement_contract": copy.deepcopy(measurement_contract),
        "entries": manifest_entries,
    }
    manifest_path.write_text(yaml.safe_dump(manifest_payload, sort_keys=False), encoding="utf-8")

    summary_path = resolve_workspace_path(
        config,
        args.summary_json or workspace.get("parse_summary_json", "../results/operator_screening_parse_summary.json"),
    )
    summary = {
        "parsed_cases": len(rows),
        "missing_reports": missing_reports,
        "csv_path": str(csv_path),
        "manifest_path": str(manifest_path),
        "archive_dir": str(archive_dir),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Parsed {len(rows)} HLS reports.")
    if missing_reports:
        print("Missing reports:")
        for case_name in missing_reports:
            print(case_name)
    print(f"CSV: {csv_path}")
    print(f"Manifest: {manifest_path}")


def _load_status_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_measurement_contract(config: dict[str, Any]) -> dict[str, Any]:
    contract = config.get("measurement_contract")
    if isinstance(contract, dict) and contract:
        return dict(contract)

    config_path = Path(str(config.get("_config_path", "")))
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        fallback = raw.get("measurement_contract")
        if isinstance(fallback, dict):
            return dict(fallback)
    return {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _load_vivado_actual(case: Any) -> dict[str, Any] | None:
    report_dir = case.vivado_downstream_reports_dir
    post_synth = _parse_vivado_stage_reports(report_dir, "post_synth", target_clock_ns=case.clock_period_ns)
    post_route = _parse_vivado_stage_reports(report_dir, "post_route", target_clock_ns=case.clock_period_ns)
    if post_synth is None or post_route is None:
        return None

    status_payload = _load_status_payload(case.vivado_downstream_status_path)
    power = _parse_post_route_power(report_dir)
    reports = {
        "post_synth_timing": str(post_synth["timing_report"]),
        "post_synth_utilization": str(post_synth["utilization_report"]),
        "post_route_timing": str(post_route["timing_report"]),
        "post_route_utilization": str(post_route["utilization_report"]),
    }
    post_route_metrics = {
        **post_route["timing"],
        **post_route["utilization"],
    }
    if power is not None:
        reports["post_route_power"] = str(power["power_report"])
        post_route_metrics.update(power["metrics"])
    return {
        "reports": reports,
        "metrics": {
            "post_synth": {
                **post_synth["timing"],
                **post_synth["utilization"],
            },
            "post_route": post_route_metrics,
        },
        "status": str(status_payload.get("status", "success")),
    }


def _parse_vivado_stage_reports(
    report_dir: Path,
    stage_name: str,
    *,
    target_clock_ns: float,
) -> dict[str, Any] | None:
    timing_report = report_dir / f"{stage_name}_timing_summary.rpt"
    utilization_report = report_dir / f"{stage_name}_utilization.rpt"
    if not timing_report.exists() or not utilization_report.exists():
        return None

    timing = parse_vivado_timing_summary_text(
        timing_report.read_text(encoding="utf-8", errors="ignore"),
        target_clock_ns=target_clock_ns,
    )
    utilization = parse_vivado_utilization_text(
        utilization_report.read_text(encoding="utf-8", errors="ignore"),
    )
    return {
        "timing": timing,
        "utilization": utilization,
        "timing_report": timing_report,
        "utilization_report": utilization_report,
    }


def _parse_post_route_power(report_dir: Path) -> dict[str, Any] | None:
    power_report = report_dir / "post_route_power.rpt"
    if not power_report.exists():
        return None
    metrics = parse_vivado_power_text(power_report.read_text(encoding="utf-8", errors="ignore"))
    if metrics.get("power_w") is None:
        return None
    return {
        "metrics": metrics,
        "power_report": power_report,
    }


def _build_preferred_metrics(
    *,
    hls_metrics: dict[str, Any],
    vivado_actual: dict[str, Any] | None,
) -> dict[str, Any]:
    preferred = dict(hls_metrics)
    if vivado_actual is None:
        return preferred

    post_route = vivado_actual.get("metrics", {}).get("post_route", {})
    if not isinstance(post_route, dict) or not post_route:
        return preferred

    preferred["dsp"] = int(post_route.get("dsp") or preferred.get("dsp") or 0)
    preferred["lut"] = int(post_route.get("lut") or preferred.get("lut") or 0)
    preferred["ff"] = int(post_route.get("ff") or preferred.get("ff") or 0)

    block_ram_tile = post_route.get("block_ram_tile")
    if block_ram_tile is not None:
        preferred["bram"] = int(math.ceil(float(block_ram_tile)))
        preferred["bram_tile"] = float(block_ram_tile)
    preferred["bram_18k"] = int(post_route.get("ramb18") or preferred.get("bram_18k") or 0)
    preferred["bram_36k"] = int(post_route.get("ramb36") or preferred.get("bram_36k") or 0)

    if post_route.get("fmax_est_mhz") is not None:
        preferred["fmax_est_mhz"] = float(post_route["fmax_est_mhz"])
        if preferred["fmax_est_mhz"] > 0:
            preferred["estimated_clock_period_ns"] = 1_000.0 / preferred["fmax_est_mhz"]

    if post_route.get("power_w") is not None:
        preferred["power_w"] = float(post_route["power_w"])
        preferred["power_source"] = str(post_route.get("power_source", "vivado_post_route"))

    return preferred


def _post_route_metrics(vivado_actual: dict[str, Any] | None) -> dict[str, Any]:
    if vivado_actual is None:
        return {}
    metrics = vivado_actual.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    post_route = metrics.get("post_route", {})
    return post_route if isinstance(post_route, dict) else {}


if __name__ == "__main__":
    main()
