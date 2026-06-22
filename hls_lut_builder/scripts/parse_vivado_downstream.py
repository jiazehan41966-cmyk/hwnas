#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
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

from common import build_cases, load_config, resolve_workspace_path, select_pilot_cases
from hwnas_fpga.hardware import (
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
    "post_synth_LUT",
    "post_synth_FF",
    "post_synth_BRAM_TILE",
    "post_synth_RAMB18",
    "post_synth_RAMB36",
    "post_synth_DSP",
    "post_synth_setup_WNS_ns",
    "post_synth_hold_WHS_ns",
    "post_synth_data_path_delay_ns",
    "post_synth_logic_delay_ns",
    "post_synth_route_delay_ns",
    "post_synth_Fmax_est",
    "post_route_LUT",
    "post_route_FF",
    "post_route_BRAM_TILE",
    "post_route_RAMB18",
    "post_route_RAMB36",
    "post_route_DSP",
    "post_route_setup_WNS_ns",
    "post_route_hold_WHS_ns",
    "post_route_data_path_delay_ns",
    "post_route_logic_delay_ns",
    "post_route_route_delay_ns",
    "post_route_Fmax_est",
    "power_w",
    "power_source",
    "report_dir",
    "downstream_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse downstream Vivado reports into CSV and manifest")
    parser.add_argument("--config", required=True, help="Path to builder config YAML")
    parser.add_argument("--project-root", default=None, help="Optional override for generated project root")
    parser.add_argument("--operator", action="append", default=None, help="Parse only specific operators")
    parser.add_argument("--case", action="append", default=None, help="Parse only specific cases")
    parser.add_argument("--pilot", action="store_true", help="Select one representative case per enabled operator")
    parser.add_argument("--csv", default=None, help="Optional CSV output override")
    parser.add_argument("--manifest", default=None, help="Optional manifest output override")
    parser.add_argument("--summary-json", default=None, help="Optional summary JSON override")
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
        raise SystemExit("No downstream Vivado cases matched the provided filters.")

    workspace = config.get("workspace", {})
    measurement_contract = _load_measurement_contract(config)
    csv_path = resolve_workspace_path(
        config,
        args.csv or workspace.get("downstream_csv_path", "../results/vivado_downstream.csv"),
    )
    manifest_path = resolve_workspace_path(
        config,
        args.manifest or workspace.get("downstream_manifest_path", "../results/vivado_downstream_manifest.yaml"),
    )
    summary_path = resolve_workspace_path(
        config,
        args.summary_json or workspace.get("downstream_parse_summary_json", "../results/vivado_downstream_parse_summary.json"),
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    missing_reports: list[str] = []

    for case in cases:
        report_dir = case.vivado_downstream_reports_dir
        status_payload = _load_status_payload(case.vivado_downstream_status_path)
        downstream_status = str(status_payload.get("status", "missing_report"))
        post_synth = _parse_stage_reports(report_dir, "post_synth", target_clock_ns=case.clock_period_ns)
        post_route = _parse_stage_reports(report_dir, "post_route", target_clock_ns=case.clock_period_ns)
        if post_synth is None or post_route is None:
            missing_reports.append(case.case_name)
            continue
        power = _parse_post_route_power(report_dir)
        post_route_metrics = {
            **post_route["timing"],
            **post_route["utilization"],
        }
        if power is not None:
            post_route_metrics.update(power["metrics"])

        height, width = case.op_spec.get("input_resolution", [0, 0])
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
            "post_synth_LUT": int(post_synth["utilization"]["lut"]),
            "post_synth_FF": int(post_synth["utilization"]["ff"]),
            "post_synth_BRAM_TILE": float(post_synth["utilization"]["block_ram_tile"]),
            "post_synth_RAMB18": int(post_synth["utilization"]["ramb18"]),
            "post_synth_RAMB36": int(post_synth["utilization"]["ramb36"]),
            "post_synth_DSP": int(post_synth["utilization"]["dsp"]),
            "post_synth_setup_WNS_ns": post_synth["timing"]["setup_wns_ns"],
            "post_synth_hold_WHS_ns": post_synth["timing"]["hold_whs_ns"],
            "post_synth_data_path_delay_ns": post_synth["timing"]["data_path_delay_ns"],
            "post_synth_logic_delay_ns": post_synth["timing"]["logic_delay_ns"],
            "post_synth_route_delay_ns": post_synth["timing"]["route_delay_ns"],
            "post_synth_Fmax_est": post_synth["timing"]["fmax_est_mhz"],
            "post_route_LUT": int(post_route["utilization"]["lut"]),
            "post_route_FF": int(post_route["utilization"]["ff"]),
            "post_route_BRAM_TILE": float(post_route["utilization"]["block_ram_tile"]),
            "post_route_RAMB18": int(post_route["utilization"]["ramb18"]),
            "post_route_RAMB36": int(post_route["utilization"]["ramb36"]),
            "post_route_DSP": int(post_route["utilization"]["dsp"]),
            "post_route_setup_WNS_ns": post_route["timing"]["setup_wns_ns"],
            "post_route_hold_WHS_ns": post_route["timing"]["hold_whs_ns"],
            "post_route_data_path_delay_ns": post_route["timing"]["data_path_delay_ns"],
            "post_route_logic_delay_ns": post_route["timing"]["logic_delay_ns"],
            "post_route_route_delay_ns": post_route["timing"]["route_delay_ns"],
            "post_route_Fmax_est": post_route["timing"]["fmax_est_mhz"],
            "power_w": "" if power is None else power["metrics"]["power_w"],
            "power_source": "implementation_not_parsed" if power is None else power["metrics"]["power_source"],
            "report_dir": str(report_dir),
            "downstream_status": downstream_status,
        }
        rows.append(row)
        reports = {
            "post_synth_timing": str(post_synth["timing_report"]),
            "post_synth_utilization": str(post_synth["utilization_report"]),
            "post_route_timing": str(post_route["timing_report"]),
            "post_route_utilization": str(post_route["utilization_report"]),
        }
        if power is not None:
            reports["post_route_power"] = str(power["power_report"])

        manifest_entries.append(
            {
                "op_spec": case.op_spec,
                "clock_mhz": case.target_clock_mhz,
                "reports": reports,
                "metrics": {
                    "post_synth": {
                        **post_synth["timing"],
                        **post_synth["utilization"],
                    },
                    "post_route": post_route_metrics,
                },
                "metadata": {
                    "op_id": case.op_id,
                    "case_name": case.case_name,
                    "implementation_name": case.implementation_name,
                    "clock_profile_name": case.clock_profile_name,
                    "downstream_status": downstream_status,
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

    summary = {
        "parsed_cases": len(rows),
        "missing_reports": missing_reports,
        "csv_path": str(csv_path),
        "manifest_path": str(manifest_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Parsed {len(rows)} downstream Vivado report sets.")
    if missing_reports:
        print("Missing downstream reports:")
        for case_name in missing_reports:
            print(case_name)
    print(f"CSV: {csv_path}")
    print(f"Manifest: {manifest_path}")


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


def _load_status_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _parse_stage_reports(report_dir: Path, stage_name: str, *, target_clock_ns: float) -> dict[str, Any] | None:
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
