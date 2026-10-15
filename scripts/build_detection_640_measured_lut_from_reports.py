#!/usr/bin/env python3
"""Build the detection 640 multi-scale measured LUT artifact from HLS/Vivado reports."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
HLS_SCRIPT_ROOT = ROOT / "hls_lut_builder" / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(HLS_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(HLS_SCRIPT_ROOT))

from common import build_cases, load_config, resolve_workspace_path
from hwnas_fpga.hardware.lut_pipeline import build_lut_from_manifest, load_lut_manifest


DEFAULT_CONFIG = ROOT / "hls_lut_builder" / "configs" / "candidate_kernels_detection_multiscale_640.yaml"
DEFAULT_OUTPUT = ROOT / "artifacts" / "detection_multiscale_640" / "hardware_cost_table.measured.zcu104.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build strict-consumable measured LUT files for the fixed detection 640 Neck/Head."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Detection HLS LUT config")
    parser.add_argument("--manifest", default=None, help="HLS manifest path override")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Human-readable measured table artifact")
    parser.add_argument("--output-lut-json", default=None, help="Strict LUT JSON output override")
    parser.add_argument("--output-status-json", default=None, help="Strict LUT status JSON output override")
    parser.add_argument("--allow-incomplete", action="store_true", help="Write artifact even when not all 14 rows are measured")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(
        Path(args.config),
        manifest_override=None if args.manifest is None else Path(args.manifest),
        lut_json_override=None if args.output_lut_json is None else Path(args.output_lut_json),
        status_json_override=None if args.output_status_json is None else Path(args.output_status_json),
        allow_incomplete=bool(args.allow_incomplete),
    )
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload["payload_sha256"] = _stable_payload_sha256(payload)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"status {payload['status']}")
    print(f"rows {payload['summary']['rows_total']}")
    print(f"measured_rows {payload['summary']['rows_measured']}")
    print(f"payload_sha256 {payload['payload_sha256']}")
    print(f"artifact {output_path}")


def build_payload(
    config_path: Path = DEFAULT_CONFIG,
    *,
    manifest_override: Path | None = None,
    lut_json_override: Path | None = None,
    status_json_override: Path | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    workspace = config.get("workspace", {})
    manifest_path = (
        manifest_override.expanduser().resolve()
        if manifest_override is not None
        else resolve_workspace_path(config, workspace["manifest_path"])
    )
    lut_path = (
        lut_json_override.expanduser().resolve()
        if lut_json_override is not None
        else resolve_workspace_path(config, workspace.get("formal_lut_json", "../results/detection_multiscale_640/formal_lut.json"))
    )
    status_path = (
        status_json_override.expanduser().resolve()
        if status_json_override is not None
        else resolve_workspace_path(config, workspace.get("formal_lut_status_json", "../results/detection_multiscale_640/formal_lut_status.json"))
    )
    cases = build_cases(config)
    expected_by_case = {case.case_name: case for case in cases}

    if not manifest_path.is_file():
        return _blocked_payload(
            config_path=config_path,
            manifest_path=manifest_path,
            lut_path=lut_path,
            status_path=status_path,
            cases=cases,
            blocker=f"HLS manifest not found: {manifest_path}",
        )

    manifest = load_lut_manifest(manifest_path)
    manifest_entries_by_case = {
        str(entry.get("metadata", {}).get("case_name") or ""): entry
        for entry in manifest["entries"]
        if isinstance(entry.get("metadata"), dict)
    }
    table, table_summary = build_lut_from_manifest(manifest_path)
    rows = [
        _row_for_case(case, manifest_entries_by_case.get(case.case_name))
        for case in cases
    ]
    status_entries = [
        _status_for_case(case, manifest_entries_by_case.get(case.case_name))
        for case in cases
    ]
    measured_rows = [row for row in rows if row["status"] == "measured"]
    status_counts = Counter(row["status"] for row in rows)
    missing_cases = [
        name for name in expected_by_case if name not in manifest_entries_by_case
    ]

    complete = len(measured_rows) == len(cases) and not missing_cases
    if not complete and not allow_incomplete:
        status = "BLOCKED_INCOMPLETE_MEASUREMENT"
    else:
        status = "PASS" if complete else "PARTIAL"

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_config": str(config_path.expanduser().resolve()),
        "source_manifest": str(manifest_path),
        "measurement_contract": manifest.get("measurement_contract") or config.get("measurement_contract"),
        "expected_cases": len(cases),
        "manifest_entries": len(manifest["entries"]),
        "entries_built": len(table),
        "status_counts": dict(status_counts),
        "claim_boundary": {
            "scope": "operator-level HLS/Vivado LUT rows for fixed detection Neck/Head at 640x640 P3/P4/P5",
            "target_part": "xczu7ev-ffvc1156-2-e",
            "backbone_search_space": "unchanged classification-derived backbone/search-space",
            "neck_head_search": "fixed_not_searched_in_stage_1",
            "board_detection_accuracy": "not_measured_by_this_artifact",
            "external_power": "not_measured_by_this_artifact",
        },
    }
    lut_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    table.save_json(str(lut_path), metadata=metadata)
    status_path.write_text(
        json.dumps({"metadata": metadata, "entries": status_entries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "artifact_type": "detection_multiscale_640_measured_hardware_cost_table",
        "schema_version": 1,
        "status": status,
        "summary": {
            "rows_total": len(cases),
            "rows_measured": len(measured_rows),
            "status_counts": dict(status_counts),
            "missing_cases": missing_cases,
            "formal_lut_json": str(lut_path),
            "formal_lut_status_json": str(status_path),
            "table_stats": table_summary.get("table_stats", {}),
        },
        "metadata": metadata,
        "rows": rows,
    }


def _blocked_payload(
    *,
    config_path: Path,
    manifest_path: Path,
    lut_path: Path,
    status_path: Path,
    cases: list[Any],
    blocker: str,
) -> dict[str, Any]:
    return {
        "artifact_type": "detection_multiscale_640_measured_hardware_cost_table",
        "schema_version": 1,
        "status": "BLOCKED_MISSING_REPORTS",
        "summary": {
            "rows_total": len(cases),
            "rows_measured": 0,
            "status_counts": {"missing_manifest": len(cases)},
            "missing_cases": [case.case_name for case in cases],
            "formal_lut_json": str(lut_path),
            "formal_lut_status_json": str(status_path),
            "blocker": blocker,
        },
        "metadata": {
            "generated_at": datetime.now().astimezone().isoformat(),
            "source_config": str(config_path.expanduser().resolve()),
            "source_manifest": str(manifest_path),
        },
        "rows": [
            {
                "case_name": case.case_name,
                "op_id": case.op_id,
                "detection_role": case.op_spec.get("detection_role"),
                "detection_level": case.op_spec.get("detection_level"),
                "status": "missing_manifest",
                "op_spec": case.op_spec,
            }
            for case in cases
        ],
    }


def _row_for_case(case: Any, entry: dict[str, Any] | None) -> dict[str, Any]:
    base = {
        "case_name": case.case_name,
        "op_id": case.op_id,
        "detection_role": case.op_spec.get("detection_role"),
        "detection_level": case.op_spec.get("detection_level"),
        "op_spec": case.op_spec,
    }
    if entry is None:
        return {**base, "status": "missing_manifest"}

    hls_estimate = entry.get("hls_estimate") if isinstance(entry.get("hls_estimate"), dict) else {}
    vivado_actual = entry.get("vivado_actual") if isinstance(entry.get("vivado_actual"), dict) else None
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    post_route = {}
    if vivado_actual is not None:
        post_route = vivado_actual.get("metrics", {}).get("post_route", {})
        if not isinstance(post_route, dict):
            post_route = {}
    status = "measured" if post_route else "measured_hls_only"
    selected_metrics = post_route if post_route else metrics
    return {
        **base,
        "status": status,
        "metrics_source": "vivado_actual.post_route" if post_route else "hls_estimate",
        "hls_report": hls_estimate.get("report") or entry.get("report"),
        "vivado_reports": None if vivado_actual is None else vivado_actual.get("reports"),
        "latency_cycles": int(metrics.get("cycles") or 0),
        "latency_ms": float(metrics.get("latency_ms") or 0.0),
        "lut": _optional_int(selected_metrics.get("lut")),
        "ff": _optional_int(selected_metrics.get("ff")),
        "bram": _optional_bram(selected_metrics),
        "dsp": _optional_int(selected_metrics.get("dsp")),
        "setup_wns_ns": _optional_float(selected_metrics.get("setup_wns_ns")),
        "fmax_est_mhz": _optional_float(selected_metrics.get("fmax_est_mhz")),
        "power_w": _optional_float(selected_metrics.get("power_w")),
    }


def _status_for_case(case: Any, entry: dict[str, Any] | None) -> dict[str, Any]:
    row = _row_for_case(case, entry)
    raw_status = row["status"]
    strict_status = "measured" if raw_status == "measured" else raw_status
    return {
        "case_name": case.case_name,
        "op_id": case.op_id,
        "status": strict_status,
        "op_type": case.op_spec.get("op", case.operator),
        "lookup_op": case.op_spec.get("op", case.operator),
        "defer_reason": None if strict_status == "measured" else strict_status,
        "root_cause_bucket": None if strict_status == "measured" else "incomplete_detection_lut_measurement",
        "op_spec": case.op_spec,
        "metrics_source": row.get("metrics_source", "none"),
        "deployable_at_200mhz": bool(strict_status == "measured"),
    }


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_bram(metrics: dict[str, Any]) -> int | None:
    if metrics.get("bram") not in (None, ""):
        return _optional_int(metrics.get("bram"))
    if metrics.get("block_ram_tile") not in (None, ""):
        value = float(metrics["block_ram_tile"])
        return int(value) if value.is_integer() else int(value) + 1
    return None


def _stable_payload_sha256(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("payload_sha256", None)
    encoded = json.dumps(clone, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    main()
