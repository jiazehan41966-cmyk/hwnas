#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
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

from common import build_cases, load_config, resolve_workspace_path
from hwnas_fpga.hardware.lut_pipeline import build_lut_from_manifest, load_lut_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a formal LUT JSON from an HLS/Vivado manifest.")
    parser.add_argument("--config", required=True, help="Path to builder config YAML")
    parser.add_argument("--manifest", default=None, help="Manifest path override")
    parser.add_argument("--output-lut-json", default=None, help="Output LUT JSON path override")
    parser.add_argument("--output-status-json", default=None, help="Output status JSON path override")
    parser.add_argument(
        "--implementation-version",
        default=None,
        help="Implementation version label. Defaults to fixed_vector_v2_YYYY_MM_DD.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    workspace = config.get("workspace", {})

    manifest_path = resolve_workspace_path(config, args.manifest or workspace["manifest_path"])
    status_path = resolve_workspace_path(
        config,
        args.output_status_json or workspace.get("formal_lut_status_json", "../results/formal_lut_status.json"),
    )
    default_lut_path = status_path.with_name(status_path.name.replace("status_", "").replace("_status", ""))
    if default_lut_path == status_path:
        default_lut_path = status_path.with_name("formal_lut.json")
    lut_path = resolve_workspace_path(config, args.output_lut_json) if args.output_lut_json else default_lut_path
    implementation_version = args.implementation_version or f"fixed_vector_v2_{datetime.now():%Y_%m_%d}"

    table, lut_summary = build_lut_from_manifest(manifest_path)
    manifest = load_lut_manifest(manifest_path)
    cases = build_cases(config)
    manifest_entries_by_case = {
        str(entry.get("metadata", {}).get("case_name") or ""): entry
        for entry in manifest["entries"]
        if isinstance(entry.get("metadata"), dict)
    }
    status_entries = [
        _case_status_entry(case, manifest_entries_by_case.get(case.case_name))
        for case in cases
    ]
    status_counts = Counter(entry["status"] for entry in status_entries)

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "implementation_version": implementation_version,
        "measurement_contract": manifest.get("measurement_contract") or config.get("measurement_contract"),
        "source_config": str(Path(args.config).expanduser().resolve()),
        "source_manifest": str(manifest_path),
        "candidate_cases": len(cases),
        "manifest_entries": len(manifest["entries"]),
        "entries_built": len(table),
        "status_counts": dict(status_counts),
        "table_stats": lut_summary.get("table_stats", {}),
        "query_key_note": (
            "v2 fixed-vector keys intentionally include pack_ch, ch_block, "
            "tile_order, target_ii, target_clock_mhz, and dsp_pack."
        ),
    }

    lut_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    table.save_json(str(lut_path), metadata=metadata)
    status_path.write_text(
        json.dumps({"metadata": metadata, "entries": status_entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Generated manifest LUT: {len(table)} entries -> {lut_path}")
    print(f"Generated manifest LUT status: {len(status_entries)} entries -> {status_path}")


def _case_status_entry(case: Any, entry: dict[str, Any] | None) -> dict[str, Any]:
    if entry is None:
        synthesis_status = _load_json_status(case.project_dir / "synthesis_status.json")
        downstream_status = _load_json_status(case.vivado_downstream_status_path)
        hls_status = synthesis_status.get("status") or "missing"
        if hls_status in {"success", "skipped_existing"}:
            status = "hls_success_downstream_missing"
        elif hls_status == "missing":
            status = "missing"
        else:
            status = f"hls_{hls_status}"
        op_spec = dict(case.op_spec)
        clock_mhz = float(case.target_clock_mhz)
        return {
            "case_name": case.case_name,
            "op_id": case.op_id,
            "status": status,
            "metrics_source": "none",
            "deployable_at_200mhz": False,
            "op_type": op_spec.get("op", case.operator),
            "op_spec": op_spec,
            "clock_mhz": clock_mhz,
            "hls_status": hls_status,
            "downstream_status": downstream_status.get("status") or "missing",
            "key_fields": _key_fields(op_spec),
        }
    return _status_entry(case, entry)


def _status_entry(case: Any, entry: dict[str, Any]) -> dict[str, Any]:
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    op_spec = entry.get("op_spec") if isinstance(entry.get("op_spec"), dict) else {}
    vivado_actual = entry.get("vivado_actual") if isinstance(entry.get("vivado_actual"), dict) else None
    hls_estimate = entry.get("hls_estimate") if isinstance(entry.get("hls_estimate"), dict) else {}
    synthesis_status = _load_json_status(case.project_dir / "synthesis_status.json")
    downstream_status_payload = _load_json_status(case.vivado_downstream_status_path)
    downstream_status = str(downstream_status_payload.get("status") or "missing")
    hls_status = str(synthesis_status.get("status") or hls_estimate.get("status") or "missing")
    clock_mhz = float(entry.get("clock_mhz") or op_spec.get("target_clock_mhz") or 0.0)

    status = "missing"
    metrics_source = "none"
    if vivado_actual and vivado_actual.get("metrics", {}).get("post_route"):
        vivado_status = str(vivado_actual.get("status") or metadata.get("downstream_status") or "success")
        if vivado_status in {"success", "skipped_existing"}:
            status = "measured_implementation"
            metrics_source = "vivado_actual.post_route"
        else:
            status = f"downstream_{vivado_status}"
            metrics_source = "vivado_actual"
    elif downstream_status not in {"missing", "success", "skipped_existing"}:
        status = f"downstream_{downstream_status}"
        metrics_source = "hls_estimate"
    elif str(hls_estimate.get("status") or metadata.get("synthesis_status") or "") in {"success", "skipped_existing"}:
        status = "measured_hls_only"
        metrics_source = "hls_estimate"

    return {
        "case_name": metadata.get("case_name"),
        "op_id": metadata.get("op_id"),
        "status": status,
        "metrics_source": metrics_source,
        "deployable_at_200mhz": bool(clock_mhz >= 199.9 and status == "measured_implementation"),
        "op_type": op_spec.get("op"),
        "op_spec": op_spec,
        "clock_mhz": clock_mhz,
        "hls_status": hls_status,
        "downstream_status": downstream_status,
        "key_fields": _key_fields(op_spec),
    }


def _key_fields(op_spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "pack_ch": op_spec.get("pack_ch"),
        "ch_block": op_spec.get("ch_block"),
        "tile_order": op_spec.get("tile_order"),
        "target_ii": op_spec.get("target_ii"),
        "target_clock_mhz": op_spec.get("target_clock_mhz"),
        "dsp_pack": op_spec.get("dsp_pack"),
    }


def _load_json_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
