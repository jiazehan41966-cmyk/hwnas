#!/usr/bin/env python3
"""Build the fresh ZCU104 operator-cost scaffold for fourstage_cfem20_v1.

This command validates the frozen search-space artifacts and creates identity,
shape, and evidence slots only.  It never imports historical cost rows and it
does not run search, training, HLS, RTL, route, bitstream, or board commands.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SEARCH_SPACE_VERSION = "fourstage_cfem20_v1"
DEFAULT_FREEZE_DIR = (
    ROOT
    / "artifacts"
    / "sonar_fourstage_operator_v2"
    / "frozen_search_space_v5_cfem20"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "zcu104_full_campaign_v1"
    / "operator_cost_lut"
    / SEARCH_SPACE_VERSION
)

EXPECTED_STAGE2 = ("k3_e3", "k3_e6", "k5_e3", "k5_e6")
EXPECTED_STAGE4 = (
    "skip",
    "mbconv_k3_e3",
    "mbconv_k5_e3",
    "dmdc_conv_sonar",
    "cfem_sonar",
)
FROZEN_FILENAMES = (
    "search_space_definition.json",
    "candidate_manifest_20.json",
    "operator_definition.json",
    "search_space_audit.json",
)
METRIC_FIELDS = (
    "latency_hls_cycles",
    "latency_hls_ms",
    "latency_board_ms",
    "fps",
    "lut",
    "ff",
    "dsp",
    "bram18",
    "bram36",
    "uram",
    "wns_ns",
    "tns_ns",
    "power_vivado_estimated_w",
    "p_idle_w",
    "p_active_w",
    "p_dynamic_w",
    "energy_total_mj_per_op",
    "energy_dynamic_mj_per_op",
    "temperature_start_c",
    "temperature_end_c",
)
EVIDENCE_FIELDS = (
    "hls_report",
    "route_utilization_report",
    "route_timing_report",
    "vivado_power_report",
    "xsa",
    "hwh",
    "bitstream",
    "runtime",
    "latency_receipt",
    "power_manifest",
    "power_raw_index",
)


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def load_frozen(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    expected = str(payload.get("payload_sha256") or "")
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    actual = canonical_sha256(unsigned)
    if expected != actual:
        raise ValueError(
            f"{path} payload hash mismatch: expected {expected}, actual {actual}"
        )
    if payload.get("search_space_version") != SEARCH_SPACE_VERSION:
        raise ValueError(f"{path} does not bind {SEARCH_SPACE_VERSION}")
    return payload


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    payload["payload_sha256"] = canonical_sha256(unsigned)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def shape_token(values: list[int]) -> str:
    return "x".join(str(int(value)) for value in values)


def make_cost_key(stage: str, definition: dict[str, Any]) -> str:
    return (
        f"{stage}__{definition['candidate_key']}__"
        f"{shape_token(definition['input_shape'])}_to_"
        f"{shape_token(definition['output_shape'])}__s{int(definition['stride'])}"
    )


def make_row(stage: str, definition: dict[str, Any], source_hash: str) -> dict[str, Any]:
    operator_parameters = {
        key: value
        for key, value in definition.items()
        if key
        not in {
            "candidate_id",
            "candidate_key",
            "input_shape",
            "output_shape",
        }
    }
    return {
        "cost_key": make_cost_key(stage, definition),
        "search_space_version": SEARCH_SPACE_VERSION,
        "stage": stage,
        "candidate_id": definition["candidate_id"],
        "operator_id": definition["candidate_key"],
        "op_name": definition["op_name"],
        "kernel": definition.get("kernel"),
        "expand_ratio": definition.get("expand_ratio"),
        "stride": int(definition["stride"]),
        "input_shape_chw": list(definition["input_shape"]),
        "output_shape_chw": list(definition["output_shape"]),
        "operator_parameters": operator_parameters,
        "operator_definition_payload_sha256": source_hash,
        "target": {
            "board": "ZCU104",
            "part": "xczu7ev-ffvc1156-2-e",
            "clock_period_ns": 5.0,
            "toolchain": "2022.2",
        },
        "metrics": {field: None for field in METRIC_FIELDS},
        "evidence": {field: None for field in EVIDENCE_FIELDS},
        "measurement_status": "PENDING_FRESH_ZCU104_MEASUREMENT",
        "gate_status": "PENDING",
        "failure_reason": None,
        "historical_cost_row_allowed": False,
    }


def build(freeze_dir: Path, output_dir: Path) -> dict[str, Any]:
    frozen_paths = {name: freeze_dir / name for name in FROZEN_FILENAMES}
    missing = [str(path) for path in frozen_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen search-space inputs: " + ", ".join(missing))

    frozen = {name: load_frozen(path) for name, path in frozen_paths.items()}
    definition = frozen["search_space_definition.json"]
    candidates = frozen["candidate_manifest_20.json"]
    operators = frozen["operator_definition.json"]
    audit = frozen["search_space_audit.json"]

    if audit.get("status") != "PASS" or not all(audit.get("checks", {}).values()):
        raise ValueError("frozen search-space audit is not PASS")
    if definition.get("raw_network_count") != 20 or len(candidates.get("rows", [])) != 20:
        raise ValueError("fourstage_cfem20_v1 must contain exactly 20 RAW networks")

    stage2 = operators.get("stage2_operator_definitions", [])
    stage4 = operators.get("active_stage4_operator_definitions", [])
    if tuple(row.get("candidate_key") for row in stage2) != EXPECTED_STAGE2:
        raise ValueError("Stage2 frozen order or identities changed")
    if tuple(row.get("candidate_key") for row in stage4) != EXPECTED_STAGE4:
        raise ValueError("Stage4 frozen order or identities changed")

    operator_definition_hash = str(operators["payload_sha256"])
    rows = [make_row("stage2", row, operator_definition_hash) for row in stage2]
    rows.extend(make_row("stage4", row, operator_definition_hash) for row in stage4)
    if len(rows) != 9 or len({row["cost_key"] for row in rows}) != 9:
        raise AssertionError("operator-cost scaffold must contain 9 unique rows")

    key_by_operator = {
        (row["stage"], row["operator_id"]): row["cost_key"] for row in rows
    }
    candidate_mappings = []
    for candidate in candidates["rows"]:
        stage2_key = key_by_operator[("stage2", candidate["stage2"]["candidate_key"])]
        stage4_key = key_by_operator[("stage4", candidate["stage4"]["candidate_key"])]
        candidate_mappings.append(
            {
                "architecture_id": candidate["architecture_id"],
                "architecture_sha256": candidate["architecture_sha256"],
                "stage2_cost_key": stage2_key,
                "stage4_cost_key": stage4_key,
                "mapping_status": "PASS_STRUCTURAL_KEYS_ONLY",
            }
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    scaffold_path = output_dir / "hardware_cost_lut.scaffold.json"
    coverage_path = output_dir / "coverage_audit.json"
    manifest_path = output_dir / "build_manifest.json"
    scaffold = {
        "schema_version": 1,
        "generated_at": generated_at,
        "campaign_id": "zcu104_full_campaign_v1",
        "search_space_version": SEARCH_SPACE_VERSION,
        "status": "SCAFFOLD_ONLY_0_OF_9_MEASURED",
        "expected_operator_rows": 9,
        "rows": rows,
        "claim_boundary": (
            "Identity and evidence slots only. Null metrics are not measurements, "
            "historical cost rows are forbidden, and HW-NAS search remains blocked."
        ),
    }
    coverage = {
        "schema_version": 1,
        "generated_at": generated_at,
        "campaign_id": "zcu104_full_campaign_v1",
        "search_space_version": SEARCH_SPACE_VERSION,
        "status": "BLOCKED_MEASURED_COVERAGE_0_OF_9",
        "operator_key_definition_coverage": {
            "status": "PASS",
            "covered": 9,
            "expected": 9,
        },
        "raw_candidate_key_mapping_coverage": {
            "status": "PASS",
            "covered": len(candidate_mappings),
            "expected": 20,
        },
        "measured_operator_coverage": {
            "status": "PENDING",
            "covered": 0,
            "expected": 9,
        },
        "measured_raw_candidate_coverage": {
            "status": "PENDING",
            "covered": 0,
            "expected": 20,
        },
        "candidate_mappings": candidate_mappings,
        "search_ready": False,
        "claim_boundary": (
            "20/20 structural key mapping is not measured LUT coverage. Search remains "
            "blocked until all 9 operator rows have accepted fresh ZCU104 evidence."
        ),
    }
    write_payload(scaffold_path, scaffold)
    write_payload(coverage_path, coverage)

    input_records = {
        name: {
            "path": portable_path(path),
            "file_sha256": sha256_file(path),
            "payload_sha256": frozen[name]["payload_sha256"],
        }
        for name, path in frozen_paths.items()
    }
    manifest = {
        "schema_version": 1,
        "generated_at": generated_at,
        "campaign_id": "zcu104_full_campaign_v1",
        "search_space_version": SEARCH_SPACE_VERSION,
        "status": "PASS_SCAFFOLD_BUILT",
        "inputs": input_records,
        "outputs": {
            "hardware_cost_lut_scaffold": {
                "path": portable_path(scaffold_path),
                "sha256": sha256_file(scaffold_path),
            },
            "coverage_audit": {
                "path": portable_path(coverage_path),
                "sha256": sha256_file(coverage_path),
            },
        },
        "forbidden_actions_executed": [],
        "claim_boundary": (
            "Scaffold construction only; no training, search, HLS, route, bitstream, "
            "board execution, latency, power, or energy measurement was run."
        ),
    }
    write_payload(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "scaffold": str(scaffold_path),
        "coverage": str(coverage_path),
        "manifest": str(manifest_path),
        "operator_rows": len(rows),
        "candidate_mappings": len(candidate_mappings),
        "measured_rows": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", type=Path, default=DEFAULT_FREEZE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = build(args.freeze_dir.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
