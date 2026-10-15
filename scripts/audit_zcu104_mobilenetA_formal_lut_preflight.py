#!/usr/bin/env python3
"""Preflight the formal ZCU104 MobileNetA operator-cost LUT.

This audit is intentionally read-only with respect to the LUT inputs.  It
checks that the recovered 88-row table is complete, payload-hash bound, and
clearly scoped as HLS/route operator evidence rather than board runtime or
power evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DEFAULT_LUT_PATH = Path(
    "artifacts/zcu104_full_campaign_v1/operator_cost_lut/"
    "mobilenetA_full_dmdc_cfem_v3_gap1280/hardware_cost_lut.formal.json"
)
DEFAULT_COVERAGE_PATH = DEFAULT_LUT_PATH.with_name("coverage_audit.formal.json")
DEFAULT_OUTPUT_PATH = DEFAULT_LUT_PATH.with_name("preflight.formal.json")

EXPECTED_DATA_KIND = "FRESH_ZCU104_HLS_ROUTE_OPERATOR_COST_WITH_RECOVERY"
EXPECTED_SEARCH_SPACE_VERSION = "mobilenetA_full_dmdc_cfem_v1"
EXPECTED_HARDWARE_LUT_VERSION = "mobilenetA_full_dmdc_cfem_v3_gap1280"
SUPPORTED_LUT_VERSIONS = {EXPECTED_HARDWARE_LUT_VERSION}
EXPECTED_ROWS = 88
EXPECTED_BOARD = "ZCU104"
EXPECTED_PART = "xczu7ev-ffvc1156-2-e"
EXPECTED_CLOCK_MHZ = 200.0
REQUIRED_COST_FIELDS = (
    "latency_cycles_hls",
    "latency_ns_hls",
    "latency_ms_hls",
    "lut_post_route",
    "ff_post_route",
    "dsp_post_route",
    "bram_tile_post_route",
    "power_w_vivado_vectorless",
)


def canonical_sha256(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hashed_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required formal LUT preflight input: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    expected_hash = str(payload.get("payload_sha256") or "")
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    actual_hash = canonical_sha256(unsigned)
    if expected_hash != actual_hash:
        raise ValueError(
            f"payload_sha256 mismatch for {path}: {expected_hash} != {actual_hash}"
        )
    return payload, sha256_file(path)


def require_equal(blockers: list[str], name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        blockers.append(f"{name}: {observed!r} != {expected!r}")


def finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def audit(
    *,
    lut_path: Path,
    coverage_path: Path,
    output_path: Path | None,
) -> dict[str, Any]:
    lut_path = lut_path.resolve()
    coverage_path = coverage_path.resolve()
    lut, lut_file_sha256 = load_hashed_json(lut_path)
    coverage, coverage_file_sha256 = load_hashed_json(coverage_path)

    blockers: list[str] = []
    warnings: list[str] = []

    require_equal(blockers, "lut.artifact_type", lut.get("artifact_type"), "hardware_cost_lut")
    require_equal(blockers, "coverage.artifact_type", coverage.get("artifact_type"), "operator_cost_lut_coverage_audit")
    for prefix, payload in (("lut", lut), ("coverage", coverage)):
        require_equal(blockers, f"{prefix}.data_kind", payload.get("data_kind"), EXPECTED_DATA_KIND)
        require_equal(
            blockers,
            f"{prefix}.search_space_version",
            payload.get("search_space_version"),
            EXPECTED_SEARCH_SPACE_VERSION,
        )
        require_equal(
            blockers,
            f"{prefix}.hardware_lut_version",
            payload.get("hardware_lut_version"),
            lut.get("hardware_lut_version"),
        )
        require_equal(blockers, f"{prefix}.search_ready", payload.get("search_ready"), True)

    require_equal(
        blockers,
        "coverage.operator_cost_payload_sha256",
        coverage.get("operator_cost_payload_sha256"),
        lut.get("payload_sha256"),
    )
    if lut.get("hardware_lut_version") not in SUPPORTED_LUT_VERSIONS:
        blockers.append("unsupported hardware LUT version")
    expected_rows = int(coverage.get("expected_rows") or 0)
    ready_rows = int(coverage.get("ready_rows") or 0)
    require_equal(blockers, "coverage.expected_rows", expected_rows, EXPECTED_ROWS)
    require_equal(blockers, "coverage.ready_rows", ready_rows, EXPECTED_ROWS)

    rows = lut.get("rows")
    if not isinstance(rows, list):
        blockers.append("lut.rows is not a list")
        rows = []
    require_equal(blockers, "len(lut.rows)", len(rows), EXPECTED_ROWS)

    seen_row_keys: set[str] = set()
    ready_count = 0
    timing_met_count = 0
    evidence_sources: dict[str, int] = {}
    operators: dict[str, int] = {}
    missing_cost_rows: list[str] = []
    target_mismatch_rows: list[str] = []
    invalid_cost_rows: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            blockers.append(f"row[{index}] is not a JSON object")
            continue
        row_key = str(row.get("row_key") or f"row[{index}]")
        if row_key in seen_row_keys:
            blockers.append(f"duplicate row_key: {row_key}")
        seen_row_keys.add(row_key)
        if row.get("ready") is True:
            ready_count += 1
        else:
            blockers.append(f"row not ready: {row_key}")
        if row.get("timing_met") is True:
            timing_met_count += 1
        evidence_source = str(row.get("evidence_source") or "UNKNOWN")
        evidence_sources[evidence_source] = evidence_sources.get(evidence_source, 0) + 1
        operator = str(row.get("operator") or "UNKNOWN")
        operators[operator] = operators.get(operator, 0) + 1
        if (
            row.get("target_board") != EXPECTED_BOARD
            or row.get("target_part") != EXPECTED_PART
            or float(row.get("target_clock_mhz") or 0.0) != EXPECTED_CLOCK_MHZ
        ):
            target_mismatch_rows.append(row_key)
        cost = row.get("cost")
        if not isinstance(cost, Mapping):
            missing_cost_rows.append(row_key)
            continue
        missing = [field for field in REQUIRED_COST_FIELDS if cost.get(field) is None]
        if missing:
            missing_cost_rows.append(f"{row_key}:{','.join(missing)}")
            continue
        if not all(finite_number(cost.get(field)) for field in REQUIRED_COST_FIELDS):
            invalid_cost_rows.append(row_key)
            continue
        positive_fields = ("latency_cycles_hls", "latency_ns_hls", "latency_ms_hls")
        if any(float(cost[field]) <= 0.0 for field in positive_fields):
            invalid_cost_rows.append(row_key)
        nonnegative_fields = (
            "lut_post_route",
            "ff_post_route",
            "dsp_post_route",
            "bram_tile_post_route",
            "power_w_vivado_vectorless",
        )
        if any(float(cost[field]) < 0.0 for field in nonnegative_fields):
            invalid_cost_rows.append(row_key)

    if ready_count != EXPECTED_ROWS:
        blockers.append(f"ready row count {ready_count} != {EXPECTED_ROWS}")
    if target_mismatch_rows:
        blockers.append(f"target mismatch rows: {target_mismatch_rows[:10]}")
    if missing_cost_rows:
        blockers.append(f"missing cost rows: {missing_cost_rows[:10]}")
    if invalid_cost_rows:
        blockers.append(f"invalid numeric cost rows: {invalid_cost_rows[:10]}")
    if timing_met_count != EXPECTED_ROWS:
        warnings.append(
            f"timing_met_rows={timing_met_count}/{EXPECTED_ROWS}; do not claim every "
            "operator row met timing from this preflight alone"
        )

    claim_boundary = lut.get("claim_boundary")
    head_gaps = [row for row in rows if row.get("operator") == "global_avg_pool"
                 and row.get("shape_name") == "gap_7_1280"]
    if len(head_gaps) != 1:
        blockers.append("missing exact MobileNetA head shape: global_avg_pool/gap_7_1280")
    else:
        spec = head_gaps[0].get("op_spec") or {}
        if (spec.get("in_channels"), spec.get("out_channels"), spec.get("input_resolution")) != (1280, 1280, [7, 7]):
            blockers.append("gap_7_1280 op_spec disagrees with the model head")
    warnings.append("Search latency/energy must use HLS cycles at frozen 200MHz, not report estimated-clock real time")
    if not isinstance(claim_boundary, Mapping):
        blockers.append("lut.claim_boundary missing")
    else:
        require_equal(
            blockers,
            "claim_boundary.board_classification",
            claim_boundary.get("board_classification"),
            "NOT_MEASURED_IN_THIS_LUT",
        )
        require_equal(
            blockers,
            "claim_boundary.external_power",
            claim_boundary.get("external_power"),
            "NOT_MEASURED_IN_THIS_LUT",
        )
        require_equal(
            blockers,
            "claim_boundary.vivado_power",
            claim_boundary.get("vivado_power"),
            "VECTORLESS_ESTIMATE_ONLY",
        )

    status = "PASS_FORMAL_ZCU104_OPERATOR_COST_LUT_PREFLIGHT" if not blockers else "FAIL"
    payload: dict[str, Any] = {
        "artifact_type": "zcu104_operator_cost_lut_preflight",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "search_ready": not blockers,
        "lut": {
            "path": str(lut_path),
            "file_sha256": lut_file_sha256,
            "payload_sha256": lut.get("payload_sha256"),
            "data_kind": lut.get("data_kind"),
            "search_space_version": lut.get("search_space_version"),
            "hardware_lut_version": lut.get("hardware_lut_version"),
            "source_status": lut.get("status"),
        },
        "coverage": {
            "path": str(coverage_path),
            "file_sha256": coverage_file_sha256,
            "payload_sha256": coverage.get("payload_sha256"),
            "operator_cost_payload_sha256": coverage.get("operator_cost_payload_sha256"),
            "source_status": coverage.get("status"),
        },
        "row_summary": {
            "expected_rows": EXPECTED_ROWS,
            "rows": len(rows),
            "ready_rows": ready_count,
            "timing_met_rows": timing_met_count,
            "operators": dict(sorted(operators.items())),
            "evidence_sources": dict(sorted(evidence_sources.items())),
        },
        "warnings": warnings,
        "blockers": blockers,
        "claim_boundary": {
            "preflight_scope": "operator HLS/route cost LUT integrity and coverage only",
            "board_classification": "NOT_MEASURED_BY_THIS_PREFLIGHT",
            "board_power": "NOT_MEASURED_BY_THIS_PREFLIGHT",
            "vivado_power_w": "VECTORLESS_ESTIMATE_ONLY_NOT_BOARD_POWER",
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)

    if output_path is not None:
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lut-path", default=str(DEFAULT_LUT_PATH))
    parser.add_argument("--coverage-path", default=str(DEFAULT_COVERAGE_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--no-output", action="store_true")
    args = parser.parse_args()
    payload = audit(
        lut_path=Path(args.lut_path),
        coverage_path=Path(args.coverage_path),
        output_path=None if args.no_output else Path(args.output),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
