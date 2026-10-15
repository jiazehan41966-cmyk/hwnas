#!/usr/bin/env python3
"""Convert the formal MobileNetA ZCU104 LUT into a candidate-HLS status index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_sha256(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_formal_lut(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    expected = str(payload.get("payload_sha256") or "")
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    observed = canonical_sha256(unsigned)
    if expected != observed:
        raise ValueError(f"formal LUT payload hash mismatch: {expected} != {observed}")
    if payload.get("search_ready") is not True:
        raise ValueError("formal LUT is not search_ready")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("formal LUT has no rows")
    return payload


def status_op_spec(row: dict[str, Any]) -> dict[str, Any]:
    op_spec = dict(row["op_spec"])
    operator = str(row.get("operator") or "")
    if operator in {"stem_conv_k3_s2", "pw_conv"}:
        op_spec["op"] = "conv"
    elif operator == "mbconv_target_slot_k3_e3":
        op_spec["op"] = "mbconv_e3_k3"
    elif operator == "mbconv_target_slot_k5_e3":
        op_spec["op"] = "mbconv_e3_k5"
    # hls_evidence.candidate_roles uses the logical op shape with scalar
    # pi/po/u defaults.  Keep that key convention so the status index is only a
    # bridge to the same measured report, not a second cost model.
    op_spec["input_parallelism"] = 1
    op_spec["output_parallelism"] = 1
    op_spec["unroll_factor"] = 1
    op_spec["groups"] = 1
    for extra in (
        "search_space_version",
        "operator_family",
        "search_role",
        "cost_key",
        "target_clock_mhz",
    ):
        op_spec.pop(extra, None)
    return op_spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-lut",
        default=(
            "artifacts/zcu104_full_campaign_v1/operator_cost_lut/"
            "mobilenetA_full_dmdc_cfem_v2_recovered/hardware_cost_lut.formal.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/zcu104_full_campaign_v1/operator_cost_lut/"
            "mobilenetA_full_dmdc_cfem_v2_recovered/hls_status_from_formal_lut.json"
        ),
    )
    parser.add_argument("--require-local-reports", action="store_true")
    args = parser.parse_args()

    lut_path = Path(args.formal_lut).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    payload = load_formal_lut(lut_path)

    entries: list[dict[str, Any]] = []
    missing_reports: list[dict[str, str]] = []
    for row in payload["rows"]:
        report_path = Path(str(row.get("hls_report_path") or "")).expanduser()
        report_ok = report_path.is_file()
        expected_sha = str(row.get("hls_report_sha256") or "")
        observed_sha = sha256_file(report_path) if report_ok else ""
        if report_ok and expected_sha and observed_sha != expected_sha:
            raise ValueError(
                f"HLS report hash mismatch for {row.get('row_key')}: "
                f"{observed_sha} != {expected_sha}"
            )
        if not report_ok:
            missing_reports.append(
                {"row_key": str(row.get("row_key")), "hls_report_path": str(report_path)}
            )
        entry = {
            "case_name": row["evidence_case_name"],
            "status": "measured" if report_ok else "missing_report_path",
            "op_type": row["operator"],
            "op_spec": status_op_spec(row),
            "report_path": str(report_path),
            "report_sha256": expected_sha,
            "source_formal_row_key": row["row_key"],
            "source_formal_row_sha256": row.get("row_sha256"),
            "source_formal_lut_sha256": sha256_file(lut_path),
        }
        entries.append(entry)
    if args.require_local_reports and missing_reports:
        raise FileNotFoundError(f"{len(missing_reports)} HLS reports are missing")
    status = {
        "schema_version": 1,
        "artifact_type": "zcu104_mobilenetA_hls_status_from_formal_lut",
        "source": {
            "formal_lut": str(lut_path),
            "formal_lut_sha256": sha256_file(lut_path),
            "formal_lut_payload_sha256": payload.get("payload_sha256"),
        },
        "claim_boundary": (
            "Status bridge for candidate-HLS report assembly only; it reuses "
            "existing HLS reports from the formal operator LUT and is not a "
            "new synthesis, route, board, or measured-power result."
        ),
        "entries": entries,
        "entry_count": len(entries),
        "measured_count": sum(entry["status"] == "measured" for entry in entries),
        "missing_reports": missing_reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "entry_count": status["entry_count"],
                "measured_count": status["measured_count"],
                "missing_report_count": len(missing_reports),
            },
            indent=2,
        )
    )
    return 0 if not missing_reports else 2


if __name__ == "__main__":
    raise SystemExit(main())
