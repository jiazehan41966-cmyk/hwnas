"""Bit-exact INT8 simulator-to-HLS testbench parity evidence gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


REQUIRED_INPUT_KINDS = ("real_sample", "boundary_tensor", "random_tensor")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_parity_records(
    records: Iterable[Mapping[str, Any]],
    *,
    quantization_contract: str,
) -> dict[str, Any]:
    rows = [dict(row) for row in records]
    kinds = {str(row.get("input_kind", "")) for row in rows}
    layers = {str(row.get("layer", "")) for row in rows if row.get("layer")}
    total_elements = sum(int(row.get("element_count", 0)) for row in rows)
    total_mismatches = sum(int(row.get("mismatch_count", -1)) for row in rows)
    invalid_rows = [
        index
        for index, row in enumerate(rows)
        if int(row.get("element_count", 0)) <= 0
        or int(row.get("mismatch_count", -1)) < 0
        or len(str(row.get("simulator_sha256", ""))) != 64
        or len(str(row.get("hls_testbench_sha256", ""))) != 64
        or (
            quantization_contract == "per_tensor_symmetric_int8_v2"
            and len(str(row.get("quantization_spec_sha256", ""))) != 64
        )
    ]
    gates = {
        "quantization_contract": (
            quantization_contract in {
                "per_tensor_symmetric_int8_v1",  # legacy audit compatibility
                "per_tensor_symmetric_int8_v2",
            }
        ),
        "records_present": bool(rows),
        "real_samples_present": "real_sample" in kinds,
        "boundary_tensors_present": "boundary_tensor" in kinds,
        "random_tensors_present": "random_tensor" in kinds,
        "layer_coverage_present": bool(layers),
        "quantization_spec_trace_present": (
            quantization_contract != "per_tensor_symmetric_int8_v2"
            or all(len(str(row.get("quantization_spec_sha256", ""))) == 64 for row in rows)
        ),
        "record_schema_valid": not invalid_rows,
        "compared_elements_positive": total_elements > 0,
        "zero_integer_mismatch": total_mismatches == 0,
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "gate": "G4_hls_integer_parity",
        "status": "PASS" if passed else "FAIL",
        "overall_pass": passed,
        "quantization_contract": quantization_contract,
        "gates": gates,
        "record_count": len(rows),
        "layers": sorted(layers),
        "input_kinds": sorted(kinds),
        "compared_element_count": total_elements,
        "mismatch_count": total_mismatches,
        "invalid_record_indices": invalid_rows,
        "claim_boundary": (
            "PASS permits creation of a board-validation run. It does not "
            "claim board parity or validation accuracy until board_validation.jsonl "
            "is complete with zero per-sample disagreement."
        ),
    }


def audit_parity_jsonl(path: str | Path, *, quantization_contract: str) -> dict[str, Any]:
    source = Path(path).resolve()
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{source}:{line_number} must be a JSON object")
            rows.append(payload)
    result = audit_parity_records(
        rows,
        quantization_contract=quantization_contract,
    )
    result["records_path"] = str(source)
    result["records_sha256"] = sha256_file(source)
    return result
