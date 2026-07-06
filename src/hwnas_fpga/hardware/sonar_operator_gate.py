"""Admission audit for denoise/edge after matched INT8 and HLS evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REQUIRED_VARIANTS = ("mbconv_control", "denoise", "edge", "denoise_edge")
REQUIRED_FOLDS = (0, 1, 2, 3, 4)
REQUIRED_SEEDS = (42, 43, 44)


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm family-wise adjusted p-values in original order."""

    count = len(p_values)
    indexed = sorted(
        ((index, max(0.0, min(1.0, float(value)))) for index, value in enumerate(p_values)),
        key=lambda item: item[1],
    )
    adjusted = [1.0] * count
    running = 0.0
    for rank, (index, value) in enumerate(indexed):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _ratio_error(reference: float, candidate: float) -> float:
    if reference <= 0:
        return float("inf")
    return abs(candidate - reference) / reference


def _complete_protocol(row: Mapping[str, Any]) -> bool:
    folds = sorted(int(value) for value in row.get("folds", []))
    seeds = sorted(int(value) for value in row.get("seeds", []))
    return (
        folds == list(REQUIRED_FOLDS)
        and seeds == list(REQUIRED_SEEDS)
        and int(row.get("completed_runs", 0)) == 15
        and row.get("claimable") is True
        and row.get("outer_leakage") is False
    )


def audit_sonar_operator_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the G5 gate without manufacturing missing experimental evidence."""

    variants = manifest.get("ablation_variants", {})
    variant_protocol = {
        name: _complete_protocol(variants.get(name, {}))
        for name in REQUIRED_VARIANTS
    }
    comparisons = manifest.get("comparisons_vs_control", {})
    comparison_names = ("denoise", "edge", "denoise_edge")
    raw_p = [
        float((comparisons.get(name) or {}).get("p_value", 1.0))
        for name in comparison_names
    ]
    adjusted_p = holm_adjust(raw_p)
    adjusted = dict(zip(comparison_names, adjusted_p))

    operator_results: dict[str, Any] = {}
    for operator in ("denoise", "edge"):
        evidence = (manifest.get("operators") or {}).get(operator, {})
        software_spec = str(evidence.get("software_spec_sha256", ""))
        hls_spec = str(evidence.get("hls_spec_sha256", ""))
        parity = evidence.get("parity", {})
        matching = evidence.get("matched_control", {})
        hls = evidence.get("hls", {})
        comparison = comparisons.get(operator, {})
        gates = {
            "single_quantization_spec": (
                len(software_spec) == 64
                and software_spec == hls_spec
                and evidence.get("quantization_contract") == "per_tensor_symmetric_int8_v1"
            ),
            "weight_export_complete": (
                evidence.get("weight_export_complete") is True
                and len(str(evidence.get("weight_export_sha256", ""))) == 64
            ),
            "real_sample_parity": int(parity.get("real_sample_count", 0)) > 0,
            "boundary_tensor_parity": int(parity.get("boundary_tensor_count", 0)) > 0,
            "random_tensor_parity": int(parity.get("random_tensor_count", 0)) > 0,
            "bit_exact_zero_mismatch": (
                int(parity.get("compared_element_count", 0)) > 0
                and int(parity.get("mismatch_count", -1)) == 0
            ),
            "output_shape_matched": matching.get("output_shape_equal") is True,
            "parameter_count_within_5pct": _ratio_error(
                float(matching.get("control_parameters", 0)),
                float(matching.get("operator_parameters", 0)),
            )
            <= 0.05,
            "macs_within_5pct": _ratio_error(
                float(matching.get("control_macs", 0)),
                float(matching.get("operator_macs", 0)),
            )
            <= 0.05,
            "four_way_ablation_complete": all(variant_protocol.values()),
            "paired_stratified_bootstrap": (
                comparison.get("method") == "paired_stratified_bootstrap"
                and int(comparison.get("iterations", 0)) >= 10_000
            ),
            "holm_significant": adjusted[operator] < 0.05,
            "macro_f1_actual_gain": float(
                comparison.get("macro_f1_mean_delta", 0.0)
            )
            > 0.0,
            "hls_evidence_complete": hls.get("evidence_complete") is True,
            "hls_feasible": hls.get("route_feasible") is True,
        }
        passed = all(gates.values())
        operator_results[operator] = {
            "status": "ADMITTED" if passed else "PAUSED",
            "may_reenter_search_space": passed,
            "gates": gates,
            "holm_adjusted_p_value": adjusted[operator],
        }

    overall = all(
        result["may_reenter_search_space"] for result in operator_results.values()
    )
    return {
        "schema_version": 1,
        "gate": "G5_sonar_operator_admission",
        "status": "PASS" if overall else "BLOCKED",
        "overall_pass": overall,
        "required_variants": list(REQUIRED_VARIANTS),
        "required_folds": list(REQUIRED_FOLDS),
        "required_seeds": list(REQUIRED_SEEDS),
        "variant_protocol_gates": variant_protocol,
        "holm_adjusted_p_values": adjusted,
        "operators": operator_results,
        "manifest_sha256": canonical_sha256(manifest),
        "boundary": (
            "Route/COM5 evidence for the historical simplified sonar kernels "
            "does not satisfy this gate. Re-entry requires matched INT8 semantics, "
            "zero-mismatch parity, a matched MBConv control, claimable ablation, "
            "and HLS feasibility."
        ),
    }


def audit_sonar_operator_manifest_path(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("sonar operator manifest must be a JSON object")
    return audit_sonar_operator_manifest(payload)
