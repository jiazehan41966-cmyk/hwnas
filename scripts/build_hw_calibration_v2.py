#!/usr/bin/env python3
"""Build evidence-tiered hardware calibration without mixing measurement levels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.hardware import FPGACostEstimator, parse_hls_report
from hwnas_fpga.hardware.calibration_v2 import (
    HARDWARE_METRICS,
    RESOURCE_METRICS,
    architecture_family,
    canonical_sha256,
    deduplicate_pairs,
    evidence_fingerprint,
    fit_ratio_model,
    fit_robust_affine,
    validate_affine_independent,
    validate_ratio_model,
    validate_ratio_model_independent,
    validation_gate,
)
from hwnas_fpga.interfaces import HardwareSpec
from hwnas_fpga.search_space import ResolvedBlockSpec


DEFAULT_LEGACY = REPO_ROOT / "artifacts" / "hw_surrogate_calibration" / "hw_surrogate_calibration.json"
DEFAULT_STRICT_LUT = REPO_ROOT / "hls_lut_builder" / "results" / "formal_lut_strict40_v1.json"
DEFAULT_STRICT_STATUS = REPO_ROOT / "hls_lut_builder" / "results" / "formal_lut_status_strict40_v1.json"
DEFAULT_THREE_TIER = REPO_ROOT / "hls_lut_builder" / "results" / "three_tier_operator_summary.csv"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "hw_surrogate_calibration_v2"
DEFAULT_FULL_NETWORK = DEFAULT_OUTPUT / "full_network_evidence.jsonl"
DEFAULT_HLS_SHORTLIST = (
    REPO_ROOT / "results" / "calibration_probe_hls_shortlist" / "shortlist_summary.json"
)

NETWORK_BUDGETS = {
    "latency_ms": 50.0,
    "dsp": 840.0,
    "lut": 203_800.0,
    "bram": 445.0,
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_jsonl_if_present(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(payload)
    return rows


def full_network_tier(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit on historical mainline networks and validate only frozen probes."""

    fingerprints = [str(row.get("fingerprint", "")) for row in rows]
    if any(not value for value in fingerprints):
        raise ValueError("full-network evidence requires a fingerprint")
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("full-network evidence contains duplicate fingerprints")
    if any(row.get("family") != "mainline_mbconv_skip" for row in rows):
        raise ValueError("semantic-mismatch operators cannot enter mainline calibration")

    training = [row for row in rows if row.get("independent_probe") is not True]
    probes = [row for row in rows if row.get("independent_probe") is True]
    resource_models = {}
    resource_validations = {}
    gates = {}
    for metric in RESOURCE_METRICS:
        train_pairs = [
            {
                "fingerprint": row["fingerprint"],
                "family": row["family"],
                "estimated": {metric: (row.get("hls") or {}).get(metric)},
                "measured": {metric: (row.get("route") or {}).get(metric)},
            }
            for row in training
        ]
        probe_pairs = [
            {
                "fingerprint": row["fingerprint"],
                "family": row["family"],
                "estimated": {metric: (row.get("hls") or {}).get(metric)},
                "measured": {metric: (row.get("route") or {}).get(metric)},
            }
            for row in probes
        ]
        model = fit_ratio_model(train_pairs, metric=metric)
        validation = validate_ratio_model_independent(
            model,
            probe_pairs,
            metric=metric,
            budget=NETWORK_BUDGETS[metric],
        )
        resource_models[metric] = model
        resource_validations[metric] = validation
        gates[metric] = validation_gate(metric, validation)

    cycle_training = [
        {
            "fingerprint": row["fingerprint"],
            "estimated": {"cycles": (row.get("hls") or {}).get("composed_cycles")},
            "measured": {"cycles": (row.get("com5") or {}).get("cycles")},
        }
        for row in training
    ]
    cycle_probes = [
        {
            "fingerprint": row["fingerprint"],
            "estimated": {"cycles": (row.get("hls") or {}).get("composed_cycles")},
            "measured": {"cycles": (row.get("com5") or {}).get("cycles")},
        }
        for row in probes
    ]
    latency_model = fit_robust_affine(cycle_training)
    latency_validation = validate_affine_independent(
        latency_model,
        cycle_probes,
        budget=NETWORK_BUDGETS["latency_ms"] * 200_000.0,
    )
    gates["latency_ms"] = validation_gate("latency_ms", latency_validation)
    return {
        "total_unique_networks": len(rows),
        "fit_networks": len(training),
        "independent_probe_networks": len(probes),
        "resource_models": resource_models,
        "latency_model": latency_model,
        "independent_validation": {
            **resource_validations,
            "latency_ms": latency_validation,
        },
        "gates": gates,
        "required_identity_fields": [
            "canonical_architecture_sha256",
            "operator_parallelism_profile",
            "int8_config",
            "fpga_part",
            "target_clock_mhz",
            "tool_versions",
            "harness_version",
        ],
    }


def op_key(op_spec: dict[str, Any]) -> str:
    normalized = dict(op_spec)
    normalized["target_clock_mhz"] = None
    return canonical_sha256(normalized)


def find_hls_report(case_name: str) -> Path | None:
    roots = [
        REPO_ROOT / "hls_lut_builder" / "results" / "p" / case_name,
        REPO_ROOT / "hls_lut_builder" / "results" / "strict40_expansion_p" / case_name,
    ]
    for root in roots:
        candidates = sorted(root.glob("project/*/syn/report/csynth.xml"))
        if candidates:
            return candidates[0]
        candidates = sorted(root.glob("project/*/syn/report/*_csynth.xml"))
        if candidates:
            return candidates[0]
    return None


def comparable_block(op_spec: dict[str, Any]) -> ResolvedBlockSpec | None:
    raw_op = str(op_spec.get("op", ""))
    if raw_op.startswith("mbconv_e"):
        op = "mbconv"
    elif raw_op in {"conv", "stem_conv_k3_s2", "pw_conv"}:
        op = "conv"
    elif raw_op == "skip":
        op = "skip"
    else:
        return None
    resolution = op_spec.get("input_resolution") or [1, 1]
    input_resolution = int(resolution[0])
    stride = int(op_spec.get("stride", 1))
    output_resolution = max(1, math.ceil(input_resolution / stride))
    return ResolvedBlockSpec(
        stage_index=0,
        block_index=0,
        op=op,
        kernel_size=int(op_spec.get("kernel_size", 1)),
        expand_ratio=int(op_spec.get("expand_ratio", 1)),
        stride=stride,
        in_channels=int(op_spec.get("in_channels", 1)),
        out_channels=int(op_spec.get("out_channels", 1)),
        input_resolution=input_resolution,
        output_resolution=output_resolution,
    )


def analytic_metrics(op_spec: dict[str, Any]) -> dict[str, float] | None:
    block = comparable_block(op_spec)
    if block is None:
        return None
    estimator = FPGACostEstimator(
        HardwareSpec(
            name="alinx_av7k325",
            clock_mhz=200,
            max_lut=203_800,
            max_bram=445,
            max_dsp=840,
        ),
        quantization_bits=8,
    )
    cost = estimator._estimate_block(block)  # intentional calibration probe
    return {
        "latency_ms": cost.latency_cycles / 200_000.0,
        "dsp": float(cost.allocated_dsp),
        "lut": float(cost.lut),
        "bram": float(cost.bram_blocks),
    }


def hls_metrics(report_path: Path) -> dict[str, float]:
    parsed = parse_hls_report(report_path)
    cycles = float(parsed["cycles"])
    bram18 = float(parsed.get("bram_18k", parsed.get("bram", 0)))
    return {
        "latency_ms": cycles / 200_000.0,
        "cycles": cycles,
        "dsp": float(parsed["dsp"]),
        "lut": float(parsed["lut"]),
        "bram": bram18 / 2.0,
    }


def load_strict40_evidence(
    lut_path: Path,
    status_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    lut = read_json(lut_path)
    status = read_json(status_path)
    formal_by_key = {
        op_key(entry["op_spec"]): entry for entry in lut.get("entries", [])
    }
    evidence: list[dict[str, Any]] = []
    analytic_hls: list[dict[str, Any]] = []
    hls_route: list[dict[str, Any]] = []
    for entry in status.get("entries", []):
        if entry.get("status") != "measured":
            continue
        op_spec = dict(entry["op_spec"])
        fingerprint = evidence_fingerprint(
            {"case_name": entry["case_name"]},
            operator_profile="baseline_pi1_po1_u1",
            harness_version="packed_stream_v1",
        )
        report_path = find_hls_report(str(entry["case_name"]))
        hls = hls_metrics(report_path) if report_path is not None else None
        formal = formal_by_key.get(op_key(op_spec))
        route = None
        if formal is not None:
            route = {
                "latency_ms": float(formal["latency_ms"]),
                "dsp": float(formal["dsp"]),
                "lut": float(formal["lut"]),
                "bram": float(formal["bram"]),
            }
        analytic = analytic_metrics(op_spec)
        family = (
            "semantic_mismatch"
            if str(op_spec.get("op")) in {"denoise", "edge", "mixconv"}
            else str(op_spec.get("op", "unknown")).split("_e", 1)[0]
        )
        row = {
            "fingerprint": fingerprint,
            "case_name": entry["case_name"],
            "family": family,
            "op_spec": op_spec,
            "analytic": analytic,
            "hls": hls,
            "formal_lut_measured": route,
            "hls_report": str(report_path.relative_to(REPO_ROOT)) if report_path else None,
            "hls_report_sha256": (
                canonical_sha256(report_path.read_text(encoding="utf-8", errors="replace"))
                if report_path
                else None
            ),
        }
        evidence.append(row)
        if analytic is not None and hls is not None:
            analytic_hls.append(
                {
                    "fingerprint": fingerprint,
                    "family": family,
                    "source": row["hls_report"],
                    "estimated": analytic,
                    "measured": hls,
                }
            )
        if hls is not None and route is not None:
            hls_route.append(
                {
                    "fingerprint": fingerprint,
                    "family": family,
                    "source": entry["case_name"],
                    "estimated": hls,
                    "measured": route,
                }
            )
    return evidence, analytic_hls, hls_route


def load_three_tier(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hls_route: list[dict[str, Any]] = []
    hls_board_cycles: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            op_type = str(row["op_type"])
            family = (
                "semantic_mismatch"
                if op_type in {"denoise", "edge", "mixconv"}
                else op_type.split("_e", 1)[0]
            )
            identity = {"case_name": row.get("case_name")}
            fingerprint = evidence_fingerprint(
                identity,
                operator_profile="baseline_pi1_po1_u1",
                harness_version=str(row.get("measurement_contract_id", "unknown")),
            )
            if row.get("vivado_status") == "success":
                hls_route.append(
                    {
                        "fingerprint": fingerprint,
                        "family": family,
                        "source": row.get("case_name"),
                        "estimated": {
                            "latency_ms": float(row["hls_latency_ms"]),
                            "dsp": float(row["hls_dsp"]),
                            "lut": float(row["hls_lut"]),
                            "bram": float(row["hls_bram18"]) / 2.0,
                        },
                        "measured": {
                            "dsp": float(row["vivado_post_route_dsp"]),
                            "lut": float(row["vivado_post_route_lut"]),
                            "bram": float(row["vivado_post_route_bram_tile"]),
                        },
                    }
                )
            if row.get("board_measured") == "yes" and row.get("board_cycles"):
                hls_board_cycles.append(
                    {
                        "fingerprint": fingerprint,
                        "family": family,
                        "source": row.get("case_name"),
                        "estimated": {"cycles": float(row["hls_cycles"])},
                        "measured": {"cycles": float(row["board_cycles"])},
                    }
                )
    return hls_route, hls_board_cycles


def load_network_pairs(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = []
    for pair in payload.get("pairs", []):
        candidate_path = REPO_ROOT / str(pair["candidate_source"])
        candidate_payload = read_json(candidate_path)
        candidate = candidate_payload.get("candidate", candidate_payload)
        architecture = candidate.get("encoding") or candidate.get("architecture")
        if not isinstance(architecture, dict):
            continue
        run = str(pair["run"])
        implementation_match = re.search(r"_([0-9a-f]{10,})$", run)
        implementation_profile = (
            implementation_match.group(1) if implementation_match else "unknown"
        )
        rows.append(
            {
                "fingerprint": evidence_fingerprint(
                    architecture,
                    operator_profile=f"full_network:{implementation_profile}",
                    harness_version="single_full_network_latency_only_harness",
                ),
                "family": architecture_family(architecture),
                "run": run,
                "arch_id": pair.get("arch_id"),
                "source": pair.get("candidate_source"),
                "architecture": architecture,
                "estimated": dict(pair["estimated"]),
                "measured": dict(pair["measured"]),
            }
        )
    return deduplicate_pairs(rows)


def summarize_tier(rows: list[dict[str, Any]], budgets: dict[str, float | None]) -> dict[str, Any]:
    models = {}
    validations = {}
    gates = {}
    for metric in HARDWARE_METRICS:
        model = fit_ratio_model(rows, metric=metric)
        validation = validate_ratio_model(
            rows,
            metric=metric,
            budget=budgets.get(metric),
        )
        models[metric] = model
        validations[metric] = validation
        gates[metric] = validation_gate(metric, validation)
    return {
        "unique_pairs": len(rows),
        "models": models,
        "leave_one_group_out_validation": validations,
        "gates": gates,
    }


def fit_affine_cycles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        row for row in rows
        if row.get("estimated", {}).get("cycles") is not None
        and row.get("measured", {}).get("cycles") is not None
    ]
    if len(usable) < 2:
        return {"available": False, "n": len(usable)}
    x = [float(row["estimated"]["cycles"]) for row in usable]
    y = [float(row["measured"]["cycles"]) for row in usable]
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    variance = sum((value - x_mean) ** 2 for value in x)
    slope = (
        sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / variance
        if variance > 0
        else 1.0
    )
    intercept = y_mean - slope * x_mean
    errors = [
        abs((intercept + slope * a) - b) / b for a, b in zip(x, y) if b > 0
    ]
    return {
        "available": True,
        "n": len(usable),
        "intercept_cycles": intercept,
        "slope": slope,
        "mape": statistics.fmean(errors),
        "claim_boundary": "operator-level fit; not a full-network COM5 model",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-network", default=str(DEFAULT_LEGACY))
    parser.add_argument("--strict-lut", default=str(DEFAULT_STRICT_LUT))
    parser.add_argument("--strict-status", default=str(DEFAULT_STRICT_STATUS))
    parser.add_argument("--three-tier", default=str(DEFAULT_THREE_TIER))
    parser.add_argument("--full-network-evidence", default=str(DEFAULT_FULL_NETWORK))
    parser.add_argument("--hls-shortlist-summary", default=str(DEFAULT_HLS_SHORTLIST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    strict_evidence, analytic_hls, strict_hls_route = load_strict40_evidence(
        Path(args.strict_lut),
        Path(args.strict_status),
    )
    three_tier_hls_route, hls_board_cycles = load_three_tier(Path(args.three_tier))
    network_pairs = load_network_pairs(Path(args.legacy_network))
    full_network_rows = read_jsonl_if_present(Path(args.full_network_evidence))
    network_calibration = full_network_tier(full_network_rows)
    shortlist_path = Path(args.hls_shortlist_summary)
    shortlist = read_json(shortlist_path) if shortlist_path.exists() else {}

    # strict40 is status-authoritative. Do not count the compact three-tier
    # snapshot again when it names the same operator case.
    strict_case_names = {str(row.get("source")) for row in strict_hls_route}
    non_overlapping_three_tier = [
        row
        for row in three_tier_hls_route
        if str(row.get("source")) not in strict_case_names
    ]
    hls_route_rows = deduplicate_pairs(
        strict_hls_route + non_overlapping_three_tier
    )
    analytic_hls_rows = deduplicate_pairs(analytic_hls)
    mainline_network = [
        row for row in network_pairs if row["family"] == "mainline_mbconv_skip"
    ]
    semantic_mismatch_network = [
        row for row in network_pairs if row["family"] != "mainline_mbconv_skip"
    ]

    tiers = {
        "analytic_to_hls_operator": summarize_tier(
            analytic_hls_rows,
            {metric: None for metric in HARDWARE_METRICS},
        ),
        "hls_to_post_route_operator": summarize_tier(
            hls_route_rows,
            {metric: None for metric in HARDWARE_METRICS},
        ),
        "legacy_analytic_to_route_com5_network_diagnostic": summarize_tier(
            mainline_network,
            NETWORK_BUDGETS,
        ),
        "hls_to_board_cycles_operator": fit_affine_cycles(hls_board_cycles),
    }
    diagnostic_mainline_gates = tiers[
        "legacy_analytic_to_route_com5_network_diagnostic"
    ]["gates"]
    independent_gates = network_calibration["gates"]
    external_probe_gate = {
        "required": 4,
        "completed": network_calibration["independent_probe_networks"],
        "pass": network_calibration["independent_probe_networks"] == 4,
        "reason": (
            "exactly four frozen semantic-safe full-network probes must be "
            "routed and measured without entering the fit"
        ),
    }
    evidence_count_gate = network_calibration["total_unique_networks"] >= 8
    hls_coverage_gate = (
        shortlist.get("g2_hls_coverage_pass") is True
        and int(shortlist.get("evidence_complete_count", 0))
        == int(shortlist.get("candidate_count", -1))
        and int(shortlist.get("candidate_count", 0)) > 0
    )
    operational_gates = {}
    prerequisites = external_probe_gate["pass"] and evidence_count_gate
    for metric in HARDWARE_METRICS:
        gate = independent_gates[metric]
        if prerequisites:
            operational_gates[metric] = gate
        else:
            operational_gates[metric] = {
                **gate,
                "hard_screening_enabled": False,
                "mode": "pass_through_to_hls",
                "reasons": list(gate.get("reasons", []))
                + ["independent full-network evidence gate incomplete"],
            }
    interval_gate = prerequisites and all(
        gate["hard_screening_enabled"] for gate in independent_gates.values()
    )
    g2_pass = interval_gate and hls_coverage_gate
    blockers = []
    if not external_probe_gate["pass"]:
        blockers.append("four frozen independent full-network probes are incomplete")
    if not evidence_count_gate:
        blockers.append("fewer than eight unique semantic-safe full-network samples")
    if not interval_gate:
        blockers.append("independent interval-screening quality gates are not all PASS")
    if not hls_coverage_gate:
        blockers.append("candidate HLS shortlist coverage is not 100%")
    payload = {
        "schema_version": 2,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "target": {
            "fpga_part": "xc7k325t-ffg900-2",
            "clock_mhz": 200.0,
            "bit_width": 8,
            "budgets": NETWORK_BUDGETS,
        },
        "legacy_calibration_v1": {
            "path": str(Path(args.legacy_network).resolve()),
            "status": "diagnostic_only_not_allowed_for_hard_screening",
        },
        "evidence_counts": {
            "strict40_measured_rows": len(strict_evidence),
            "strict40_analytic_hls_comparable": len(analytic_hls_rows),
            "hls_route_unique_operator_rows": len(hls_route_rows),
            "legacy_network_raw_rows": len(read_json(Path(args.legacy_network)).get("pairs", [])),
            "legacy_network_unique_rows": len(network_pairs),
            "mainline_network_unique_rows": len(mainline_network),
            "semantic_mismatch_network_unique_rows": len(semantic_mismatch_network),
        },
        "tiers": tiers,
        "full_network_calibration": network_calibration,
        "network_rows": mainline_network,
        "semantic_mismatch_rows": semantic_mismatch_network,
        "strict40_evidence": strict_evidence,
        "external_probe_gate": external_probe_gate,
        "screening_policy": {
            "gates": operational_gates,
            "diagnostic_in_sample_gates": diagnostic_mainline_gates,
            "independent_probe_gates": independent_gates,
            "rule": (
                "certified_reject only when a validated optimistic lower bound "
                "exceeds budget; failed gates pass through to HLS"
            ),
            "power_energy": "not_measured_not_an_objective",
        },
        "hls_shortlist": {
            "path": str(shortlist_path.resolve()),
            "coverage_pass": hls_coverage_gate,
            "candidate_count": shortlist.get("candidate_count", 0),
            "evidence_complete_count": shortlist.get("evidence_complete_count", 0),
        },
        "g2_pass": g2_pass,
        "g2_blockers": blockers,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "calibration_v2.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "screening_policy_v2.json").write_text(
        json.dumps(payload["screening_policy"], indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Hardware calibration v2",
        "",
        f"- G2 pass: `{payload['g2_pass']}`",
        f"- strict40 measured rows: `{len(strict_evidence)}`",
        f"- unique network rows: `{len(network_pairs)}`",
        f"- unique mainline network rows: `{len(mainline_network)}`",
        "- denoise/edge/mixconv are excluded from the semantic-safe mainline.",
        "",
        "| tier | unique n | metric | MAPE | P90 APE | Spearman | hard screen |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for tier_name, tier in tiers.items():
        if "leave_one_group_out_validation" not in tier:
            continue
        for metric, validation in tier["leave_one_group_out_validation"].items():
            gate = tier["gates"][metric]
            lines.append(
                f"| {tier_name} | {tier['unique_pairs']} | {metric} "
                f"| {validation.get('mape', 0):.1%} "
                f"| {validation.get('p90_ape', 0):.1%} "
                f"| {validation.get('spearman')} "
                f"| {gate['hard_screening_enabled']} |"
            )
    lines += ["", "## G2 blockers", ""]
    lines.extend(f"- {item}" for item in payload["g2_blockers"])
    (output_dir / "calibration_v2.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(json_path),
                "g2_pass": False,
                "counts": payload["evidence_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
