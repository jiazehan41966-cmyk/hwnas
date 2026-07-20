#!/usr/bin/env python3
"""Resolve candidate HLS cases, synthesize missing cases, and build HLS Pareto."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
BUILDER_SCRIPTS = REPO_ROOT / "hls_lut_builder" / "scripts"
for path in (SRC_ROOT, BUILDER_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import build_cases, find_first_existing_report, load_config as load_builder_config
from hwnas_fpga.hardware.hls_evidence import (
    assemble_candidate_hls_report,
    candidate_roles,
    load_status_index,
    normalized_op_key,
)
from hwnas_fpga.runtime import (
    apply_operator_policies_to_search_space,
    build_constraints,
    build_cost_estimator,
    build_hardware_spec,
    build_search_space,
    load_config,
)
from hwnas_fpga.training import load_architecture_from_artifact


DEFAULT_STATUS = (
    REPO_ROOT / "hls_lut_builder" / "results" / "formal_lut_status_strict40_v1.json"
)


def desired_case_kind(role: dict[str, Any]) -> str:
    op = str(role["op_spec"]["op"])
    if role["role"] == "stem":
        return "stem_conv_k3_s2"
    if role["role"] == "head_conv":
        return "pw_conv"
    return op


def operator_template(kind: str) -> tuple[str, str]:
    templates = REPO_ROOT / "hls_lut_builder" / "templates"
    if kind == "stem_conv_k3_s2":
        return str(templates / "stem_conv_k3_s2.cpp.tmpl"), "stem_conv_k3_s2_kernel"
    if kind == "pw_conv" or kind == "conv":
        return str(templates / "pw_conv.cpp.tmpl"), "pw_conv_kernel"
    if kind.startswith("mbconv_e"):
        return str(templates / "mbconv.cpp.tmpl"), f"{kind}_kernel"
    if kind == "global_avg_pool":
        return str(templates / "global_avg_pool.cpp.tmpl"), "global_avg_pool_kernel"
    if kind == "fc_layer":
        return str(templates / "fc_layer.cpp.tmpl"), "fc_layer_kernel"
    raise ValueError(f"No HLS template for missing case kind {kind}")


def generate_missing_config(
    missing_roles: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> Path:
    base = load_builder_config(
        REPO_ROOT / "hls_lut_builder" / "configs" / "candidate_kernels.yaml"
    )
    shapes: dict[str, Any] = {}
    operators: dict[str, Any] = {}
    unique: dict[str, dict[str, Any]] = {}
    for role in missing_roles:
        unique.setdefault(normalized_op_key(role["op_spec"]), role)
    for index, role in enumerate(unique.values()):
        spec = role["op_spec"]
        kind = desired_case_kind(role)
        template, top_function = operator_template(kind)
        shape_name = f"shape_{index:04d}"
        operator_name = f"candidate_case_{index:04d}"
        resolution = int(spec["input_resolution"][0])
        stride = int(spec["stride"])
        shapes[shape_name] = {
            "feature_h": resolution,
            "feature_w": resolution,
            "in_channels": int(spec["in_channels"]),
            "out_channels": int(spec["out_channels"]),
            "stride": stride,
            "padding": int(spec["kernel_size"]) // 2,
            "out_h": max(1, (resolution + stride - 1) // stride),
            "out_w": max(1, (resolution + stride - 1) // stride),
            "op_spec": dict(spec),
        }
        operators[operator_name] = {
            "enabled": True,
            "template": template,
            "top_function": top_function,
            "op_type": kind,
            "shape_refs": [shape_name],
            "implementation_refs": ["baseline_pi1_po1_u1"],
            "clock_profile_refs": ["main_5ns"],
            "parameters": {
                "kernel_size": int(spec["kernel_size"]),
                "padding": int(spec["kernel_size"]) // 2,
                "expand_ratio": int(spec["expand_ratio"]),
                "apply_relu": 0 if kind == "fc_layer" else 1,
            },
            "op_spec_defaults": dict(spec),
        }
    config = {
        "toolchain": base["toolchain"],
        "workspace": {
            "project_root": str(
                REPO_ROOT
                / "hls_lut_builder"
                / "results"
                / "candidate_hls_cache"
            ),
            "report_archive": str(output_dir / "report_archive"),
            "csv_path": str(output_dir / "hls.csv"),
            "manifest_path": str(output_dir / "hls_manifest.yaml"),
            "synthesis_summary_json": str(output_dir / "synthesis_summary.json"),
            "parse_summary_json": str(output_dir / "parse_summary.json"),
        },
        "measurement_contract": base["measurement_contract"],
        "defaults": base["defaults"],
        "clock_profiles": {"main_5ns": base["clock_profiles"]["main_5ns"]},
        "implementation_profiles": {
            "baseline_pi1_po1_u1": base["implementation_profiles"][
                "baseline_pi1_po1_u1"
            ]
        },
        "representative_shapes": shapes,
        "operators": operators,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "missing_hls_cases.yaml"
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def cache_status(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = load_builder_config(config_path)
    entries = []
    for case in build_cases(config):
        report = find_first_existing_report(case.report_candidates)
        entries.append(
            {
                "case_name": case.case_name,
                "status": "measured" if report is not None else "missing",
                "op_type": case.operator,
                "op_spec": case.op_spec,
                "report_path": str(report.resolve()) if report else None,
            }
        )
    payload = {
        "metadata": {"source": str(config_path.resolve())},
        "entries": entries,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def merge_status(*payloads: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for entry in payload.get("entries", []):
            key = normalized_op_key(entry["op_spec"])
            existing = index.get(key)
            if existing is None or (
                existing.get("status") != "measured"
                and entry.get("status") == "measured"
            ):
                index[key] = dict(entry)
    return index


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    metrics = ("latency_ms", "dsp", "lut", "bram")
    left_values = left["aggregate_hls"]
    right_values = right["aggregate_hls"]
    no_worse = all(float(left_values[key]) <= float(right_values[key]) for key in metrics)
    strictly_better = any(float(left_values[key]) < float(right_values[key]) for key in metrics)
    return no_worse and strictly_better


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument(
        "--config",
        default="configs/search/nksid_fpga_search_mobile_anchor_av7k325.yaml",
    )
    parser.add_argument("--status", default=str(DEFAULT_STATUS))
    parser.add_argument(
        "--output-dir",
        default="results/candidate_hls_shortlist",
    )
    parser.add_argument("--run-synthesis", action="store_true")
    parser.add_argument("--vitis-hls", default=None)
    parser.add_argument("--timeout-minutes", type=int, default=30)
    args = parser.parse_args()

    config = load_config(args.config)
    constraints = build_constraints(config)
    hardware = build_hardware_spec(config)
    space = build_search_space(
        config,
        image_size=config["dataset"]["image_size"],
        input_channels=config["dataset"]["input_channels"],
        num_classes=config["dataset"]["num_classes"],
        constraints=constraints,
    )
    estimator = build_cost_estimator(
        config,
        hardware_spec=hardware,
        constraints=constraints,
    )
    space, policy = apply_operator_policies_to_search_space(
        space,
        estimator.operator_policies,
    )
    space = space.pre_prune(estimator)

    original_status_payload = json.loads(
        Path(args.status).read_text(encoding="utf-8-sig")
    )
    original_index = load_status_index(args.status)
    candidates = []
    missing_roles = []
    for candidate_value in args.candidate:
        path = Path(candidate_value)
        architecture = load_architecture_from_artifact(path)
        roles = candidate_roles(architecture, space)
        for role in roles:
            if not role["required_hls"]:
                continue
            entry = original_index.get(normalized_op_key(role["op_spec"]))
            if entry is None or entry.get("status") != "measured":
                missing_roles.append(role)
        candidates.append((path, architecture))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_config = generate_missing_config(
        missing_roles,
        output_dir=output_dir,
    )
    cache_status_path = output_dir / "candidate_hls_cache_status.json"
    if args.run_synthesis and missing_roles:
        command = [
            sys.executable,
            str(BUILDER_SCRIPTS / "run_synthesis.py"),
            "--config",
            str(missing_config),
            "--timeout-minutes",
            str(args.timeout_minutes),
        ]
        if args.vitis_hls:
            command += ["--vitis-hls", args.vitis_hls]
        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    cache_payload = cache_status(missing_config, cache_status_path)
    status_index = merge_status(original_status_payload, cache_payload)

    reports = []
    report_dir = output_dir / "candidate_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    for path, architecture in candidates:
        report = assemble_candidate_hls_report(
            candidate_path=path,
            architecture=architecture,
            search_space=space,
            status_index=status_index,
            repo_root=REPO_ROOT,
        )
        report_path = report_dir / f"{path.stem}.candidate_hls_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path.resolve())
        reports.append(report)

    complete = [report for report in reports if report["evidence_complete"]]
    pareto = [
        report
        for report in complete
        if not any(
            dominates(other, report)
            for other in complete
            if other is not report
        )
    ]
    summary = {
        "schema_version": 1,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "evidence_source": "hls_estimate",
        "claim_boundary": "HLS Pareto is not post-route or COM5 evidence.",
        "operator_policy": policy,
        "candidate_count": len(reports),
        "evidence_complete_count": len(complete),
        "missing_unique_case_count": len(
            {normalized_op_key(role["op_spec"]) for role in missing_roles}
        ),
        "missing_hls_config": str(missing_config.resolve()),
        "cache_status": str(cache_status_path.resolve()),
        "reports": reports,
        "hls_hardware_pareto": [
            {
                "candidate_path": report["candidate_path"],
                "report_path": report["report_path"],
                "aggregate_hls": report["aggregate_hls"],
            }
            for report in pareto
        ],
        "g2_hls_coverage_pass": bool(reports) and len(complete) == len(reports),
    }
    (output_dir / "shortlist_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate_count": len(reports),
                "complete": len(complete),
                "missing_unique_cases": summary["missing_unique_case_count"],
                "g2_hls_coverage_pass": summary["g2_hls_coverage_pass"],
                "output": str((output_dir / "shortlist_summary.json").resolve()),
            },
            indent=2,
        )
    )
    return 0 if summary["g2_hls_coverage_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
