#!/usr/bin/env python3
"""Plan, build, and measure an isolated e2e harness for an RL best candidate.

This script intentionally writes under sonar_classifier_rl_best_e2e_board and
does not update the canonical current84/fullcombo/arch84 ledgers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
BOARD_HARNESS_DIR = ROOT / "hls_lut_builder" / "board_harness"
RESULT_ROOT = BOARD_HARNESS_DIR / "results" / "sonar_classifier_rl_best_e2e_board"
REFERENCE_TRACE = (
    BOARD_HARNESS_DIR
    / "results"
    / "sonar_classifier_end_to_end_board"
    / "sonar_classifier_e2e_architecture_trace.csv"
)
REFERENCE_SUMMARY = (
    BOARD_HARNESS_DIR
    / "results"
    / "sonar_classifier_end_to_end_board"
    / "sonar_classifier_e2e_summary.json"
)

sys.path.insert(0, str(SCRIPT_DIR))

from batch_measure_ops import DEFAULT_BOARD_CONFIG, DEFAULT_VIVADO, build_bitstream  # noqa: E402
import current84_fullcombo_board_validation as fullcombo  # noqa: E402
import export_retrained_weights_to_e2e_mem as trained_export  # noqa: E402
import sonar_classifier_e2e_board_planning as planning  # noqa: E402
import sonar_classifier_e2e_board_validation as e2e  # noqa: E402
import sonar_classifier_e2e_variant_build as variant  # noqa: E402


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


TRACE_FIELDS = [
    "layer_index",
    "layer_role",
    "op",
    "kernel_size",
    "expand_ratio",
    "input_h",
    "output_h",
    "input_channels",
    "output_channels",
    "stride",
    "expected_current84_combo_query",
    "current84_combo_case_name",
    "current84_mapping_status",
    "board_measurement_namespace",
    "combo_decomposition",
    "component_roles",
    "component_case_names",
    "fullcombo_board_status",
    "fullcombo_uart_status_code",
    "fullcombo_wns_ns",
    "fullcombo_timing_clean",
    "prediction_baseline_cycles",
    "prediction_baseline_latency_ms",
    "notes",
]

READY_MAPPING_STATUSES = {
    "mapped_to_current84_fullcombo",
    "mapped_to_arch84_extension",
    "mapped_to_phase0_v4_sonar_direct",
}

PHASE0_V4_SONAR_STAGE3_K3_DIRECT_CASES = {
    "denoise": "denoise_sonar_stage3_28_32_32_s1_baseline_pi1_po1_u1_main_5ns",
    "edge": "edge_sonar_stage3_28_32_32_s1_baseline_pi1_po1_u1_main_5ns",
}
PHASE0_V4_SONAR_STAGE3_K3_CASE_DIR_NAMES = {
    "denoise_sonar_stage3_28_32_32_s1_baseline_pi1_po1_u1_main_5ns": "d_s3k3",
    "edge_sonar_stage3_28_32_32_s1_baseline_pi1_po1_u1_main_5ns": "e_s3k3",
}
PHASE0_V4_SONAR_STAGE3_K3_CASE_ROOT = (
    ROOT / "hls_lut_builder" / "results" / "phase0_v4_sonar_stage3_k3_pilot" / "p"
)

IDENTITY_FIELDS = [
    "layer_role",
    "op",
    "kernel_size",
    "expand_ratio",
    "input_h",
    "output_h",
    "input_channels",
    "output_channels",
    "stride",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated RL-best full-network e2e board validation"
    )
    parser.add_argument("mode", choices=("plan", "generate", "build", "measure", "all", "refresh"))
    parser.add_argument("--candidate-path", type=str, help="RL best_candidate.json")
    parser.add_argument("--spec-path", type=str, help="Previously generated harness spec")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--result-root", default=str(RESULT_ROOT))
    parser.add_argument("--board-config", default=str(DEFAULT_BOARD_CONFIG))
    parser.add_argument("--serial-port", default="COM5")
    parser.add_argument("--vivado", default=str(DEFAULT_VIVADO))
    parser.add_argument("--hw-server-url", default="TCP:localhost:3121")
    parser.add_argument("--hw-device-pattern", default="*xc7k325t*")
    parser.add_argument("--hw-target-index", type=int, default=0)
    parser.add_argument("--program-retries", type=int, default=2)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--uart-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--post-program-delay-seconds", type=float, default=1.0)
    parser.add_argument("--measurement-timeout-minutes", type=int, default=30)
    parser.add_argument("--fifo-depth", type=int, default=64)
    parser.add_argument("--fifo-style", choices=("lut", "lut_registered", "bram"), default="lut_registered")
    parser.add_argument("--timing-variant", default="place_altspread_route_more_global")
    parser.add_argument("--impl-profile", choices=("default", "fanout_only", "fanout_floorplan"), default="default")
    parser.add_argument("--argmax-pipeline", action="store_true")
    parser.add_argument(
        "--dynamic-validation",
        action="store_true",
        help=(
            "Generate the repeatable UART v1 input/output harness instead of the "
            "legacy boot-time fixed-ROM harness."
        ),
    )
    parser.add_argument(
        "--dynamic-uart-baud",
        type=int,
        default=921_600,
        help="Compile-time UART baud for --dynamic-validation.",
    )
    parser.add_argument(
        "--completion-mode",
        choices=("axil_and_sink", "sink_and_argmax"),
        default="axil_and_sink",
    )
    parser.add_argument("--watchdog-cycles", type=int, default=200_000_000)
    parser.add_argument("--bitstream-timeout-minutes", type=int, default=240)
    parser.add_argument("--allow-timing-fail-measurement", action="store_true")
    parser.add_argument("--expected-bitstream-sha256", default="")
    parser.add_argument(
        "--parameter-mode",
        choices=("latency_only_deterministic", "trained_checkpoint_mem"),
        default="latency_only_deterministic",
    )
    parser.add_argument(
        "--trained-checkpoint",
        default="",
        help="final_best_model.pt used when --parameter-mode trained_checkpoint_mem.",
    )
    parser.add_argument(
        "--trained-weight-manifest",
        default="",
        help="Optional output path for trained_weight_manifest.json.",
    )
    parser.add_argument(
        "--component-override",
        action="append",
        default=[],
        help="Implementation-only override in role=case_name form. May be repeated.",
    )
    parser.add_argument(
        "--component-overrides-from-manifest",
        default="",
        help="Existing claimable harness_manifest.json whose role->case_name mapping should be reused.",
    )
    parser.add_argument("--skip-measure", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.mode in {"plan", "generate", "build", "all"} and not args.candidate_path:
        parser.error("--candidate-path is required for plan/generate/build/all")
    if args.mode == "measure" and not (args.run_name or args.spec_path):
        parser.error("measure requires --run-name or --spec-path")
    if (
        args.mode in {"generate", "build", "all"}
        and args.parameter_mode == "trained_checkpoint_mem"
        and not args.trained_checkpoint
    ):
        parser.error("--trained-checkpoint is required with --parameter-mode trained_checkpoint_mem")
    if (
        args.mode in {"generate", "build", "all"}
        and args.dynamic_validation
        and args.parameter_mode != "trained_checkpoint_mem"
    ):
        parser.error(
            "--dynamic-validation requires --parameter-mode trained_checkpoint_mem"
        )
    return args


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    if not value:
        raise ValueError("empty name after sanitization")
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def configure_result_root(args: argparse.Namespace) -> None:
    global RESULT_ROOT
    RESULT_ROOT = Path(args.result_root).expanduser().resolve()


def component_overrides_from_manifest(path: str) -> dict[str, str]:
    if not path:
        return {}
    manifest_path = Path(path).expanduser().resolve()
    manifest = read_json(manifest_path)
    overrides: dict[str, str] = {}
    for component in manifest.get("components", []):
        if not isinstance(component, dict):
            continue
        role = str(component.get("role", "")).strip()
        case_name = str(component.get("case_name", "")).strip()
        if role and case_name:
            overrides[role] = case_name
    if not overrides:
        raise ValueError(f"No component role/case_name overrides found in manifest: {manifest_path}")
    return overrides


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def candidate_payload(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = read_json(path)
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else payload
    if not isinstance(candidate, dict):
        raise ValueError(f"Unsupported candidate payload: {path}")
    architecture = candidate.get("encoding") or candidate.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError(f"Candidate has no architecture/encoding dict: {path}")
    return candidate, architecture


def default_run_name(candidate: dict[str, Any], candidate_path: Path) -> str:
    arch_id = str(candidate.get("arch_id") or candidate_path.stem)
    return sanitize_name(f"{arch_id}_rl_best_e2e")


def run_root(run_name: str) -> Path:
    return RESULT_ROOT / sanitize_name(run_name)


def target_name(run_name: str) -> str:
    return f"sonar_classifier_rl_best_e2e__{sanitize_name(run_name)}"


def project_root(run_name: str) -> Path:
    return run_root(run_name) / "harness_project"


def measurement_json_path(run_name: str) -> Path:
    return run_root(run_name) / "measurements" / f"{target_name(run_name)}.measurement.json"


def normalize_identity(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        item: dict[str, str] = {}
        for field in IDENTITY_FIELDS:
            value = row.get(field, "")
            item[field] = "" if value is None else str(value)
        normalized.append(item)
    return normalized


def identity_gate(trace_rows: list[dict[str, Any]], reference_trace: Path) -> dict[str, Any]:
    reference_rows = read_csv(reference_trace)
    candidate_norm = normalize_identity(trace_rows)
    reference_norm = normalize_identity(reference_rows)
    mismatches: list[dict[str, Any]] = []
    max_len = max(len(candidate_norm), len(reference_norm))
    for idx in range(max_len):
        cand = candidate_norm[idx] if idx < len(candidate_norm) else None
        ref = reference_norm[idx] if idx < len(reference_norm) else None
        if cand != ref:
            mismatches.append({"index": idx + 1, "candidate": cand, "reference": ref})
    return {
        "reference_trace": str(reference_trace),
        "reference_trace_exists": reference_trace.exists(),
        "matches_reference_arch84_e2e": bool(reference_rows) and not mismatches,
        "candidate_layer_count": len(candidate_norm),
        "reference_layer_count": len(reference_norm),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "rule": "All topology fields must match before reusing the fixed arch_84 e2e bitstream.",
    }


def trace_candidate(architecture: dict[str, Any]) -> list[dict[str, Any]]:
    fullcombo_rows = planning.read_csv(planning.FULLCOMBO_RUNLIST)
    fullcombo_status_rows = planning.read_csv(planning.FULLCOMBO_STATUS)
    extension_rows = planning.read_csv(planning.ARCH84_EXTENSION_RUNLIST)
    extension_status_rows = planning.read_csv(planning.ARCH84_EXTENSION_STATUS)
    selection = {"selected": {"architecture": architecture}}
    rows = planning.trace_architecture(
        selection,
        fullcombo_rows,
        fullcombo_status_rows,
        extension_rows,
        extension_status_rows,
    )
    apply_phase0_v4_sonar_stage3_direct_mapping(rows)
    return rows


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_phase0_v4_sonar_stage3_k3_row(row: dict[str, Any]) -> bool:
    op = str(row.get("op") or "")
    if op not in PHASE0_V4_SONAR_STAGE3_K3_DIRECT_CASES:
        return False
    return (
        str(row.get("layer_role") or "") == "stage3_block0"
        and _int_or_none(row.get("kernel_size")) == 3
        and _int_or_none(row.get("expand_ratio")) == 1
        and _int_or_none(row.get("input_h")) == 28
        and _int_or_none(row.get("output_h")) == 28
        and _int_or_none(row.get("input_channels")) == 32
        and _int_or_none(row.get("output_channels")) == 32
        and _int_or_none(row.get("stride")) == 1
    )


def apply_phase0_v4_sonar_stage3_direct_mapping(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if not is_phase0_v4_sonar_stage3_k3_row(row):
            continue
        op = str(row["op"])
        case_name = PHASE0_V4_SONAR_STAGE3_K3_DIRECT_CASES[op]
        notes = str(row.get("notes") or "")
        sonar_note = "phase0_v4_sonar_stage3_k3_direct_mapping"
        row.update(
            {
                "expected_current84_combo_query": case_name,
                "current84_combo_case_name": case_name,
                "current84_mapping_status": "mapped_to_phase0_v4_sonar_direct",
                "board_measurement_namespace": "phase0_v4_sonar_stage3_k3_pilot",
                "combo_decomposition": "direct",
                "component_roles": op,
                "component_case_names": case_name,
                "fullcombo_board_status": "operator_post_route_deployable_not_full_network_measured",
                "notes": f"{notes}; {sonar_note}" if notes else sonar_note,
            }
        )


def resolve_rl_best_case_dir(case_name: str) -> Path:
    if case_name in PHASE0_V4_SONAR_STAGE3_K3_DIRECT_CASES.values():
        candidates = [
            PHASE0_V4_SONAR_STAGE3_K3_CASE_ROOT
            / PHASE0_V4_SONAR_STAGE3_K3_CASE_DIR_NAMES.get(case_name, case_name),
            PHASE0_V4_SONAR_STAGE3_K3_CASE_ROOT / case_name,
        ]
        for candidate in candidates:
            if (candidate / "project" / "solution1" / "impl" / "verilog").exists() or (
                candidate / "project" / "solution1" / "syn" / "verilog"
            ).exists():
                return candidate.resolve()
    return variant.resolve_case_dir(case_name)


def build_harness_spec(
    *,
    run_name: str,
    candidate: dict[str, Any],
    candidate_path: Path,
    architecture: dict[str, Any],
    trace_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    mapping_ok = all(row.get("current84_mapping_status") in READY_MAPPING_STATUSES for row in trace_rows)
    missing_components = [
        row
        for row in trace_rows
        if not row.get("component_case_names") or not row.get("component_roles")
    ]
    status = "ready_to_generate" if mapping_ok and not missing_components else "not_generated_mapping_incomplete"
    return {
        "target_name": target_name(run_name),
        "run_name": run_name,
        "harness_kind": "single_full_network_latency_only_harness",
        "status": status,
        "source": {
            "candidate_path": str(candidate_path.resolve()),
            "candidate_sha256": sha256_file(candidate_path),
            "arch_id": candidate.get("arch_id", ""),
            "metrics": candidate.get("metrics", {}),
        },
        "architecture": architecture,
        "required_gates": {
            "architecture_mapping": "complete" if mapping_ok else "incomplete",
            "missing_component_rows": len(missing_components),
        },
        "topology": [
            {
                "layer_index": row["layer_index"],
                "layer_role": row["layer_role"],
                "op": row["op"],
                "combo_case_name": row["current84_combo_case_name"],
                "measurement_namespace": row["board_measurement_namespace"],
                "decomposition": row["combo_decomposition"],
                "component_roles": row["component_roles"],
                "component_case_names": row["component_case_names"],
            }
            for row in trace_rows
        ],
        "streaming": {
            "layer_boundary": "stream_fifo_between_layers",
            "output_word_count": 8,
            "uart_protocol": "status_code, cycles, checksum, word_count",
        },
        "parameter_manifest": str(planning.ARCH84_EXTENSION_PARAMS.resolve()),
        "parameter_mode": "latency_only_deterministic",
        "accuracy_claim": "none",
        "ledger_boundary": [
            "Do not write this RL-best variant measurement into current84/fullcombo/arch84 canonical ledgers.",
            "Do not claim true RL-best e2e latency until this exact harness is measured on COM5.",
        ],
    }


def write_plan_outputs(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    candidate_path = Path(args.candidate_path).expanduser().resolve()
    candidate, architecture = candidate_payload(candidate_path)
    name = sanitize_name(args.run_name) if args.run_name else default_run_name(candidate, candidate_path)
    root = run_root(name)
    root.mkdir(parents=True, exist_ok=True)

    trace_rows = trace_candidate(architecture)
    gate = identity_gate(trace_rows, REFERENCE_TRACE)
    spec = build_harness_spec(
        run_name=name,
        candidate=candidate,
        candidate_path=candidate_path,
        architecture=architecture,
        trace_rows=trace_rows,
    )
    spec["parameter_mode"] = args.parameter_mode
    if args.parameter_mode == "trained_checkpoint_mem":
        spec["trained_checkpoint"] = {
            "path": str(Path(args.trained_checkpoint).expanduser().resolve()),
            "sha256": sha256_file(Path(args.trained_checkpoint).expanduser().resolve()),
        }
    preflight = {
        "generated_at": timestamp(),
        "run_name": name,
        "target_name": target_name(name),
        "mode": "planning_only",
        "parameter_mode": args.parameter_mode,
        "candidate_path": str(candidate_path),
        "candidate_arch_id": candidate.get("arch_id", ""),
        "identity_gate": gate,
        "harness_spec_status": spec["status"],
        "may_reuse_reference_arch84_bitstream": gate["matches_reference_arch84_e2e"],
        "reference_summary": str(REFERENCE_SUMMARY),
        "outputs": {
            "architecture_trace_csv": str(root / "rl_best_e2e_architecture_trace.csv"),
            "harness_spec_json": str(root / "rl_best_e2e_harness_spec.json"),
            "identity_gate_json": str(root / "rl_best_e2e_identity_gate.json"),
            "preflight_json": str(root / "rl_best_e2e_preflight.json"),
            "readme": str(root / "README.md"),
        },
    }

    write_csv(root / "rl_best_e2e_architecture_trace.csv", trace_rows, TRACE_FIELDS)
    write_json(root / "rl_best_e2e_harness_spec.json", spec)
    write_json(root / "rl_best_e2e_identity_gate.json", gate)
    write_json(root / "rl_best_e2e_preflight.json", preflight)
    (root / "README.md").write_text(
        "\n".join(
            [
                f"# RL Best E2E Board Validation: {name}",
                "",
                f"- Target: `{target_name(name)}`",
                f"- Candidate: `{candidate_path}`",
                f"- Harness spec status: `{spec['status']}`",
                f"- Reference arch_84 bitstream reusable: `{gate['matches_reference_arch84_e2e']}`",
                "",
                "True hardware latency may only be reported after this exact target writes a COM5 measurement JSON with UART status_code=0 and WNS>=0.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return name, preflight


def load_spec_for_run(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.spec_path:
        spec_path = Path(args.spec_path).expanduser().resolve()
        spec = read_json(spec_path)
        run_name = sanitize_name(args.run_name or str(spec.get("run_name") or spec_path.parent.name))
        return run_name, spec
    if args.run_name:
        spec_path = run_root(args.run_name) / "rl_best_e2e_harness_spec.json"
        name = sanitize_name(args.run_name)
        if spec_path.exists():
            return name, read_json(spec_path)
        if args.candidate_path:
            planned_name, _ = write_plan_outputs(args)
            return planned_name, read_json(run_root(planned_name) / "rl_best_e2e_harness_spec.json")
        return name, {}
    name, _ = write_plan_outputs(args)
    return name, read_json(run_root(name) / "rl_best_e2e_harness_spec.json")


def generate_harness(args: argparse.Namespace) -> dict[str, Any]:
    name, spec = load_spec_for_run(args)
    if spec.get("status") != "ready_to_generate":
        raise RuntimeError(f"RL best e2e spec is not ready_to_generate: {spec.get('status')}")
    root = run_root(name)
    project = project_root(name)
    if project.exists():
        if not args.force:
            raise FileExistsError(f"Harness project already exists; pass --force to regenerate: {project}")
        resolved = project.resolve()
        allowed = RESULT_ROOT.resolve()
        if allowed not in resolved.parents:
            raise RuntimeError(f"Refusing to remove path outside RL e2e result root: {resolved}")
        shutil.rmtree(project)
    (project / "export_rtl").mkdir(parents=True, exist_ok=True)

    board_config = Path(args.board_config).expanduser().resolve()
    component_rows = e2e.flattened_components(spec)
    overrides = component_overrides_from_manifest(args.component_overrides_from_manifest)
    overrides.update(variant.parse_component_overrides(args.component_override))
    seen_overrides: set[str] = set()
    override_records: list[dict[str, str]] = []
    component_specs: list[dict[str, Any]] = []
    for row in component_rows:
        effective_row = dict(row)
        if effective_row["role"] in overrides:
            seen_overrides.add(effective_row["role"])
            override_records.append(
                {
                    "role": effective_row["role"],
                    "original_case_name": effective_row["case_name"],
                    "override_case_name": overrides[effective_row["role"]],
                }
            )
            effective_row["case_name"] = overrides[effective_row["role"]]
        case_dir = resolve_rl_best_case_dir(effective_row["case_name"])
        item = fullcombo.build_component_spec(
            effective_row["role"],
            effective_row["case_name"],
            case_dir,
            project / "export_rtl",
        )
        item.update(effective_row)
        component_specs.append(item)
    missing_overrides = sorted(set(overrides) - seen_overrides)
    if missing_overrides:
        raise ValueError(f"Component overrides did not match RL-best roles: {missing_overrides}")

    e2e.write_full_network_top(
        project,
        component_specs,
        board_config,
        args.fifo_depth,
        args.fifo_style,
        args.watchdog_cycles,
        argmax_pipeline=args.argmax_pipeline,
        completion_mode=args.completion_mode,
        dynamic_validation=args.dynamic_validation,
        dynamic_uart_baud=args.dynamic_uart_baud,
    )
    fullcombo.write_constraints(project, board_config)
    artifact = target_name(name)
    fullcombo.write_build_tcl(project, artifact, board_config, recovery_variant=args.timing_variant)
    impl_artifacts = variant.apply_impl_profile(project, args.impl_profile)

    manifest = {
        "case_name": artifact,
        "run_name": name,
        "module_name": "sonar_classifier_e2e_harness",
        "board_config": str(board_config),
        "harness_kind": (
            "dynamic_validation_uart_v1"
            if args.dynamic_validation
            else "single_full_network_latency_only_harness"
        ),
        "implementation_variant": True,
        "parameter_mode": args.parameter_mode,
        "accuracy_claim": "none",
        "validation_eligibility": (
            "eligible_only_after_full_outer_validation_bit_exact_run"
            if args.dynamic_validation
            else "fixed_input_latency_only"
        ),
        "timing_variant": args.timing_variant,
        "impl_profile": args.impl_profile,
        "component_overrides": override_records,
        "component_overrides_from_manifest": (
            str(Path(args.component_overrides_from_manifest).expanduser().resolve())
            if args.component_overrides_from_manifest
            else ""
        ),
        "argmax_pipeline": bool(args.argmax_pipeline),
        "completion_mode": args.completion_mode,
        "dynamic_validation": bool(args.dynamic_validation),
        "dynamic_uart_baud": (
            int(args.dynamic_uart_baud) if args.dynamic_validation else None
        ),
        "harness_fifo_depth": args.fifo_depth,
        "harness_fifo_style": args.fifo_style,
        "impl_profile_artifacts": impl_artifacts,
        "source_spec": str(root / "rl_best_e2e_harness_spec.json"),
        "source_candidate": spec.get("source", {}),
        "generated": {
            "project_root": str(project),
            "rtl_dir": str(project / "rtl"),
            "export_rtl_dir": str(project / "export_rtl"),
            "constraints": str(project / "constraints" / "harness.xdc"),
            "build_tcl": str(project / "scripts" / "build_bitstream.tcl"),
            "bitstream": str(project / "bitstream" / f"{artifact}.bit"),
        },
        "streaming": {
            "input_axis_word_count": int(component_specs[0]["case_meta"]["input_word_count"]),
            "output_axis_word_count": int(component_specs[-1]["case_meta"]["output_word_count"]),
            "classifier_class_count": int(component_specs[-1]["constants"].get("OUT_CH", 8)),
            "uart_checksum_field": "packed {argmax[7:0], checksum24[23:0]} for e2e only",
        },
        "layers": spec.get("topology", []),
        "components": [
            {
                "role": item["role"],
                "case_name": item["case_name"],
                "layer_index": item["layer_index"],
                "layer_role": item["layer_role"],
                "renamed_module": item["renamed_module"],
                "input_word_count": item["case_meta"]["input_word_count"],
                "output_word_count": item["case_meta"]["output_word_count"],
                "input_tdata_width": item["top_ports"]["input_stream_TDATA"],
                "output_tdata_width": item["top_ports"]["output_stream_TDATA"],
            }
            for item in component_specs
        ],
        "notes": spec.get("ledger_boundary", []),
    }
    manifest_path = project / "harness_manifest.json"
    write_json(manifest_path, manifest)
    if args.parameter_mode == "trained_checkpoint_mem":
        trained_manifest = (
            Path(args.trained_weight_manifest).expanduser().resolve()
            if args.trained_weight_manifest
            else project / "trained_weight_manifest.json"
        )
        trained_export.export_to_project(
            checkpoint=Path(args.trained_checkpoint).expanduser().resolve(),
            candidate_path=Path(args.candidate_path).expanduser().resolve(),
            harness_manifest=manifest_path.resolve(),
            output_manifest=trained_manifest,
        )
        manifest = read_json(manifest_path)
    return manifest


def build_harness(args: argparse.Namespace) -> dict[str, Any]:
    name, _ = load_spec_for_run(args)
    project = project_root(name)
    if not (project / "harness_manifest.json").exists() or args.force:
        generate_harness(args)
    vivado = Path(args.vivado).expanduser().resolve()
    timeout_seconds = args.bitstream_timeout_minutes * 60 if args.bitstream_timeout_minutes > 0 else None
    result = build_bitstream(project, vivado, force=args.force, timeout_seconds=timeout_seconds)
    root = run_root(name)
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_text = str(result.get("stdout", ""))
    (logs_dir / "vivado_bitstream.out.log").write_text(
        stdout_text,
        encoding="utf-8",
        errors="replace",
    )
    (logs_dir / "vivado_bitstream.err.log").write_text(
        str(result.get("stderr", "")),
        encoding="utf-8",
        errors="replace",
    )
    evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
    if not evidence:
        evidence = variant.collect_vivado_evidence(project, root)
    util_report = evidence.get("placed_utilization_report") or evidence.get("synth_utilization_report")
    util = fullcombo.parse_utilization(Path(util_report)) if util_report else {}
    wns = (
        fullcombo.parse_wns(project / "reports" / "timing_summary.rpt")
        or variant.parse_wns_from_log(stdout_text)
        or ""
    )
    failure_summary = result.get("failure_summary", "") or variant.summarize_build_failure(stdout_text)
    payload = {
        "generated_at": timestamp(),
        "run_name": name,
        "target_name": target_name(name),
        "status": result.get("status", ""),
        "returncode": result.get("returncode", ""),
        "bitstream": result.get("bitstream") or "",
        "wns_ns": wns,
        "timing_clean": (variant.parse_float(wns) is not None and variant.parse_float(wns) >= 0.0),
        "lut": util.get("board_harness_lut", ""),
        "ff": util.get("board_harness_ff", ""),
        "bram": util.get("board_harness_bram_tile", ""),
        "dsp": util.get("board_harness_dsp", ""),
        "global_congestion_level": variant.parse_global_congestion(stdout_text),
        "build_result_json": str(root / "build_result.json"),
        "stdout_log": str(logs_dir / "vivado_bitstream.out.log"),
        "stderr_log": str(logs_dir / "vivado_bitstream.err.log"),
        "evidence": evidence,
        "failure_summary": failure_summary,
    }
    write_json(root / "build_result.json", payload)
    write_measurement_preflight(name, payload)
    return payload


def measurement_readiness(name: str, build_payload: dict[str, Any]) -> dict[str, Any]:
    bitstream_text = str(build_payload.get("bitstream", ""))
    bitstream = Path(bitstream_text) if bitstream_text else Path()
    bitstream_exists = bool(bitstream_text) and bitstream.exists()
    bitstream_hash = sha256_file(bitstream) if bitstream_exists else ""
    wns_value = variant.parse_float(build_payload.get("wns_ns", ""))
    timing_clean = wns_value is not None and wns_value >= 0.0
    measurement_path = measurement_json_path(name)
    measurement_payload = read_json(measurement_path)
    uart_status = (
        fullcombo.parse_int(fullcombo.frame_value(measurement_payload, "status_code"))
        if measurement_payload
        else None
    )
    strict_latency_usable = bool(measurement_payload) and uart_status == 0 and timing_clean
    blockers: list[str] = []
    if build_payload.get("status") != "success":
        blockers.append("implementation_not_success")
    if not bitstream_exists:
        blockers.append("bitstream_missing")
    if not timing_clean:
        blockers.append("timing_not_clean")
    return {
        "run_name": name,
        "target_name": target_name(name),
        "status": build_payload.get("status", ""),
        "bitstream_exists": bitstream_exists,
        "bitstream": str(bitstream) if bitstream_exists else "",
        "bitstream_sha256": bitstream_hash,
        "wns_ns": build_payload.get("wns_ns", ""),
        "timing_clean": timing_clean,
        "measurement_ready": not blockers and not measurement_payload,
        "measurement_json_path": str(measurement_path) if measurement_payload else "",
        "strict_latency_usable": strict_latency_usable,
        "readiness_blockers": "|".join(blockers),
        "suggested_measure_command": (
            f"python hls_lut_builder\\board_harness\\scripts\\rl_best_e2e_board_validation.py "
            f"measure --run-name {name} --serial-port COM5 --expected-bitstream-sha256 {bitstream_hash}"
            if not blockers and not measurement_payload
            else ""
        ),
    }


def write_measurement_preflight(name: str, build_payload: dict[str, Any]) -> dict[str, Any]:
    root = run_root(name)
    readiness = measurement_readiness(name, build_payload)
    preflight = {
        "generated_at": timestamp(),
        "run_name": name,
        "target_name": target_name(name),
        "measurement_scope": "isolated RL-best implementation variant",
        "readiness": readiness,
        "build_result": str(root / "build_result.json"),
        "guardrails": [
            "Bitstream must be under the RL-best harness project bitstream directory.",
            "Measurement JSON must be under this isolated RL-best result directory.",
            "If --expected-bitstream-sha256 is supplied, it must match the bitstream bytes.",
            "Do not update canonical e2e latency ledgers from this isolated run.",
        ],
    }
    write_json(root / "measurement_preflight.json", preflight)
    return preflight


def enrich_build_payload_from_logs(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    root = run_root(name)
    project = project_root(name)
    stdout_log = Path(str(payload.get("stdout_log", "")))
    stdout_text = stdout_log.read_text(encoding="utf-8", errors="replace") if stdout_log.exists() else ""
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    if not evidence:
        evidence = variant.collect_vivado_evidence(project, root)
    util_report = evidence.get("placed_utilization_report") or evidence.get("synth_utilization_report")
    util = fullcombo.parse_utilization(Path(util_report)) if util_report else {}
    wns = (
        str(payload.get("wns_ns", "") or "")
        or fullcombo.parse_wns(project / "reports" / "timing_summary.rpt")
        or variant.parse_wns_from_log(stdout_text)
        or ""
    )
    failure_summary = str(payload.get("failure_summary", "") or "") or variant.summarize_build_failure(stdout_text)
    payload.update(
        {
            "wns_ns": wns,
            "timing_clean": (variant.parse_float(wns) is not None and variant.parse_float(wns) >= 0.0),
            "lut": payload.get("lut") or util.get("board_harness_lut", ""),
            "ff": payload.get("ff") or util.get("board_harness_ff", ""),
            "bram": payload.get("bram") or util.get("board_harness_bram_tile", ""),
            "dsp": payload.get("dsp") or util.get("board_harness_dsp", ""),
            "global_congestion_level": payload.get("global_congestion_level")
            or variant.parse_global_congestion(stdout_text),
            "evidence": evidence,
            "failure_summary": failure_summary,
        }
    )
    write_json(root / "build_result.json", payload)
    return payload


def parse_measurement_stdout_runs(stdout_text: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    index = 0
    while index < len(stdout_text):
        brace = stdout_text.find("{", index)
        if brace < 0:
            break
        try:
            payload, end = decoder.raw_decode(stdout_text[brace:])
        except json.JSONDecodeError:
            index = brace + 1
            continue
        if isinstance(payload, dict) and isinstance(payload.get("frame"), dict):
            runs.append(payload)
        index = brace + end
    return runs


def write_stability_evidence(measurements_dir: Path, runs: list[dict[str, Any]]) -> None:
    status_codes = {str(fullcombo.frame_value(run, "status_code")) for run in runs}
    cycles = {str(fullcombo.frame_value(run, "cycles")) for run in runs}
    checksums = {str(fullcombo.frame_value(run, "checksum")) for run in runs}
    latency_ms = {str(run.get("latency_ms", "")) for run in runs}
    write_json(
        measurements_dir / "stability_runs.json",
        {
            "generated_at": timestamp(),
            "run_count": len(runs),
            "status_codes": sorted(status_codes),
            "cycles": sorted(cycles),
            "checksums": sorted(checksums),
            "latency_ms": sorted(latency_ms),
            "stable": (
                len(status_codes) == 1
                and len(cycles) == 1
                and len(checksums) == 1
                and len(latency_ms) == 1
            ),
            "runs": runs,
        },
    )
    with (measurements_dir / "measurements_raw_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_idx",
                "status_code",
                "cycles",
                "checksum",
                "word_count",
                "latency_ms",
                "clock_freq_hz",
                "serial_port",
                "bitstream",
            ],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "run_idx": run.get("run_idx"),
                    "status_code": fullcombo.frame_value(run, "status_code"),
                    "cycles": fullcombo.frame_value(run, "cycles"),
                    "checksum": fullcombo.frame_value(run, "checksum"),
                    "word_count": fullcombo.frame_value(run, "word_count"),
                    "latency_ms": run.get("latency_ms"),
                    "clock_freq_hz": run.get("clock_freq_hz"),
                    "serial_port": run.get("serial_port"),
                    "bitstream": run.get("bitstream"),
                }
            )


def measure_harness(args: argparse.Namespace) -> dict[str, Any]:
    name, _ = load_spec_for_run(args)
    root = run_root(name)
    project = project_root(name)
    build_payload = read_json(root / "build_result.json")
    if not build_payload:
        raise FileNotFoundError(f"Missing build result: {root / 'build_result.json'}")
    readiness = measurement_readiness(name, build_payload)
    if not readiness["bitstream_exists"]:
        raise FileNotFoundError(f"Bitstream is missing: {build_payload.get('bitstream', '')}")
    if not readiness["timing_clean"] and not args.allow_timing_fail_measurement:
        raise RuntimeError(
            f"Refusing timing-fail measurement without --allow-timing-fail-measurement: WNS={readiness['wns_ns']}"
        )
    bitstream = Path(str(build_payload["bitstream"]))
    if not is_relative_to(bitstream, project / "bitstream"):
        raise RuntimeError(f"Refusing bitstream outside RL-best project: {bitstream}")
    expected_hash = str(args.expected_bitstream_sha256 or "").strip().lower()
    actual_hash = sha256_file(bitstream).lower()
    if expected_hash and expected_hash != actual_hash:
        raise RuntimeError(f"Bitstream SHA256 mismatch expected={expected_hash} actual={actual_hash}")

    measurement_path = measurement_json_path(name)
    if measurement_path.exists() and not args.force:
        raise FileExistsError(f"Measurement JSON already exists; pass --force to overwrite: {measurement_path}")
    measurements_dir = root / "measurements"
    logs_dir = root / "logs"
    measurements_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    measurement = fullcombo.measure_harness(
        manifest=project / "harness_manifest.json",
        bitstream=bitstream,
        csv_path=measurements_dir / "measurements_raw.csv",
        json_out=measurement_path,
        serial_port=args.serial_port,
        vivado=Path(args.vivado).expanduser().resolve(),
        hw_server_url=args.hw_server_url,
        hw_device_pattern=args.hw_device_pattern,
        hw_target_index=args.hw_target_index,
        program_retries=args.program_retries,
        runs=args.runs,
        uart_timeout_seconds=args.uart_timeout_seconds,
        post_program_delay_seconds=args.post_program_delay_seconds,
        timeout_seconds=args.measurement_timeout_minutes * 60 if args.measurement_timeout_minutes > 0 else None,
        skip_measure=args.skip_measure,
    )
    stdout_text = str(measurement.get("stdout", ""))
    (logs_dir / "measurement.out.log").write_text(
        stdout_text,
        encoding="utf-8",
        errors="replace",
    )
    runs = parse_measurement_stdout_runs(stdout_text)
    if runs:
        write_stability_evidence(measurements_dir, runs)
    write_measurement_preflight(name, build_payload)
    return measurement


def refresh(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if RESULT_ROOT.exists():
        for path in sorted(RESULT_ROOT.glob("*/build_result.json")):
            name = path.parent.name
            row = enrich_build_payload_from_logs(name, read_json(path))
            row.update(measurement_readiness(name, row))
            rows.append(row)
            write_measurement_preflight(name, read_json(path))
    write_json(
        RESULT_ROOT / "rl_best_e2e_summary.json",
        {
            "generated_at": timestamp(),
            "run_count": len(rows),
            "measurement_ready_count": sum(1 for row in rows if row.get("measurement_ready") is True),
            "strict_latency_usable_count": sum(1 for row in rows if row.get("strict_latency_usable") is True),
            "runs": rows,
        },
    )
    return rows


def main() -> int:
    args = parse_args()
    configure_result_root(args)
    if args.mode == "plan":
        name, payload = write_plan_outputs(args)
        print(json.dumps({"run_name": name, **payload}, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "generate":
        print(json.dumps(generate_harness(args), ensure_ascii=False, indent=2))
        return 0
    if args.mode == "build":
        print(json.dumps(build_harness(args), ensure_ascii=False, indent=2))
        return 0
    if args.mode == "measure":
        print(json.dumps(measure_harness(args), ensure_ascii=False, indent=2))
        return 0
    if args.mode == "all":
        generate_harness(args)
        build_harness(args)
        print(json.dumps(measure_harness(args), ensure_ascii=False, indent=2))
        return 0
    if args.mode == "refresh":
        print(json.dumps(refresh(args), ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
