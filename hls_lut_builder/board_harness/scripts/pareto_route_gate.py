#!/usr/bin/env python3
"""Create cacheable route-gate JSON artifacts from Pareto top-K candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
BOARD_HARNESS_DIR = REPO_ROOT / "hls_lut_builder" / "board_harness"
DEFAULT_VALIDATION_SCRIPT = SCRIPT_DIR / "rl_best_e2e_board_validation.py"
DEFAULT_FINAL_REPORT_SCRIPT = SCRIPT_DIR / "final_hardware_validation_report.py"
DEFAULT_OUTPUT_ROOT = BOARD_HARNESS_DIR / "results" / "pareto_route_gate"
DEFAULT_LOWDSP_OVERRIDE_CATALOG = (
    BOARD_HARNESS_DIR / "configs" / "phase0_v3_lowdsp_override_catalog.yaml"
)
RL_E2E_RESULT_ROOT = BOARD_HARNESS_DIR / "results" / "sonar_classifier_rl_best_e2e_board"

sys.path.insert(0, str(REPO_ROOT / "src"))

from hwnas_fpga.hardware.route_gate import (  # noqa: E402
    classify_route_gate_result,
    dedupe_candidates_by_signature,
    encoding_signature,
    pareto_topk_candidates,
    select_route_clean_best,
)
from hwnas_fpga.hardware.validation_report import build_final_hardware_validation_report  # noqa: E402


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sanitize_name(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in {"_", "-", "."}:
            keep.append(char)
        else:
            keep.append("_")
    name = "".join(keep).strip("_.-")
    return name or "candidate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pareto top-K Vivado route-gate planner")
    parser.add_argument("mode", choices=("plan", "quick", "full", "refresh"))
    parser.add_argument("--pareto-selection", required=True, help="Path to results/pareto_selection.json")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--validation-script", default=str(DEFAULT_VALIDATION_SCRIPT))
    parser.add_argument("--final-report-script", default=str(DEFAULT_FINAL_REPORT_SCRIPT))
    parser.add_argument(
        "--implementation-variant-manifest",
        action="append",
        default=[],
        help=(
            "Existing harness_manifest.json for a full-route implementation variant. "
            "When source_candidate.arch_id matches a Pareto candidate, the planner "
            "uses that run_name/build/measurement path for the full-route gate. "
            "Kept as an alias for --full-implementation-variant-manifest."
        ),
    )
    parser.add_argument(
        "--quick-implementation-variant-manifest",
        action="append",
        default=[],
        help=(
            "Existing harness_manifest.json for quick/default implementation evidence. "
            "When source_candidate.arch_id matches a Pareto candidate, the planner "
            "uses that run_name/build path for the quick gate."
        ),
    )
    parser.add_argument(
        "--full-implementation-variant-manifest",
        action="append",
        default=[],
        help=(
            "Existing harness_manifest.json for full-route implementation evidence. "
            "When source_candidate.arch_id matches a Pareto candidate, the planner "
            "uses that run_name/build/measurement path for the full-route gate."
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--topk", type=int, default=24)
    parser.add_argument("--full-topk", type=int, default=8)
    parser.add_argument(
        "--full-max-actual-dsp",
        type=int,
        default=700,
        help="Maximum actual Vivado DSP for full-route PASS candidates to be declared clean.",
    )
    parser.add_argument("--serial-port", default="COM5")
    parser.add_argument("--quick-timing-variant", default="place_altspread_route_more_global")
    parser.add_argument("--full-timing-variant", default="place_altspread_route_more_global")
    parser.add_argument("--quick-impl-profile", default="default")
    parser.add_argument("--full-impl-profile", default="fanout_only")
    parser.add_argument("--quick-timeout-minutes", type=int, default=180)
    parser.add_argument("--full-timeout-minutes", type=int, default=240)
    parser.add_argument(
        "--lowdsp-override-catalog",
        action="append",
        default=None,
        help=(
            "YAML catalog mapping v3 stage/block choices to low-DSP component override cases. "
            "Full-route candidates without complete catalog coverage are skipped."
        ),
    )
    parser.add_argument("--execute", action="store_true", help="Actually invoke the validation build script")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def candidate_filename(selection_index: int, arch_id: str) -> str:
    return f"{selection_index:03d}_{sanitize_name(arch_id)}.candidate.json"


def run_name(prefix: str, selection_index: int, arch_id: str, signature: str) -> str:
    return sanitize_name(f"{prefix}_rank{selection_index:03d}_{arch_id}_{signature[:10]}")


def target_name(run_name_value: str) -> str:
    return f"sonar_classifier_rl_best_e2e__{sanitize_name(run_name_value)}"


def measurement_json_for_run(run_name_value: str) -> Path:
    return (
        RL_E2E_RESULT_ROOT
        / sanitize_name(run_name_value)
        / "measurements"
        / f"{target_name(run_name_value)}.measurement.json"
    )


def manifest_encoding_signature(manifest: dict[str, Any]) -> str:
    source_candidate = manifest.get("source_candidate")
    if not isinstance(source_candidate, dict):
        return ""

    for key in ("encoding_signature", "architecture_signature"):
        value = source_candidate.get(key)
        if value:
            return str(value)

    candidate_path_raw = source_candidate.get("candidate_path")
    if not candidate_path_raw:
        return ""
    candidate_payload = read_json(Path(str(candidate_path_raw)).expanduser())
    candidate = (
        candidate_payload.get("candidate")
        if isinstance(candidate_payload.get("candidate"), dict)
        else candidate_payload
    )
    if not isinstance(candidate, dict):
        return ""
    encoding = candidate.get("encoding") or candidate.get("architecture")
    if isinstance(encoding, dict):
        return str(candidate.get("encoding_signature") or encoding_signature(encoding))
    pareto = candidate_payload.get("pareto")
    if isinstance(pareto, dict) and pareto.get("encoding_signature"):
        return str(pareto["encoding_signature"])
    return ""


def variant_lookup_keys(*, arch_id: str, signature: str = "") -> list[str]:
    keys = []
    if arch_id:
        keys.append(arch_id)
    if signature:
        keys.append(f"encoding_signature:{signature}")
    return keys


def resolve_implementation_variant(
    variants: dict[str, dict[str, Any]],
    *,
    arch_id: str,
    signature: str,
) -> dict[str, Any] | None:
    for key in variant_lookup_keys(arch_id=arch_id, signature=signature):
        variant = variants.get(key)
        if variant is not None:
            return variant
    return None


def load_implementation_variants(paths: list[str]) -> dict[str, dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        manifest = read_json(path)
        source_candidate = manifest.get("source_candidate")
        if not isinstance(source_candidate, dict):
            continue
        arch_id = str(source_candidate.get("arch_id") or "")
        run_name_value = str(manifest.get("run_name") or "")
        if not arch_id or not run_name_value:
            continue
        signature = manifest_encoding_signature(manifest)
        overrides = []
        for item in manifest.get("component_overrides") or []:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            case_name = item.get("override_case_name")
            if role and case_name:
                overrides.append(f"{role}={case_name}")
        variant = {
            "source_manifest": str(path),
            "run_name": run_name_value,
            "target_name": manifest.get("case_name") or target_name(run_name_value),
            "timing_variant": manifest.get("timing_variant") or "",
            "impl_profile": manifest.get("impl_profile") or "",
            "arch_id": arch_id,
            "encoding_signature": signature,
            "component_overrides": overrides,
            "argmax_pipeline": bool(manifest.get("argmax_pipeline")),
            "completion_mode": manifest.get("completion_mode") or "",
            "fifo_depth": manifest.get("harness_fifo_depth"),
            "fifo_style": manifest.get("harness_fifo_style") or "",
            "accuracy_claim": manifest.get("accuracy_claim"),
            "parameter_mode": manifest.get("parameter_mode"),
        }
        for key in variant_lookup_keys(arch_id=arch_id, signature=signature):
            variants[key] = variant
    return variants


def load_lowdsp_override_catalog(paths: list[str] | None) -> dict[str, Any]:
    catalog: dict[str, Any] = {
        "defaults": {},
        "direct": {},
        "mbconv": {},
        "source_catalogs": [],
    }
    for raw_path in paths or []:
        path = Path(raw_path).expanduser().resolve()
        payload = read_yaml(path)
        if not payload:
            continue
        catalog["source_catalogs"].append(str(path))
        for section in ("defaults", "direct", "mbconv"):
            section_payload = payload.get(section)
            if isinstance(section_payload, Mapping):
                catalog[section].update(dict(section_payload))
    return catalog


def _first_block(stage: Mapping[str, Any]) -> Mapping[str, Any]:
    blocks = stage.get("blocks")
    if isinstance(blocks, list) and blocks and isinstance(blocks[0], Mapping):
        return blocks[0]
    return {}


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mapping_to_component_overrides(mapping: Mapping[str, Any]) -> list[str]:
    overrides = []
    for role, case_name in mapping.items():
        if role and case_name:
            overrides.append(f"{role}={case_name}")
    return overrides


def _append_direct_override(
    *,
    catalog: Mapping[str, Any],
    case_key: str,
    missing_label: str,
    overrides: list[str],
    missing: list[str],
) -> None:
    direct = catalog.get("direct")
    direct_cases = direct if isinstance(direct, Mapping) else {}
    stage_mapping = direct_cases.get(case_key)
    if isinstance(stage_mapping, Mapping):
        overrides.extend(_mapping_to_component_overrides(stage_mapping))
    else:
        missing.append(missing_label)


def resolve_lowdsp_overrides(
    encoding: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve v3 low-DSP component overrides for one architecture encoding."""
    if not catalog.get("source_catalogs"):
        return {
            "status": "not_requested",
            "component_overrides": [],
            "missing": [],
            "source_catalogs": [],
        }

    stages = encoding.get("stages")
    if not isinstance(stages, list) or len(stages) < 4:
        return {
            "status": "lowdsp_coverage_missing",
            "component_overrides": [],
            "missing": ["encoding.stages must contain four stages"],
            "source_catalogs": list(catalog.get("source_catalogs") or []),
        }

    overrides: list[str] = []
    missing: list[str] = []

    stage0 = stages[0] if isinstance(stages[0], Mapping) else {}
    stage0_block = _first_block(stage0)
    if (
        stage0_block.get("op") == "conv"
        and _int_value(stage0_block.get("kernel_size")) == 1
        and _int_value(stage0_block.get("expand_ratio"), 1) == 1
        and _int_value(stage0.get("channels")) == 16
        and _int_value(encoding.get("stem_channels")) == 32
    ):
        _append_direct_override(
            catalog=catalog,
            case_key="stage0_conv_32_16",
            missing_label="stage0_conv_32_16",
            overrides=overrides,
            missing=missing,
        )
    else:
        missing.append("stage0 conv k1 e1 32->16")

    mbconv = catalog.get("mbconv")
    mbconv_cases = mbconv if isinstance(mbconv, Mapping) else {}
    for stage_index in (1, 2, 3):
        stage = stages[stage_index] if isinstance(stages[stage_index], Mapping) else {}
        block = _first_block(stage)
        op = block.get("op")
        if op == "skip":
            continue
        expand_ratio = _int_value(block.get("expand_ratio"), 1)
        kernel_size = _int_value(block.get("kernel_size"))
        if op in {"denoise", "edge"} and stage_index == 3 and kernel_size == 3 and expand_ratio == 1:
            case_key = f"stage{stage_index}_{op}_k{kernel_size}_e{expand_ratio}"
            _append_direct_override(
                catalog=catalog,
                case_key=case_key,
                missing_label=case_key,
                overrides=overrides,
                missing=missing,
            )
            continue
        if op != "mbconv" or kernel_size != 3:
            missing.append(f"stage{stage_index} {op or 'unknown'} k{block.get('kernel_size')} e{expand_ratio}")
            continue
        case_key = f"stage{stage_index}_e{expand_ratio}"
        stage_mapping = mbconv_cases.get(case_key)
        if isinstance(stage_mapping, Mapping):
            overrides.extend(_mapping_to_component_overrides(stage_mapping))
        else:
            missing.append(case_key)

    return {
        "status": "covered" if not missing else "lowdsp_coverage_missing",
        "component_overrides": overrides if not missing else [],
        "missing": missing,
        "source_catalogs": list(catalog.get("source_catalogs") or []),
    }


def _catalog_default(
    catalog: Mapping[str, Any],
    key: str,
    fallback: Any = None,
) -> Any:
    defaults = catalog.get("defaults")
    if isinstance(defaults, Mapping) and defaults.get(key) not in (None, ""):
        return defaults.get(key)
    return fallback


def build_command(
    *,
    args: argparse.Namespace,
    candidate_path: Path,
    run_name_value: str,
    timing_variant: str,
    impl_profile: str,
    timeout_minutes: int,
    component_overrides: list[str] | None = None,
    argmax_pipeline: bool = False,
    completion_mode: str = "",
    fifo_depth: Any = None,
    fifo_style: str = "",
) -> list[str]:
    command = [
        args.python,
        str(Path(args.validation_script).expanduser().resolve()),
        "build",
        "--candidate-path",
        str(candidate_path.resolve()),
        "--run-name",
        run_name_value,
        "--timing-variant",
        timing_variant,
        "--impl-profile",
        impl_profile,
        "--bitstream-timeout-minutes",
        str(timeout_minutes),
    ]
    for override in component_overrides or []:
        command.extend(["--component-override", override])
    if argmax_pipeline:
        command.append("--argmax-pipeline")
    if completion_mode:
        command.extend(["--completion-mode", completion_mode])
    if fifo_depth not in (None, ""):
        command.extend(["--fifo-depth", str(fifo_depth)])
    if fifo_style:
        command.extend(["--fifo-style", fifo_style])
    if args.force:
        command.append("--force")
    return command


def build_measure_command(
    *,
    args: argparse.Namespace,
    run_name_value: str,
    expected_bitstream_sha256: str = "",
) -> list[str]:
    command = [
        args.python,
        str(Path(args.validation_script).expanduser().resolve()),
        "measure",
        "--run-name",
        run_name_value,
        "--serial-port",
        args.serial_port,
    ]
    if expected_bitstream_sha256:
        command.extend(["--expected-bitstream-sha256", expected_bitstream_sha256])
    return command


def build_report_command(
    *,
    args: argparse.Namespace,
    candidate_path: str,
    route_build_result: str,
    measurement_json: str,
    output: str,
) -> list[str]:
    command = [
        args.python,
        str(Path(args.final_report_script).expanduser().resolve()),
        "--candidate-path",
        candidate_path,
        "--route-build-result",
        route_build_result,
        "--output",
        output,
        "--expected-serial-port",
        args.serial_port,
    ]
    if measurement_json:
        command.extend(["--measurement-json", measurement_json])
    return command


def prepare_plan(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).expanduser().resolve()
    pareto_path = Path(args.pareto_selection).expanduser().resolve()
    pareto_payload = read_json(pareto_path)
    raw_lowdsp_catalog_paths = getattr(args, "lowdsp_override_catalog", None)
    lowdsp_catalog_paths = list(
        raw_lowdsp_catalog_paths or [str(DEFAULT_LOWDSP_OVERRIDE_CATALOG)]
    )
    lowdsp_catalog = load_lowdsp_override_catalog(lowdsp_catalog_paths)
    candidates = dedupe_candidates_by_signature(
        pareto_topk_candidates(pareto_payload, topk=args.topk)
    )
    quick_implementation_variants = load_implementation_variants(
        args.quick_implementation_variant_manifest
    )
    full_implementation_variants = load_implementation_variants(
        [*args.implementation_variant_manifest, *args.full_implementation_variant_manifest]
    )

    candidate_dir = output_root / "candidates"
    plan_candidates: list[dict[str, Any]] = []
    for fallback_index, candidate in enumerate(candidates, start=1):
        selection_index = int(candidate.get("selection_index") or fallback_index)
        arch_id = str(candidate.get("arch_id") or f"candidate_{selection_index:03d}")
        signature = str(candidate.get("encoding_signature") or "")
        candidate_path = candidate_dir / candidate_filename(selection_index, arch_id)
        write_json(
            candidate_path,
            {
                "candidate": {
                    "arch_id": arch_id,
                    "encoding": candidate["encoding"],
                    "metrics": candidate.get("metrics", {}),
                },
                "pareto": {
                    "selection_index": candidate.get("selection_index"),
                    "rank": candidate.get("rank"),
                    "objective_values": candidate.get("objective_values", {}),
                    "encoding_signature": signature,
                },
                "source": {
                    "pareto_selection_json": str(pareto_path),
                },
            },
        )
        quick_run = run_name("quick", selection_index, arch_id, signature)
        full_run = run_name("full", selection_index, arch_id, signature)
        quick_variant = resolve_implementation_variant(
            quick_implementation_variants,
            arch_id=arch_id,
            signature=signature,
        )
        if quick_variant:
            quick_run = str(quick_variant["run_name"])
            quick_timing_variant = str(quick_variant.get("timing_variant") or args.quick_timing_variant)
            quick_impl_profile = str(quick_variant.get("impl_profile") or args.quick_impl_profile)
            quick_component_overrides = list(quick_variant.get("component_overrides") or [])
            quick_argmax_pipeline = bool(quick_variant.get("argmax_pipeline"))
            quick_completion_mode = str(quick_variant.get("completion_mode") or "")
            quick_fifo_depth = quick_variant.get("fifo_depth")
            quick_fifo_style = str(quick_variant.get("fifo_style") or "")
        else:
            quick_timing_variant = args.quick_timing_variant
            quick_impl_profile = args.quick_impl_profile
            quick_component_overrides = []
            quick_argmax_pipeline = False
            quick_completion_mode = ""
            quick_fifo_depth = None
            quick_fifo_style = ""
        lowdsp_coverage = resolve_lowdsp_overrides(candidate["encoding"], lowdsp_catalog)
        full_route_skipped = False
        full_route_skip_reason = ""
        full_variant = resolve_implementation_variant(
            full_implementation_variants,
            arch_id=arch_id,
            signature=signature,
        )
        if full_variant:
            full_run = str(full_variant["run_name"])
            full_timing_variant = str(full_variant.get("timing_variant") or args.full_timing_variant)
            full_impl_profile = str(full_variant.get("impl_profile") or args.full_impl_profile)
            full_component_overrides = list(full_variant.get("component_overrides") or [])
            full_argmax_pipeline = bool(full_variant.get("argmax_pipeline"))
            full_completion_mode = str(full_variant.get("completion_mode") or "")
            full_fifo_depth = full_variant.get("fifo_depth")
            full_fifo_style = str(full_variant.get("fifo_style") or "")
        else:
            if lowdsp_coverage["status"] == "covered":
                full_timing_variant = str(
                    _catalog_default(lowdsp_catalog, "timing_variant", args.full_timing_variant)
                )
                full_impl_profile = str(
                    _catalog_default(lowdsp_catalog, "impl_profile", args.full_impl_profile)
                )
                full_component_overrides = list(lowdsp_coverage.get("component_overrides") or [])
                full_argmax_pipeline = bool(_catalog_default(lowdsp_catalog, "argmax_pipeline", False))
                full_completion_mode = str(_catalog_default(lowdsp_catalog, "completion_mode", ""))
                full_fifo_depth = _catalog_default(lowdsp_catalog, "fifo_depth", None)
                full_fifo_style = str(_catalog_default(lowdsp_catalog, "fifo_style", ""))
            else:
                full_timing_variant = args.full_timing_variant
                full_impl_profile = args.full_impl_profile
                full_component_overrides = []
                full_argmax_pipeline = False
                full_completion_mode = ""
                full_fifo_depth = None
                full_fifo_style = ""
                if lowdsp_coverage["status"] == "lowdsp_coverage_missing":
                    full_route_skipped = True
                    full_route_skip_reason = "lowdsp_coverage_missing"
        full_command = []
        if not full_route_skipped:
            full_command = build_command(
                args=args,
                candidate_path=candidate_path,
                run_name_value=full_run,
                timing_variant=full_timing_variant,
                impl_profile=full_impl_profile,
                timeout_minutes=args.full_timeout_minutes,
                component_overrides=full_component_overrides,
                argmax_pipeline=full_argmax_pipeline,
                completion_mode=full_completion_mode,
                fifo_depth=full_fifo_depth,
                fifo_style=full_fifo_style,
            )
        plan_candidates.append(
            {
                "selection_index": selection_index,
                "rank": candidate.get("rank"),
                "arch_id": arch_id,
                "cache_key": f"{arch_id}:{signature}",
                "encoding_signature": signature,
                "metrics": candidate.get("metrics", {}),
                "candidate_path": str(candidate_path),
                "quick_run_name": quick_run,
                "full_run_name": full_run,
                "quick_implementation_variant": quick_variant,
                "full_implementation_variant": full_variant,
                "implementation_variant": full_variant,
                "lowdsp_coverage": lowdsp_coverage,
                "full_route_skipped": full_route_skipped,
                "full_route_skip_reason": full_route_skip_reason,
                "quick_build_result": str(RL_E2E_RESULT_ROOT / quick_run / "build_result.json"),
                "full_build_result": str(RL_E2E_RESULT_ROOT / full_run / "build_result.json"),
                "quick_gate_json": str(output_root / "gates" / f"{selection_index:03d}_{sanitize_name(arch_id)}.quick_gate.json"),
                "full_gate_json": str(output_root / "gates" / f"{selection_index:03d}_{sanitize_name(arch_id)}.full_gate.json"),
                "measurement_json": str(measurement_json_for_run(full_run)),
                "final_validation_report_json": str(output_root / "reports" / f"{selection_index:03d}_{sanitize_name(arch_id)}.final_hardware_validation_report.json"),
                "quick_command": build_command(
                    args=args,
                    candidate_path=candidate_path,
                    run_name_value=quick_run,
                    timing_variant=quick_timing_variant,
                    impl_profile=quick_impl_profile,
                    timeout_minutes=args.quick_timeout_minutes,
                    component_overrides=quick_component_overrides,
                    argmax_pipeline=quick_argmax_pipeline,
                    completion_mode=quick_completion_mode,
                    fifo_depth=quick_fifo_depth,
                    fifo_style=quick_fifo_style,
                ),
                "full_command": full_command,
                "measurement_command": build_measure_command(
                    args=args,
                    run_name_value=full_run,
                ),
            }
        )

    plan = {
        "generated_at": timestamp(),
        "mode": "plan",
        "pareto_selection_json": str(pareto_path),
        "output_root": str(output_root),
        "validation_script": str(Path(args.validation_script).expanduser().resolve()),
        "final_report_script": str(Path(args.final_report_script).expanduser().resolve()),
        "candidate_count": len(plan_candidates),
        "topk": args.topk,
        "full_topk": args.full_topk,
        "full_max_actual_dsp": getattr(args, "full_max_actual_dsp", 700),
        "dedupe_rule": "encoding_signature",
        "cache_policy": "reuse existing build_result.json unless --force is passed",
        "lowdsp_override_catalogs": list(lowdsp_catalog.get("source_catalogs") or []),
        "quick_implementation_variant_manifests": args.quick_implementation_variant_manifest,
        "full_implementation_variant_manifests": [
            *args.implementation_variant_manifest,
            *args.full_implementation_variant_manifest,
        ],
        "implementation_variant_manifests": args.implementation_variant_manifest,
        "guardrails": [
            "Quick/full route gate JSON is implementation evidence only; COM5 latency still requires measurement JSON.",
            "Final best may only be selected from candidates with full_route_gate.gate_status == PASS and actual Vivado DSP within full_max_actual_dsp.",
            "Full route is skipped when the v3 low-DSP override catalog does not fully cover a candidate structure.",
            "This planner does not delete existing result directories.",
        ],
        "candidates": plan_candidates,
    }
    write_json(output_root / "route_gate_plan.json", plan)
    return plan


def execute_jobs(jobs: list[dict[str, Any]], cwd: Path, *, force: bool = False) -> list[dict[str, Any]]:
    results = []
    for job in jobs:
        command = list(job["command"])
        build_result = Path(str(job["build_result"]))
        if build_result.exists() and not force:
            results.append(
                {
                    "command": command,
                    "build_result": str(build_result),
                    "status": "skipped_existing_cache",
                    "started_at": None,
                    "finished_at": timestamp(),
                    "returncode": 0,
                    "stdout_tail": "",
                    "stderr_tail": "",
                }
            )
            continue
        started_at = timestamp()
        result = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        results.append(
            {
                "command": command,
                "build_result": str(build_result),
                "status": "executed",
                "started_at": started_at,
                "finished_at": timestamp(),
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-4000:],
                "stderr_tail": result.stderr[-4000:],
            }
        )
    return results


def write_candidate_gate_json(
    *,
    path: Path,
    phase: str,
    item: dict[str, Any],
    build_result_path: str,
    build_payload: dict[str, Any],
    gate: dict[str, Any],
) -> None:
    write_json(
        path,
        {
            "generated_at": timestamp(),
            "phase": phase,
            "candidate": {
                "selection_index": item.get("selection_index"),
                "rank": item.get("rank"),
                "arch_id": item.get("arch_id"),
                "encoding_signature": item.get("encoding_signature"),
                "metrics": item.get("metrics", {}),
                "candidate_path": item.get("candidate_path"),
            },
            "cache": {
                "cache_key": item.get("cache_key"),
                "build_result_json": build_result_path,
                "build_result_exists": bool(build_payload),
            },
            "gate": gate,
            "command": item.get(f"{phase}_command"),
        },
    )


def refresh_summary(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).expanduser().resolve()
    plan = read_json(output_root / "route_gate_plan.json") or prepare_plan(args)
    full_max_actual_dsp = getattr(args, "full_max_actual_dsp", plan.get("full_max_actual_dsp", 700))
    rows = []
    for item in plan.get("candidates", []):
        if not isinstance(item, dict):
            continue
        full_route_skipped = bool(item.get("full_route_skipped"))
        quick_payload = read_json(Path(str(item.get("quick_build_result", ""))))
        full_payload = (
            {}
            if full_route_skipped
            else read_json(Path(str(item.get("full_build_result", ""))))
        )
        measurement_json = str(item.get("measurement_json", ""))
        measurement_payload = read_json(Path(measurement_json)) if measurement_json else {}
        quick_gate = classify_route_gate_result(quick_payload)
        full_gate = classify_route_gate_result(
            full_payload,
            max_actual_dsp=full_max_actual_dsp,
        )
        full_gate["lowdsp_coverage"] = item.get("lowdsp_coverage")
        if full_route_skipped:
            full_gate.update(
                {
                    "gate_status": "FAIL",
                    "route_success": False,
                    "timing_clean": False,
                    "failure_reason": item.get("full_route_skip_reason") or "full_route_skipped",
                    "actual_dsp_acceptable": False,
                }
            )
        bitstream = Path(str(full_payload.get("bitstream", "")))
        bitstream_sha256 = str(full_payload.get("bitstream_sha256") or sha256_file(bitstream))
        route_payload_for_report = dict(full_payload)
        if bitstream_sha256:
            route_payload_for_report["bitstream_sha256"] = bitstream_sha256
        final_validation = build_final_hardware_validation_report(
            candidate_payload={
                "candidate": {
                    "arch_id": item.get("arch_id"),
                    "encoding_signature": item.get("encoding_signature"),
                    "metrics": item.get("metrics", {}),
                }
            },
            route_payload=route_payload_for_report,
            measurement_payload=measurement_payload,
            measurement_json_path=measurement_json if measurement_payload else "",
            expected_serial_port=args.serial_port,
        )
        final_validation_report_json = Path(str(item.get("final_validation_report_json", "")))
        if str(final_validation_report_json):
            write_json(final_validation_report_json, final_validation)
        report_command = build_report_command(
            args=args,
            candidate_path=str(item.get("candidate_path", "")),
            route_build_result=str(item.get("full_build_result", "")),
            measurement_json=measurement_json if measurement_payload else "",
            output=str(item.get("final_validation_report_json", "")),
        )
        measure_command = build_measure_command(
            args=args,
            run_name_value=str(item.get("full_run_name", "")),
            expected_bitstream_sha256=bitstream_sha256,
        )
        quick_gate_json = Path(str(item.get("quick_gate_json", "")))
        full_gate_json = Path(str(item.get("full_gate_json", "")))
        if str(quick_gate_json):
            write_candidate_gate_json(
                path=quick_gate_json,
                phase="quick",
                item=item,
                build_result_path=str(item.get("quick_build_result", "")),
                build_payload=quick_payload,
                gate=quick_gate,
            )
        if str(full_gate_json):
            write_candidate_gate_json(
                path=full_gate_json,
                phase="full",
                item=item,
                build_result_path=str(item.get("full_build_result", "")),
                build_payload=full_payload,
                gate=full_gate,
            )
        rows.append(
            {
                "candidate": {
                    "selection_index": item.get("selection_index"),
                    "rank": item.get("rank"),
                    "arch_id": item.get("arch_id"),
                    "encoding_signature": item.get("encoding_signature"),
                    "metrics": item.get("metrics", {}),
                    "candidate_path": item.get("candidate_path"),
                },
                "quick_route_gate": quick_gate,
                "full_route_gate": full_gate,
                "quick_build_result": item.get("quick_build_result"),
                "full_build_result": item.get("full_build_result"),
                "quick_gate_json": item.get("quick_gate_json"),
                "full_gate_json": item.get("full_gate_json"),
                "measurement_json": measurement_json,
                "final_validation_report_json": item.get("final_validation_report_json"),
                "final_validation": final_validation,
                "lowdsp_coverage": item.get("lowdsp_coverage"),
                "full_route_skipped": full_route_skipped,
                "full_route_skip_reason": item.get("full_route_skip_reason"),
                "quick_command": item.get("quick_command"),
                "full_command": item.get("full_command"),
                "measurement_command": measure_command,
                "final_report_command": report_command,
            }
        )

    clean_best = select_route_clean_best(rows, max_actual_dsp=full_max_actual_dsp)
    claimable_rows = [
        row
        for row in rows
        if row["final_validation"].get("claimable_real_e2e_latency") is True
    ]
    claimable_best = select_route_clean_best(claimable_rows, max_actual_dsp=full_max_actual_dsp)
    final_candidate_manifest = {
        "generated_at": timestamp(),
        "selection_stage": (
            "claimable_board_best"
            if claimable_best is not None
            else "route_clean_best"
            if clean_best is not None
            else "none"
        ),
        "candidate": (
            claimable_best["candidate"]
            if claimable_best is not None
            else clean_best["candidate"]
            if clean_best is not None
            else None
        ),
        "route_gate": (
            claimable_best["full_route_gate"]
            if claimable_best is not None
            else clean_best["full_route_gate"]
            if clean_best is not None
            else None
        ),
        "final_validation": (
            claimable_best["final_validation"]
            if claimable_best is not None
            else clean_best["final_validation"]
            if clean_best is not None
            else None
        ),
        "artifacts": (
            {
                "full_build_result": (
                    claimable_best if claimable_best is not None else clean_best
                ).get("full_build_result"),
                "measurement_json": (
                    claimable_best if claimable_best is not None else clean_best
                ).get("measurement_json"),
                "final_validation_report_json": (
                    claimable_best if claimable_best is not None else clean_best
                ).get("final_validation_report_json"),
            }
            if (claimable_best is not None or clean_best is not None)
            else {}
        ),
        "selection_rules": {
            "route_clean_best": "candidate must have full_route_gate.gate_status == PASS",
            "claimable_board_best": "route-clean candidate must also have matching COM5 status_code=0 measurement",
        },
    }
    summary = {
        "generated_at": timestamp(),
        "plan_json": str(output_root / "route_gate_plan.json"),
        "candidate_count": len(rows),
        "quick_pass_count": sum(1 for row in rows if row["quick_route_gate"]["gate_status"] == "PASS"),
        "quick_borderline_count": sum(1 for row in rows if row["quick_route_gate"]["gate_status"] == "BORDERLINE"),
        "full_route_clean_count": sum(1 for row in rows if row["full_route_gate"]["gate_status"] == "PASS"),
        "full_max_actual_dsp": full_max_actual_dsp,
        "full_route_skipped_count": sum(1 for row in rows if row.get("full_route_skipped")),
        "lowdsp_coverage_missing_count": sum(
            1
            for row in rows
            if (row.get("lowdsp_coverage") or {}).get("status") == "lowdsp_coverage_missing"
        ),
        "claimable_real_e2e_latency_count": len(claimable_rows),
        "final_best_selection_rule": "selected only from full_route_gate PASS rows with actual Vivado DSP within full_max_actual_dsp",
        "claimable_board_latency_rule": "requires full-route PASS, matching bitstream, COM5 status_code=0, measurement latency/cycles, and actual Vivado DSP within full_max_actual_dsp",
        "route_clean_best": None if clean_best is None else clean_best["candidate"],
        "claimable_board_best": None if claimable_best is None else claimable_best["candidate"],
        "candidates": rows,
    }
    write_json(output_root / "route_gate_summary.json", summary)
    write_json(output_root / "final_route_clean_best.json", summary["route_clean_best"])
    write_json(output_root / "final_claimable_board_best.json", summary["claimable_board_best"])
    write_json(output_root / "final_candidate_manifest.json", final_candidate_manifest)
    return summary


def main() -> int:
    args = parse_args()
    plan = prepare_plan(args)
    output_root = Path(args.output_root).expanduser().resolve()
    if args.mode == "plan":
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    if args.mode in {"quick", "full"}:
        if args.mode == "quick":
            selected_items = list(plan["candidates"])
        else:
            selected_items = list(plan["candidates"][: max(0, args.full_topk)])
        jobs = [
            {
                "command": item[f"{args.mode}_command"],
                "build_result": item[f"{args.mode}_build_result"],
                "arch_id": item["arch_id"],
                "selection_index": item["selection_index"],
            }
            for item in selected_items
            if item.get(f"{args.mode}_command")
            and not item.get(f"{args.mode}_route_skipped")
        ]
        execution = {
            "generated_at": timestamp(),
            "mode": args.mode,
            "execute": bool(args.execute),
            "cache_policy": "skip existing build_result.json unless --force is passed",
            "jobs": jobs,
            "results": execute_jobs(jobs, REPO_ROOT, force=args.force) if args.execute else [],
        }
        write_json(output_root / f"{args.mode}_gate_execution.json", execution)
        print(json.dumps(refresh_summary(args), ensure_ascii=False, indent=2))
        return 0

    if args.mode == "refresh":
        print(json.dumps(refresh_summary(args), ensure_ascii=False, indent=2))
        return 0

    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
