#!/usr/bin/env python3
"""Phase0 v4 sonar-operator precheck.

This audit is local-only. It does not train, run Vivado, or access COM ports.
It checks whether denoise/edge/mixconv can be admitted into a Phase0 v4
low-DSP search space under the current strict board LUT and route override
rules.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "search"
    / "nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_v3_lowdsp_av7k325.yaml"
)
DEFAULT_LUT = (
    REPO_ROOT
    / "hls_lut_builder"
    / "results"
    / "nas_board_lut_strict_current84_arch84"
    / "nas_board_lut_strict.json"
)
DEFAULT_STATUS = DEFAULT_LUT.with_name("nas_board_lut_status.json")
DEFAULT_OPERATOR_MANIFEST = REPO_ROOT / "hls_lut_builder" / "configs" / "operator_manifest.yaml"
DEFAULT_CANDIDATE_KERNELS = REPO_ROOT / "hls_lut_builder" / "configs" / "candidate_kernels.yaml"
DEFAULT_LOWDSP_CATALOG = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "configs"
    / "phase0_v3_lowdsp_override_catalog.yaml"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "phase0_v4_sonar_op_precheck"

sys.path.insert(0, str(REPO_ROOT / "src"))

from hwnas_fpga.hardware.cost import FPGACostEstimator  # noqa: E402
from hwnas_fpga.hardware.lookup_table import FormalLutStatusEntry, LutQueryEngine, LutTable  # noqa: E402
from hwnas_fpga.interfaces import HardwareSpec  # noqa: E402
from hwnas_fpga.models.builder import build_block  # noqa: E402
from hwnas_fpga.runtime import build_constraints, build_search_space, load_config  # noqa: E402
from hwnas_fpga.search_space import BlockSpec, ResolvedBlockSpec  # noqa: E402


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_csv(value: str, *, cast=str) -> tuple[Any, ...]:
    return tuple(cast(item.strip()) for item in str(value).split(",") if item.strip())


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_formal_status_entries(
    path: Path,
    *,
    accept_deployable_measured_implementation: bool,
) -> dict[Any, FormalLutStatusEntry]:
    if not accept_deployable_measured_implementation:
        return LutQueryEngine.load_formal_status_json(str(path))

    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"Unsupported formal LUT status JSON format: {path}")

    index: dict[Any, FormalLutStatusEntry] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            continue
        entry_payload = dict(raw_entry)
        if (
            str(entry_payload.get("status", "")).strip() == "measured_implementation"
            and truthy(entry_payload.get("deployable_at_200mhz"))
        ):
            entry_payload["status"] = "measured"
            entry_payload.setdefault("defer_reason", "deployable_measured_implementation")
        entry = FormalLutStatusEntry.from_dict(entry_payload)
        if entry.op_spec is None:
            raise ValueError(
                f"Formal LUT status entry for case {entry.case_name} is missing op_spec."
            )
        index[entry.op_spec] = entry
    return index


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_route_gate_module() -> Any:
    script_path = REPO_ROOT / "hls_lut_builder" / "board_harness" / "scripts" / "pareto_route_gate.py"
    spec = importlib.util.spec_from_file_location("pareto_route_gate_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load route gate script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def operator_policy(manifest: Mapping[str, Any], op: str) -> dict[str, Any]:
    status = manifest.get("operator_status")
    if isinstance(status, Mapping) and isinstance(status.get(op), Mapping):
        return dict(status[op])
    return {}


def manifest_operator_group(manifest: Mapping[str, Any], op: str) -> str:
    operators = manifest.get("operators")
    if not isinstance(operators, Mapping):
        return "missing"
    for group_name, values in operators.items():
        if isinstance(values, list) and op in values:
            return str(group_name)
    return "missing"


def candidate_kernel_enabled(candidate_kernels: Mapping[str, Any], op: str) -> bool | None:
    operators = candidate_kernels.get("operators")
    if not isinstance(operators, Mapping):
        return None
    entry = operators.get(op)
    if not isinstance(entry, Mapping):
        return None
    return bool(entry.get("enabled", False))


def resolved_stage_shapes(config_path: Path) -> tuple[dict[str, int], ...]:
    config = load_config(str(config_path))
    dataset_cfg = config.get("dataset", {})
    constraints = build_constraints(config)
    search_space = build_search_space(
        config,
        image_size=int(dataset_cfg.get("image_size", 224)),
        input_channels=int(dataset_cfg.get("input_channels", 1)),
        num_classes=int(dataset_cfg.get("num_classes", 8)),
        constraints=constraints,
    )
    baseline = search_space.baseline_architecture()
    shapes = []
    for block in search_space.resolve_blocks(baseline):
        shapes.append(
            {
                "stage": int(block.stage_index),
                "block": int(block.block_index),
                "in_channels": int(block.in_channels),
                "out_channels": int(block.out_channels),
                "input_resolution": int(block.input_resolution),
                "output_resolution": int(block.output_resolution),
                "stride": int(block.stride),
            }
        )
    return tuple(shapes)


def baseline_encoding(config_path: Path) -> dict[str, Any]:
    config = load_config(str(config_path))
    dataset_cfg = config.get("dataset", {})
    constraints = build_constraints(config)
    search_space = build_search_space(
        config,
        image_size=int(dataset_cfg.get("image_size", 224)),
        input_channels=int(dataset_cfg.get("input_channels", 1)),
        num_classes=int(dataset_cfg.get("num_classes", 8)),
        constraints=constraints,
    )
    return search_space.baseline_architecture().to_dict()


def row_for_candidate(
    *,
    op: str,
    kernel: int,
    stage_shape: Mapping[str, int],
    estimator: FPGACostEstimator,
    engine: LutQueryEngine,
    base_encoding: Mapping[str, Any],
    route_gate: Any,
    lowdsp_catalog: Mapping[str, Any],
    manifest: Mapping[str, Any],
    candidate_kernels: Mapping[str, Any],
) -> dict[str, Any]:
    expand = 1
    block = ResolvedBlockSpec(
        stage_index=int(stage_shape["stage"]),
        block_index=int(stage_shape["block"]),
        op=op,
        kernel_size=int(kernel),
        expand_ratio=expand,
        stride=int(stage_shape["stride"]),
        in_channels=int(stage_shape["in_channels"]),
        out_channels=int(stage_shape["out_channels"]),
        input_resolution=int(stage_shape["input_resolution"]),
        output_resolution=int(stage_shape["output_resolution"]),
    )
    op_spec = estimator._block_to_op_spec(block)
    query = engine.query_with_status(op_spec)

    try:
        build_block(
            BlockSpec(op=op, kernel_size=int(kernel), expand_ratio=expand, stride=int(stage_shape["stride"])),
            int(stage_shape["in_channels"]),
            int(stage_shape["out_channels"]),
        )
        model_builder_status = "ok"
    except Exception as exc:  # pragma: no cover - diagnostic path
        model_builder_status = f"error:{exc}"

    try:
        estimator._block_complexity(block)
        estimator._block_to_op_spec(block)
        cost_estimator_status = "ok"
    except Exception as exc:  # pragma: no cover - diagnostic path
        cost_estimator_status = f"error:{exc}"

    encoding = deepcopy(dict(base_encoding))
    stage_index = int(stage_shape["stage"])
    encoding["stages"][stage_index]["blocks"][0] = {
        "op": op,
        "kernel_size": int(kernel),
        "expand_ratio": expand,
        "stride": int(stage_shape["stride"]),
    }
    lowdsp = route_gate.resolve_lowdsp_overrides(encoding, lowdsp_catalog)
    policy = operator_policy(manifest, op)

    return {
        "operator": op,
        "kernel": int(kernel),
        "expand": expand,
        "stage": stage_index,
        "block": int(stage_shape["block"]),
        "shape": {
            "input_resolution": int(stage_shape["input_resolution"]),
            "output_resolution": int(stage_shape["output_resolution"]),
            "in_channels": int(stage_shape["in_channels"]),
            "out_channels": int(stage_shape["out_channels"]),
            "stride": int(stage_shape["stride"]),
        },
        "lookup_op": op_spec.op,
        "formal_status": query.status,
        "strict_lut_hit": query.entry is not None,
        "strict_lut_result": "hit" if query.entry is not None else f"miss:{query.status}",
        "formal_case_name": None if query.status_entry is None else query.status_entry.case_name,
        "true_miss": query.entry is None and query.status == "missing",
        "deferred_hit": query.entry is None and query.status not in {"measured", "missing"},
        "model_builder_status": model_builder_status,
        "cost_estimator_status": cost_estimator_status,
        "hls_template_exists": (REPO_ROOT / "hls_lut_builder" / "templates" / f"{op}.cpp.tmpl").exists(),
        "hls_manifest_group": manifest_operator_group(manifest, op),
        "candidate_kernel_enabled": candidate_kernel_enabled(candidate_kernels, op),
        "operator_search_enabled": policy.get("search_enabled"),
        "operator_deployment_status": policy.get("deployment_status"),
        "operator_deployable_at_200mhz": (
            policy.get("hardware_cost_policy", {}) or {}
        ).get("deployable_at_200mhz"),
        "lowdsp_override_status": lowdsp.get("status"),
        "lowdsp_override_covered": lowdsp.get("status") == "covered",
        "lowdsp_missing": list(lowdsp.get("missing") or []),
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Phase0 v4 Sonar-Op Precheck",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "## Conclusion",
        "",
        f"- Status: `{payload['status']}`",
        f"- Can launch Phase0 v4 RL300: `{payload['can_launch_phase0_v4_rl300']}`",
        f"- Strict LUT hits: `{payload['summary']['strict_lut_hits']}` / `{payload['summary']['matrix_rows']}`",
        f"- True LUT misses: `{payload['summary']['true_misses']}`",
        f"- Deferred hits: `{payload['summary']['deferred_hits']}`",
        f"- Low-DSP override covered rows: `{payload['summary']['lowdsp_override_covered_rows']}` / `{payload['summary']['matrix_rows']}`",
        "",
        "## Coverage Matrix",
        "",
        "| operator | kernel | expand | stage | shape | strict LUT | formal status | low-DSP override | HLS/operator policy |",
        "|---|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in payload["coverage_matrix"]:
        shape = row["shape"]
        shape_text = (
            f"{shape['input_resolution']}:{shape['in_channels']}"
            f"->{shape['output_resolution']}:{shape['out_channels']} s{shape['stride']}"
        )
        hls_text = (
            f"group={row['hls_manifest_group']}; "
            f"kernel_enabled={row['candidate_kernel_enabled']}; "
            f"deploy={row['operator_deployment_status']}; "
            f"200MHz={row['operator_deployable_at_200mhz']}"
        )
        lines.append(
            "| "
            f"`{row['operator']}` | {row['kernel']} | {row['expand']} | {row['stage']} | "
            f"`{shape_text}` | {row['strict_lut_result']} | {row['formal_status']} | "
            f"{row['lowdsp_override_status']} | `{hls_text}` |"
        )

    lines.extend(
        [
            "",
            "## Blocking Items",
            "",
        ]
    )
    for item in payload["blocking_items"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Phase0 v4 Search-Space Draft",
            "",
        ]
    )
    for item in payload["search_space_draft"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Guardrails",
            "",
        ]
    )
    for item in payload["guardrails"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Phase0 v4 sonar operator coverage")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--lut", default=str(DEFAULT_LUT))
    parser.add_argument("--status", default=str(DEFAULT_STATUS))
    parser.add_argument("--operator-manifest", default=str(DEFAULT_OPERATOR_MANIFEST))
    parser.add_argument("--candidate-kernels", default=str(DEFAULT_CANDIDATE_KERNELS))
    parser.add_argument(
        "--lowdsp-catalog",
        action="append",
        default=[str(DEFAULT_LOWDSP_CATALOG)],
        help="Low-DSP override catalog path. May be repeated to layer v3 and pilot catalogs.",
    )
    parser.add_argument("--operators", default="denoise,edge,mixconv")
    parser.add_argument("--kernels", default="3,5")
    parser.add_argument("--stages", default="1,2,3")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--no-accept-deployable-measured-implementation",
        action="store_true",
        help=(
            "Do not normalize deployable measured_implementation status rows to measured. "
            "The default keeps isolated manifest-LUT pilots query-compatible with strict precheck."
        ),
    )
    parser.add_argument(
        "--blocked-exit-zero",
        action="store_true",
        help="Write BLOCKED status but return zero. Useful when generating reports in a larger workflow.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    lut_path = Path(args.lut).expanduser().resolve()
    status_path = Path(args.status).expanduser().resolve()
    manifest_path = Path(args.operator_manifest).expanduser().resolve()
    candidate_kernels_path = Path(args.candidate_kernels).expanduser().resolve()
    lowdsp_catalog_paths = [Path(raw_path).expanduser().resolve() for raw_path in (args.lowdsp_catalog or [])]
    output_dir = Path(args.output_dir).expanduser().resolve()

    operators = parse_csv(args.operators, cast=str)
    kernels = parse_csv(args.kernels, cast=int)
    stages = set(parse_csv(args.stages, cast=int))

    lut_table = LutTable.load_json(str(lut_path))
    accept_measured_impl = not args.no_accept_deployable_measured_implementation
    status_entries = load_formal_status_entries(
        status_path,
        accept_deployable_measured_implementation=accept_measured_impl,
    )
    engine = LutQueryEngine(
        lut_table,
        strict_formal_lut=True,
        allow_shape_only_match=False,
        formal_status_entries=status_entries,
    )
    estimator = FPGACostEstimator(
        HardwareSpec(name="phase0_v4_sonar_precheck", clock_mhz=200),
        lut_query_engine=engine,
        strict_formal_lut=True,
        strict_lut_label="strict board LUT",
    )
    route_gate = load_route_gate_module()
    lowdsp_catalog = route_gate.load_lowdsp_override_catalog([str(path) for path in lowdsp_catalog_paths])
    manifest = read_yaml(manifest_path)
    candidate_kernels = read_yaml(candidate_kernels_path)
    base_encoding = baseline_encoding(config_path)
    shapes = tuple(shape for shape in resolved_stage_shapes(config_path) if shape["stage"] in stages)

    coverage_matrix = [
        row_for_candidate(
            op=op,
            kernel=kernel,
            stage_shape=shape,
            estimator=estimator,
            engine=engine,
            base_encoding=base_encoding,
            route_gate=route_gate,
            lowdsp_catalog=lowdsp_catalog,
            manifest=manifest,
            candidate_kernels=candidate_kernels,
        )
        for op in operators
        for kernel in kernels
        for shape in shapes
    ]

    strict_hits = sum(1 for row in coverage_matrix if row["strict_lut_hit"])
    true_misses = sum(1 for row in coverage_matrix if row["true_miss"])
    deferred_hits = sum(1 for row in coverage_matrix if row["deferred_hit"])
    lowdsp_covered = sum(1 for row in coverage_matrix if row["lowdsp_override_covered"])
    model_errors = [row for row in coverage_matrix if row["model_builder_status"] != "ok"]
    cost_errors = [row for row in coverage_matrix if row["cost_estimator_status"] != "ok"]

    blocking_items: list[str] = []
    if true_misses:
        blocking_items.append(
            f"strict LUT has {true_misses} true misses for sonar-op candidate rows"
        )
    if deferred_hits:
        blocking_items.append(
            f"formal LUT status has {deferred_hits} deferred/infeasible rows"
        )
    if lowdsp_covered != len(coverage_matrix):
        blocking_items.append(
            f"low-DSP route override catalog covers {lowdsp_covered}/{len(coverage_matrix)} sonar-op rows"
        )
    if model_errors:
        blocking_items.append(f"model builder errors: {len(model_errors)} rows")
    if cost_errors:
        blocking_items.append(f"cost estimator errors: {len(cost_errors)} rows")

    can_launch = not blocking_items and strict_hits == len(coverage_matrix)
    status = "PASS" if can_launch else "BLOCKED"

    search_space_draft = [
        "Keep Phase0 v3 low-DSP skeleton unchanged: fixed stage0 conv k1 e1; stage1 mbconv k3 e3/e6; stage2 mbconv k3 e3; stage3 mbconv k3 e3 or skip.",
        "Do not add any sonar op to the formal Phase0 v4 RL300 config until every admitted row is measured in strict LUT/status and covered by low-DSP route overrides.",
        "First unblocked sonar admission should be the lowest-risk stage3 k3 rows; stage1/stage2 stride-2 sonar rows and all k5 rows need separate HLS/LUT/route evidence.",
        "Keep mixconv in a separate ablation lane until its search/LUT kernel_size semantics are fixed and measured.",
    ]

    payload = {
        "generated_at": timestamp(),
        "runs_server": False,
        "runs_training": False,
        "runs_vivado": False,
        "runs_com_measurement": False,
        "config": str(config_path),
        "lut": str(lut_path),
        "formal_status": str(status_path),
        "operator_manifest": str(manifest_path),
        "candidate_kernels": str(candidate_kernels_path),
        "lowdsp_catalog": [str(path) for path in lowdsp_catalog_paths],
        "accept_deployable_measured_implementation": accept_measured_impl,
        "status": status,
        "can_launch_phase0_v4_rl300": can_launch,
        "config_created": False,
        "summary": {
            "matrix_rows": len(coverage_matrix),
            "strict_lut_hits": strict_hits,
            "true_misses": true_misses,
            "deferred_hits": deferred_hits,
            "lowdsp_override_covered_rows": lowdsp_covered,
            "model_builder_error_rows": len(model_errors),
            "cost_estimator_error_rows": len(cost_errors),
        },
        "coverage_matrix": coverage_matrix,
        "blocking_items": blocking_items,
        "search_space_draft": search_space_draft,
        "guardrails": [
            "Do not overwrite Phase0 v3 rl_arch_186/242/276 artifacts.",
            "Do not launch remote/server RL300 from a BLOCKED precheck.",
            "Do not run Vivado full route or COM5 in this precheck.",
            "Actual Vivado DSP target remains <=700 and final claimability still requires full route gate.",
            "physical_risk and early_expand_pressure stay out of the main reward unless a separate experiment explicitly changes that policy.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sonar_op_coverage.json"
    md_path = output_dir / "sonar_op_coverage.md"
    write_json(json_path, payload)
    md_path.write_text(render_markdown(payload), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if status != "PASS" and not args.blocked_exit_zero:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
