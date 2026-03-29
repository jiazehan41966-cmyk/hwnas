#!/usr/bin/env python3
"""Operator ablation experiment entrypoint for fixed macro templates."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Optional, Sequence

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

import run_backbone_baseline as baseline_utils

from hwnas_fpga.experiment import ExperimentTracker
from hwnas_fpga.hardware import BackboneCostEstimator
from hwnas_fpga.models import build_model, get_macro_template, list_macro_templates
from hwnas_fpga.runtime import (
    build_constraints,
    build_hardware_spec,
    load_config,
    pick,
)
from hwnas_fpga.search_space import ArchitectureSpec, BlockSpec, SearchSpace, SearchSpaceConfig, StageSpec


DEFAULT_SINGLE_OPS = ("dw_pw_conv", "mbconv", "fused_mbconv", "mixconv", "denoise", "edge")
DEFAULT_POOL_ROLE = "search_anchor"
DEFAULT_BACKBONE_TO_MACRO_TEMPLATE = {
    "mobilenet_v2": "mobilenet_v2_like",
    "fbnet_like": "fbnet_like",
    "shufflenet_v2": "mobilenet_v2_like",
    "efficientnet_b0": "mobilenet_v2_like",
    "simplecnn": "mobilenet_v2_like",
}
DEFAULT_ROLE_TO_MACRO_TEMPLATE = {
    "search_anchor": None,
    "accuracy_anchor": None,
    "lightweight_anchor": None,
}


@dataclass(frozen=True)
class OperatorAblationCandidate:
    arch_id: str
    display_name: str
    stage_ops: tuple[str, ...]
    use_skip_connections: bool = False
    category: str = "single"
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, stage_count: int) -> "OperatorAblationCandidate":
        stage_ops = payload.get("stage_ops", ())
        if isinstance(stage_ops, str):
            stage_ops = [stage_ops]
        stage_ops = tuple(str(op) for op in stage_ops)
        if len(stage_ops) == 1:
            stage_ops = stage_ops * stage_count
        if len(stage_ops) != stage_count:
            raise ValueError(
                f"candidate {payload.get('arch_id', '<unknown>')} expected {stage_count} stage ops, "
                f"got {len(stage_ops)}"
            )
        return cls(
            arch_id=str(payload["arch_id"]),
            display_name=str(payload.get("display_name", payload["arch_id"])),
            stage_ops=stage_ops,
            use_skip_connections=bool(payload.get("use_skip_connections", False)),
            category=str(payload.get("category", "mixed")),
            notes=str(payload.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arch_id": self.arch_id,
            "display_name": self.display_name,
            "stage_ops": list(self.stage_ops),
            "use_skip_connections": self.use_skip_connections,
            "category": self.category,
            "notes": self.notes,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW-NAS operator ablation experiment")
    parser.add_argument("--config", type=str, default="configs/operator_ablation_nksid_av7k325.yaml")
    parser.add_argument("--dataset", type=str, default=None, choices=("dummy", "nksid"))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--macro-template", type=str, default=None, choices=tuple(sorted(list_macro_templates().keys())))
    parser.add_argument(
        "--selected-backbone-pool",
        type=str,
        default=None,
        help="Optional selected_backbone_pool.json emitted by run_backbone_baseline.py",
    )
    parser.add_argument(
        "--pool-role",
        type=str,
        default=None,
        choices=("search_anchor", "accuracy_anchor", "lightweight_anchor"),
        help="Role to resolve from selected_backbone_pool.json",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        help="Run only specific operator ablation candidates; repeatable",
    )
    parser.add_argument("--cpu-latency-runs", type=int, default=None)
    parser.add_argument("--cpu-latency-warmup", type=int, default=None)
    return parser.parse_args()


def _prettify_op_name(op: str) -> str:
    labels = {
        "dw_pw_conv": "DW-PW Conv",
        "mbconv": "MBConv",
        "fused_mbconv": "Fused MBConv",
        "skip": "Skip",
        "mixconv": "MixConv",
        "denoise": "Denoise",
        "edge": "Edge",
    }
    return labels.get(op, op)


def _default_mixed_candidates(stage_count: int) -> tuple[OperatorAblationCandidate, ...]:
    fused_prefix = ["fused_mbconv"] * min(2, stage_count)
    mbconv_suffix = ["mbconv"] * max(0, stage_count - len(fused_prefix))
    mobile_core_ops = tuple(fused_prefix + mbconv_suffix)

    return (
        OperatorAblationCandidate(
            arch_id="mobile_core",
            display_name="Mobile Core (Fused->MBConv)",
            stage_ops=mobile_core_ops,
            use_skip_connections=False,
            category="mixed",
            notes="fused blocks in shallow stages, mbconv in deeper stages",
        ),
        OperatorAblationCandidate(
            arch_id="mobile_core_skip",
            display_name="Mobile Core + Skip",
            stage_ops=mobile_core_ops,
            use_skip_connections=True,
            category="mixed",
            notes="residual interior blocks replaced with skip connections",
        ),
        OperatorAblationCandidate(
            arch_id="mobile_plus_mixconv",
            display_name="Mobile + MixConv",
            stage_ops=("mixconv",) + mobile_core_ops[1:],
            use_skip_connections=True,
            category="mixed",
            notes="mixconv reserved for the first stage",
        ),
        OperatorAblationCandidate(
            arch_id="mobile_plus_denoise",
            display_name="Mobile + Denoise",
            stage_ops=("denoise",) + mobile_core_ops[1:],
            use_skip_connections=True,
            category="mixed",
            notes="denoise reserved for the first stage",
        ),
        OperatorAblationCandidate(
            arch_id="mobile_plus_edge",
            display_name="Mobile + Edge",
            stage_ops=("edge",) + mobile_core_ops[1:],
            use_skip_connections=True,
            category="mixed",
            notes="edge-aware operator reserved for the first stage",
        ),
    )


def resolve_ablation_candidates(
    ablation_cfg: dict[str, Any],
    *,
    stage_count: int,
    requested: Optional[Iterable[str]],
) -> list[OperatorAblationCandidate]:
    candidates: list[OperatorAblationCandidate] = []
    single_ops = [str(op) for op in ablation_cfg.get("single_ops", DEFAULT_SINGLE_OPS)]
    for op in single_ops:
        candidates.append(
            OperatorAblationCandidate(
                arch_id=f"{op}_only",
                display_name=f"{_prettify_op_name(op)} only",
                stage_ops=(op,) * stage_count,
                use_skip_connections=False,
                category="single",
                notes="single-operator macro template",
            )
        )

    raw_mixed_candidates = ablation_cfg.get("mixed_candidates")
    if raw_mixed_candidates:
        candidates.extend(
            OperatorAblationCandidate.from_dict(candidate, stage_count=stage_count)
            for candidate in raw_mixed_candidates
        )
    else:
        candidates.extend(_default_mixed_candidates(stage_count))

    if not requested:
        return candidates

    filters = {item.strip().lower() for item in requested if item and item.strip()}
    selected = [candidate for candidate in candidates if candidate.arch_id.lower() in filters]
    if not selected:
        raise ValueError(f"No operator ablation candidates matched filters: {sorted(filters)}")
    return selected


def build_search_space_for_template(
    template: dict[str, Any],
    *,
    input_channels: int,
    image_size: int,
    num_classes: int,
    constraints: Any,
    op_choices: Sequence[str],
) -> SearchSpace:
    stage_specs = tuple(template["stages"])
    return SearchSpace(
        SearchSpaceConfig(
            input_channels=input_channels,
            image_size=image_size,
            stem_channels=int(template["stem_channels"]),
            stem_stride=int(template["stem_stride"]),
            stage_strides=tuple(int(stage["stride"]) for stage in stage_specs),
            channel_choices=tuple(sorted({int(stage["channels"]) for stage in stage_specs})),
            depth_choices=tuple(sorted({int(stage["depth"]) for stage in stage_specs})),
            kernel_choices=tuple(sorted({1, *[int(stage["kernel"]) for stage in stage_specs]})),
            expand_choices=tuple(sorted({1, *[int(stage["expand"]) for stage in stage_specs]})),
            op_choices=tuple(op_choices),
            head_channels=_template_head_channels(template),
            num_classes=num_classes,
            hardware_constraints=constraints,
        )
    )


def _block_kernel_for_op(op: str, default_kernel: int) -> int:
    return 1 if op == "skip" else int(default_kernel)


def _block_expand_for_op(op: str, default_expand: int) -> int:
    if op in {"mbconv", "fused_mbconv"}:
        return int(default_expand)
    return 1


def build_architecture_from_template(
    template: dict[str, Any],
    candidate: OperatorAblationCandidate,
    *,
    input_channels: int,
    num_classes: int,
) -> ArchitectureSpec:
    stages: list[StageSpec] = []
    for stage_index, stage_template in enumerate(template["stages"]):
        blocks: list[BlockSpec] = []
        depth = int(stage_template["depth"])
        stride = int(stage_template["stride"])
        stage_op = candidate.stage_ops[stage_index]
        for block_index in range(depth):
            block_stride = stride if block_index == 0 else 1
            block_op = "skip" if candidate.use_skip_connections and block_index > 0 else stage_op
            blocks.append(
                BlockSpec(
                    op=block_op,
                    kernel_size=_block_kernel_for_op(block_op, int(stage_template["kernel"])),
                    expand_ratio=_block_expand_for_op(block_op, int(stage_template["expand"])),
                    stride=block_stride,
                )
            )
        stages.append(
            StageSpec(
                channels=int(stage_template["channels"]),
                depth=depth,
                stride=stride,
                blocks=tuple(blocks),
            )
        )

    return ArchitectureSpec(
        input_channels=input_channels,
        stem_channels=int(template["stem_channels"]),
        stem_stride=int(template["stem_stride"]),
        stages=tuple(stages),
        head_channels=_template_head_channels(template),
        num_classes=num_classes,
    )


def _result_sort_key(result: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(result["evaluation"]["macro_f1"]),
        float(result["evaluation"]["top1"]),
        -float(result["cost_estimate"]["cpu_latency_ms"]),
    )


def build_operator_retention_table(
    results: list[dict[str, Any]],
    *,
    ablation_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if not results:
        return []

    retention_cfg = dict(ablation_cfg.get("retention", {}))
    keep_margin = float(retention_cfg.get("keep_margin", 0.02))
    conditional_margin = float(retention_cfg.get("conditional_margin", 0.05))
    best_macro = max(float(result["evaluation"]["macro_f1"]) for result in results)
    all_ops = sorted({op for result in results for op in result["ops_used"]})

    table: list[dict[str, Any]] = []
    for op in all_ops:
        op_results = [result for result in results if op in result["ops_used"]]
        best_result = max(op_results, key=_result_sort_key)
        best_macro_f1 = float(best_result["evaluation"]["macro_f1"])
        macro_gap = max(0.0, best_macro - best_macro_f1)
        feasible_runs = sum(1 for result in op_results if bool(result["cost_estimate"]["feasible"]))
        best_latency = min(float(result["cost_estimate"]["latency_ms"]) for result in op_results)

        if feasible_runs > 0 and macro_gap <= keep_margin:
            status = "keep"
            rationale = "close to the best macro_f1 with at least one feasible run"
        elif macro_gap <= conditional_margin:
            status = "conditional"
            rationale = "competitive enough to keep as a stage-limited option"
        else:
            status = "drop"
            rationale = "clearly behind the best operator set under the fixed macro template"

        table.append(
            {
                "op": op,
                "display_name": _prettify_op_name(op),
                "status": status,
                "best_macro_f1": best_macro_f1,
                "best_top1": float(best_result["evaluation"]["top1"]),
                "best_fpga_latency_ms": float(best_result["cost_estimate"]["latency_ms"]),
                "best_cpu_latency_ms": float(best_result["cost_estimate"]["cpu_latency_ms"]),
                "feasible_runs": feasible_runs,
                "total_runs": len(op_results),
                "best_candidate": best_result["arch_id"],
                "macro_gap_to_best": macro_gap,
                "rationale": rationale,
            }
        )

    return table


def write_ablation_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "arch_id",
        "display_name",
        "macro_template",
        "macro_template_source",
        "selected_pool_role",
        "selected_pool_backbone",
        "category",
        "stage_ops",
        "use_skip_connections",
        "ops_used",
        "params",
        "macs",
        "cpu_latency_ms",
        "fpga_latency_ms",
        "top1",
        "top5",
        "macro_f1",
        "weighted_f1",
        "peak_dsp",
        "peak_bram",
        "peak_lut",
        "power_w",
        "feasible",
        "violations",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_retention_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "op",
        "display_name",
        "status",
        "best_macro_f1",
        "best_top1",
        "best_fpga_latency_ms",
        "best_cpu_latency_ms",
        "feasible_runs",
        "total_runs",
        "best_candidate",
        "macro_gap_to_best",
        "rationale",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _normalize_mapping(mapping: Any) -> dict[str, str | None]:
    if mapping is None:
        return {}
    if not isinstance(mapping, dict):
        raise ValueError("mapping overrides must be a dictionary")
    normalized: dict[str, str | None] = {}
    for raw_key, raw_value in mapping.items():
        key = str(raw_key).strip().lower()
        value = None if raw_value is None else str(raw_value).strip()
        normalized[key] = value
    return normalized


def _template_head_channels(template: dict[str, Any]) -> Optional[int]:
    head_channels = template.get("head_channels")
    return None if head_channels is None else int(head_channels)


def resolve_macro_template_selection(
    *,
    args: argparse.Namespace,
    ablation_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.macro_template is not None:
        template = get_macro_template(args.macro_template)
        return template, {
            "macro_template": template["name"],
            "macro_template_display_name": template["display_name"],
            "macro_template_source": "cli",
        }

    pool_cfg = dict(ablation_cfg.get("selected_backbone_pool", {}))
    selected_pool_path = pick(args.selected_backbone_pool, pool_cfg.get("path"), None)
    selected_pool_role = str(pick(args.pool_role, pool_cfg.get("role"), DEFAULT_POOL_ROLE))
    if selected_pool_path:
        pool_path = Path(str(selected_pool_path)).expanduser()
        if not pool_path.is_absolute():
            pool_path = (Path.cwd() / pool_path).resolve()
        if not pool_path.exists():
            raise FileNotFoundError(f"selected_backbone_pool.json not found: {pool_path}")

        pool_payload = _load_json(pool_path)
        roles = pool_payload.get("roles", {})
        if not isinstance(roles, dict) or selected_pool_role not in roles:
            raise ValueError(
                f"Role '{selected_pool_role}' not found in selected_backbone_pool.json: {pool_path}"
            )

        role_payload = roles[selected_pool_role]
        if not isinstance(role_payload, dict):
            raise ValueError(
                f"Role '{selected_pool_role}' must map to an object in selected_backbone_pool.json"
            )

        role_to_template = {
            **DEFAULT_ROLE_TO_MACRO_TEMPLATE,
            **_normalize_mapping(pool_cfg.get("role_to_macro_template")),
        }
        backbone_to_template = {
            **DEFAULT_BACKBONE_TO_MACRO_TEMPLATE,
            **_normalize_mapping(pool_cfg.get("backbone_to_macro_template")),
        }

        backbone_name = str(role_payload.get("backbone", "")).strip().lower()
        resolved_template_name = role_to_template.get(selected_pool_role)
        resolution_rule = "role_to_macro_template"
        if not resolved_template_name:
            resolved_template_name = backbone_to_template.get(backbone_name)
            resolution_rule = "backbone_to_macro_template"
        if not resolved_template_name:
            raise ValueError(
                "Unable to map selected_backbone_pool role "
                f"'{selected_pool_role}' with backbone '{backbone_name}' to a macro template"
            )

        template = get_macro_template(resolved_template_name)
        return template, {
            "macro_template": template["name"],
            "macro_template_display_name": template["display_name"],
            "macro_template_source": "selected_backbone_pool",
            "selected_backbone_pool_path": str(pool_path),
            "selected_backbone_pool_role": selected_pool_role,
            "selected_backbone_pool_arch_id": role_payload.get("arch_id"),
            "selected_backbone_pool_display_name": role_payload.get("display_name"),
            "selected_backbone_pool_backbone": role_payload.get("backbone"),
            "selected_backbone_pool_resolution_rule": resolution_rule,
        }

    config_template_name = ablation_cfg.get("macro_template")
    if config_template_name is not None:
        template = get_macro_template(str(config_template_name))
        return template, {
            "macro_template": template["name"],
            "macro_template_display_name": template["display_name"],
            "macro_template_source": "config",
        }

    template = get_macro_template("fbnet_like")
    return template, {
        "macro_template": template["name"],
        "macro_template_display_name": template["display_name"],
        "macro_template_source": "default",
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    project_cfg = config.get("project", {})
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    hardware_cfg = config.get("hardware", {})
    ablation_cfg = config.get("ablation", {})

    seed = int(pick(args.seed, project_cfg.get("seed"), 42))
    baseline_utils.set_seed(seed)

    dataset_name = pick(args.dataset, dataset_cfg.get("name"), "dummy")
    if dataset_name not in {"dummy", "nksid"}:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    default_device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    device = str(pick(args.device, None, default_device))
    image_size = int(pick(args.image_size, dataset_cfg.get("image_size"), 224))
    input_channels = int(dataset_cfg.get("input_channels", 1))
    batch_size = int(pick(args.batch_size, training_cfg.get("batch_size"), 32))
    num_workers = int(pick(args.num_workers, dataset_cfg.get("num_workers"), 0))
    fold = int(pick(args.fold, dataset_cfg.get("fold"), 0))
    data_dir = pick(args.data_dir, dataset_cfg.get("data_dir"), None)
    num_classes = pick(args.num_classes, dataset_cfg.get("num_classes"), None)
    if data_dir is not None:
        dataset_cfg["data_dir"] = data_dir

    training_cfg = {
        **training_cfg,
        "epochs": int(pick(args.epochs, training_cfg.get("epochs"), 10)),
        "batch_size": batch_size,
        "selection_metric": str(ablation_cfg.get("selection_metric", "macro_f1")),
    }
    cpu_latency_runs = int(pick(args.cpu_latency_runs, hardware_cfg.get("cpu_latency_runs"), 30))
    cpu_latency_warmup = int(pick(args.cpu_latency_warmup, hardware_cfg.get("cpu_latency_warmup"), 10))
    topk = int(ablation_cfg.get("topk_accuracy", 5))

    template, template_selection = resolve_macro_template_selection(args=args, ablation_cfg=ablation_cfg)
    candidates = resolve_ablation_candidates(
        ablation_cfg,
        stage_count=len(template["stages"]),
        requested=args.candidate,
    )

    output_dir = pick(args.output_dir, project_cfg.get("output_dir"), "results")
    run_name = pick(args.run_name, project_cfg.get("run_name"), None)
    output_root = Path(output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()

    constraints = build_constraints(config)
    hardware_spec = build_hardware_spec(config)
    estimator = BackboneCostEstimator(
        hardware_spec=hardware_spec,
        constraints=constraints,
        quantization_bits=int(hardware_cfg.get("quantization_bits", 8)),
        cpu_latency_warmup=cpu_latency_warmup,
        cpu_latency_runs=cpu_latency_runs,
    )

    tracker = ExperimentTracker(
        output_root=output_root,
        project_name=str(project_cfg.get("name", "operator-ablation")),
        script_name="run_operator_ablation",
        search_method="operator_ablation",
        dataset_name=str(dataset_name),
        run_name=run_name,
    )
    tracker.save_config(config, vars(args))

    try:
        with tracker.capture_console():
            print(f"Run dir: {tracker.root_dir}")
            print(f"Using device: {device}")
            print(f"Dataset: {dataset_name}")
            print(f"Image size: {image_size}")
            print(f"Input channels: {input_channels}")
            print(f"Board: {hardware_spec.name} @ {hardware_spec.clock_mhz}MHz")
            print(f"Macro template: {template['display_name']} ({template['name']})")
            print(f"Macro template source: {template_selection['macro_template_source']}")
            if template_selection.get("selected_backbone_pool_role"):
                print(
                    "Selected backbone pool: "
                    f"{template_selection['selected_backbone_pool_role']} -> "
                    f"{template_selection.get('selected_backbone_pool_backbone')} "
                    f"({template_selection.get('selected_backbone_pool_display_name')})"
                )
            print(f"Operator ablation candidates: {[candidate.arch_id for candidate in candidates]}")

            train_loader, val_loader, class_weights, resolved_num_classes, class_names = (
                baseline_utils.create_baseline_data_pipeline(
                    dataset_name=str(dataset_name),
                    dataset_cfg=dataset_cfg,
                    batch_size=batch_size,
                    image_size=image_size,
                    input_channels=input_channels,
                    num_classes=num_classes,
                    fold=fold,
                    num_workers=num_workers,
                    device=device,
                )
            )
            tracker.write_dataset_summary(
                {
                    "dataset": dataset_name,
                    "data_dir": data_dir,
                    "fold": fold,
                    "image_size": image_size,
                    "input_channels": input_channels,
                    "batch_size": batch_size,
                    "train_samples": len(train_loader.dataset),
                    "val_samples": len(val_loader.dataset),
                    "num_classes": resolved_num_classes,
                    "class_names": class_names,
                    "class_weights": None if class_weights is None else class_weights.tolist(),
                }
            )

            op_choices = tuple(
                sorted(
                    {
                        op
                        for candidate in candidates
                        for op in (
                            list(candidate.stage_ops)
                            + (["skip"] if candidate.use_skip_connections else [])
                        )
                    }
                )
            )
            search_space = build_search_space_for_template(
                template,
                input_channels=input_channels,
                image_size=image_size,
                num_classes=resolved_num_classes,
                constraints=constraints,
                op_choices=op_choices,
            )
            tracker.write_search_space_summary(
                {
                    "macro_template": template["name"],
                    "stage_count": len(template["stages"]),
                    "stem_channels": template["stem_channels"],
                    "stem_stride": template["stem_stride"],
                    "head_channels": template["head_channels"],
                    "stage_specs": [dict(stage) for stage in template["stages"]],
                    "op_choices": list(op_choices),
                    **template_selection,
                }
            )

            results_dir = tracker.results_dir / "operators"
            results_dir.mkdir(parents=True, exist_ok=True)

            summary_rows: list[dict[str, Any]] = []
            all_results: list[dict[str, Any]] = []
            best_result: dict[str, Any] | None = None

            for candidate in candidates:
                print(f"\n=== Candidate: {candidate.display_name} ({candidate.arch_id}) ===")
                architecture = build_architecture_from_template(
                    template,
                    candidate,
                    input_channels=input_channels,
                    num_classes=resolved_num_classes,
                )
                validation_errors = search_space.validate(architecture)
                if validation_errors:
                    raise ValueError(
                        f"Candidate {candidate.arch_id} produced illegal architecture: {validation_errors}"
                    )

                model = build_model(
                    architecture,
                    num_classes=resolved_num_classes,
                    head_channels=_template_head_channels(template),
                )
                cost_estimate = estimator.estimate(
                    model,
                    input_shape=(1, input_channels, image_size, image_size),
                )
                print(
                    "Complexity: "
                    f"params={cost_estimate.params:,}, macs={cost_estimate.macs:,}, "
                    f"cpu_latency={cost_estimate.cpu_latency_ms:.2f}ms, "
                    f"fpga_latency={cost_estimate.latency_ms:.2f}ms"
                )
                print(
                    "Hardware: "
                    f"DSP={cost_estimate.peak_dsp}, BRAM={cost_estimate.peak_bram}, "
                    f"LUT={cost_estimate.peak_lut}, power={cost_estimate.power_w:.2f}W, "
                    f"feasible={cost_estimate.feasible}"
                )

                trained_model, training_summary, history, evaluation = baseline_utils.train_backbone(
                    model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    training_cfg=training_cfg,
                    class_weights=class_weights,
                    device=device,
                    num_classes=resolved_num_classes,
                    class_names=class_names,
                    topk=topk,
                )

                result = {
                    "arch_id": candidate.arch_id,
                    "display_name": candidate.display_name,
                    "macro_template": template["name"],
                    "macro_template_selection": dict(template_selection),
                    "candidate": candidate.to_dict(),
                    "architecture": architecture.to_dict(),
                    "evaluation": evaluation,
                    "training_summary": training_summary,
                    "history": history,
                    "cost_estimate": cost_estimate.to_dict(),
                    "ops_used": sorted(set(candidate.stage_ops) | ({"skip"} if candidate.use_skip_connections else set())),
                }
                baseline_utils.write_json(results_dir / f"{candidate.arch_id}.json", result)
                checkpoint_payload = {
                    "arch_id": candidate.arch_id,
                    "display_name": candidate.display_name,
                    "macro_template": template["name"],
                    "macro_template_selection": dict(template_selection),
                    "candidate": candidate.to_dict(),
                    "architecture": architecture.to_dict(),
                    "cost_estimate": cost_estimate.to_dict(),
                    "training_summary": training_summary,
                    "evaluation": evaluation,
                    "history": history,
                    "model_state_dict": trained_model.state_dict(),
                }
                tracker.save_named_checkpoint(f"{candidate.arch_id}.pt", checkpoint_payload)

                summary_row = {
                    "arch_id": candidate.arch_id,
                    "display_name": candidate.display_name,
                    "macro_template": template["name"],
                    "macro_template_source": template_selection["macro_template_source"],
                    "selected_pool_role": template_selection.get("selected_backbone_pool_role"),
                    "selected_pool_backbone": template_selection.get("selected_backbone_pool_backbone"),
                    "category": candidate.category,
                    "stage_ops": ",".join(candidate.stage_ops),
                    "use_skip_connections": candidate.use_skip_connections,
                    "ops_used": ",".join(result["ops_used"]),
                    "params": int(cost_estimate.params),
                    "macs": int(cost_estimate.macs),
                    "cpu_latency_ms": float(cost_estimate.cpu_latency_ms),
                    "fpga_latency_ms": float(cost_estimate.latency_ms),
                    "top1": float(evaluation["top1"]),
                    "top5": float(evaluation["top5"]),
                    "macro_f1": float(evaluation["macro_f1"]),
                    "weighted_f1": float(evaluation["weighted_f1"]),
                    "peak_dsp": int(cost_estimate.peak_dsp),
                    "peak_bram": int(cost_estimate.peak_bram),
                    "peak_lut": int(cost_estimate.peak_lut),
                    "power_w": float(cost_estimate.power_w),
                    "feasible": bool(cost_estimate.feasible),
                    "violations": ";".join(cost_estimate.violations),
                }
                summary_rows.append(summary_row)
                all_results.append(result)
                best_result = result if baseline_utils.compare_results(result, best_result) else best_result

            retention_table = build_operator_retention_table(all_results, ablation_cfg=ablation_cfg)
            baseline_utils.write_json(tracker.results_dir / "operator_ablation_summary.json", summary_rows)
            write_ablation_summary_csv(tracker.results_dir / "operator_ablation_summary.csv", summary_rows)
            baseline_utils.write_json(tracker.results_dir / "operator_retention_table.json", retention_table)
            write_retention_csv(tracker.results_dir / "operator_retention_table.csv", retention_table)

            if best_result is not None:
                baseline_utils.write_json(tracker.results_dir / "best_operator_candidate.json", best_result)
                print(
                    f"\nSelected operator candidate: {best_result['display_name']} "
                    f"(macro_f1={best_result['evaluation']['macro_f1']:.4f}, "
                    f"feasible={best_result['cost_estimate']['feasible']})"
                )

            tracker.finalize(
                status="completed",
                candidates=[],
                feasible_candidates=[],
                infeasible_candidates=[],
                best_candidate=None,
                pareto_candidates=None,
                extra={
                    "macro_template": template["name"],
                    "macro_template_selection": template_selection,
                    "operator_candidates": [candidate.to_dict() for candidate in candidates],
                    "best_operator_candidate": None if best_result is None else best_result["arch_id"],
                    "retention_table": retention_table,
                },
            )
    except Exception as exc:
        tracker.finalize(
            status="failed",
            candidates=[],
            feasible_candidates=[],
            infeasible_candidates=[],
            best_candidate=None,
            pareto_candidates=None,
            extra={"error": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
