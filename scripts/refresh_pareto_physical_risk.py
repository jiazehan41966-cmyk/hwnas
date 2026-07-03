#!/usr/bin/env python3
"""Rebuild Pareto artifacts with physical-risk metrics for an existing candidate pool."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hwnas_fpga.interfaces import CandidateMetrics, SearchCandidate  # noqa: E402
from hwnas_fpga.runtime import (  # noqa: E402
    build_constraints,
    build_cost_estimator,
    build_hardware_spec,
    build_search_space,
    load_config,
)
from hwnas_fpga.search.pareto import (  # noqa: E402
    ParetoFrontSelector,
    build_pareto_objectives,
    build_pareto_selection_summary,
    compute_pareto_front,
    compute_pareto_ranks,
    write_pareto_selection_artifacts,
)
from hwnas_fpga.search_space import ArchitectureSpec  # noqa: E402


DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "search"
    / "nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_av7k325.yaml"
)
DEFAULT_SOURCE_RUN_RESULTS = (
    REPO_ROOT
    / "results_port57244"
    / "nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_port57244_v1"
    / "results"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "phase1_pareto_physical_risk_reanalysis_port57244"
)


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh physical-risk-aware Pareto artifacts")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--candidates-json", default=str(DEFAULT_SOURCE_RUN_RESULTS / "candidates.json"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--selection-metric", default="")
    parser.add_argument("--topk", type=int, default=None)
    return parser.parse_args()


def _candidate_metrics_from(payload: Mapping[str, Any]) -> CandidateMetrics:
    allowed = {field.name for field in fields(CandidateMetrics)}
    return CandidateMetrics(**{key: value for key, value in payload.items() if key in allowed})


def _merge_metrics(
    original_metrics: Mapping[str, Any],
    refreshed_metrics: CandidateMetrics,
) -> CandidateMetrics:
    merged = dict(original_metrics)
    for key in fields(CandidateMetrics):
        value = getattr(refreshed_metrics, key.name)
        if value is not None:
            merged[key.name] = value
    return _candidate_metrics_from(merged)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    candidates_path = Path(args.candidates_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    config = load_config(str(config_path))
    dataset_cfg = config.get("dataset", {}) if isinstance(config.get("dataset"), dict) else {}
    search_cfg = config.get("search", {}) if isinstance(config.get("search"), dict) else {}
    pareto_cfg = search_cfg.get("pareto", {}) if isinstance(search_cfg.get("pareto"), dict) else {}
    selection_metric = args.selection_metric or str(search_cfg.get("selection_metric") or "macro_f1")
    topk = int(args.topk if args.topk is not None else pareto_cfg.get("topk", 8))
    selection_method = str(pareto_cfg.get("selection_method") or "rank")

    constraints = build_constraints(config)
    search_space = build_search_space(
        config,
        image_size=int(dataset_cfg.get("image_size") or 224),
        input_channels=int(dataset_cfg.get("input_channels") or 1),
        num_classes=(
            None if dataset_cfg.get("num_classes") is None else int(dataset_cfg.get("num_classes"))
        ),
        constraints=constraints,
    )
    hardware_spec = build_hardware_spec(config)
    estimator = build_cost_estimator(
        config,
        hardware_spec=hardware_spec,
        constraints=constraints,
    )

    raw_candidates = read_json(candidates_path)
    if not isinstance(raw_candidates, list):
        raise ValueError(f"candidates JSON must contain a list: {candidates_path}")

    feasible_candidates: list[SearchCandidate] = []
    infeasible_records: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_candidates, start=1):
        if not isinstance(raw, Mapping):
            continue
        encoding = raw.get("encoding")
        if not isinstance(encoding, Mapping):
            continue
        arch_id = str(raw.get("arch_id") or f"candidate_{index}")
        architecture = ArchitectureSpec.from_dict(encoding)
        estimate = estimator.estimate(architecture, search_space)
        original_metrics = raw.get("metrics") if isinstance(raw.get("metrics"), Mapping) else {}
        refreshed_metrics = _merge_metrics(original_metrics, estimate.to_candidate_metrics())
        if estimate.violations:
            infeasible_records.append(
                {
                    "candidate_index": index,
                    "arch_id": arch_id,
                    "violations": list(estimate.violations),
                    "physical_metrics": dict(estimate.physical_metrics),
                }
            )
            continue
        feasible_candidates.append(
            SearchCandidate(
                arch_id=arch_id,
                encoding=dict(encoding),
                metrics=refreshed_metrics,
            )
        )

    objectives, directions = build_pareto_objectives(
        search_cfg.get("objective_weights", {}),
        constraints,
        selection_metric=selection_metric,
    )
    pareto_front = compute_pareto_front(feasible_candidates, objectives, directions)
    ranks = compute_pareto_ranks(feasible_candidates, objectives, directions)
    selector = ParetoFrontSelector(
        objectives=objectives,
        directions=directions,
        selection_method=selection_method,
    )
    selected_candidates = selector.select(feasible_candidates, k=topk)
    summary = build_pareto_selection_summary(
        candidates=feasible_candidates,
        pareto_front=pareto_front,
        selected_candidates=selected_candidates,
        ranks=ranks,
        objectives=objectives,
        directions=directions,
        selection_method=selection_method,
        topk=topk,
        hypervolume=0.0,
    )
    summary.update(
        {
            "generated_at": timestamp(),
            "source_candidates_json": str(candidates_path),
            "source_config": str(config_path),
            "reanalysis": {
                "kind": "physical_risk_candidate_pool_reanalysis",
                "original_candidate_count": len(raw_candidates),
                "feasible_candidate_count": len(feasible_candidates),
                "infeasible_candidate_count": len(infeasible_records),
                "selection_metric": selection_metric,
                "physical_hard_constraints_applied": True,
                "note": (
                    "This artifact re-ranks an existing candidate pool with current "
                    "physical constraints; it is not a fresh training/search run."
                ),
            },
            "lut_stats": estimator.get_lut_stats(),
        }
    )

    write_pareto_selection_artifacts(output_dir, summary)
    write_json(output_dir / "infeasible_candidates.json", {"records": infeasible_records})
    write_json(output_dir / "lut_stats.json", estimator.get_lut_stats())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
