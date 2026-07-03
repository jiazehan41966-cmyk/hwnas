#!/usr/bin/env python3
"""Audit Phase0 architecture diversity before remote training or route gate.

This script is intentionally local-only: it does not contact the remote server,
does not launch Vivado, and does not run COM measurements. It can either inspect
existing search artifacts or sample the configured RL controller without
training to catch obvious exploration collapse before an expensive RL300 run.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "search"
    / "nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_v2_av7k325.yaml"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "phase0_v2_diversity_audit"
    / "diversity_audit.json"
)

sys.path.insert(0, str(REPO_ROOT / "src"))

from hwnas_fpga.hardware.route_gate import encoding_signature  # noqa: E402
from hwnas_fpga.runtime import (  # noqa: E402
    build_constraints,
    build_cost_estimator,
    build_hardware_spec,
    build_search_space,
    load_config,
)
from hwnas_fpga.search.rl_searcher import RLSearcher  # noqa: E402


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_pareto_encodings(payload: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for key in ("selected_topk", "ranked_candidates"):
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            encoding = row.get("encoding")
            if isinstance(encoding, Mapping):
                yield str(row.get("arch_id") or ""), encoding


def iter_jsonl_encodings(path: Path) -> Iterable[tuple[str, Mapping[str, Any]]]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        candidate = record.get("candidate") if isinstance(record, Mapping) else None
        if not isinstance(candidate, Mapping):
            continue
        encoding = candidate.get("encoding")
        if isinstance(encoding, Mapping):
            yield str(candidate.get("arch_id") or ""), encoding


def summarize_encodings(
    encodings: Iterable[tuple[str, Mapping[str, Any]]],
    *,
    source_kind: str,
    min_unique: int,
) -> dict[str, Any]:
    rows = []
    seen: set[str] = set()
    for index, (arch_id, encoding) in enumerate(encodings, start=1):
        signature = encoding_signature(encoding)
        seen.add(signature)
        rows.append(
            {
                "index": index,
                "arch_id": arch_id,
                "encoding_signature": signature,
            }
        )
    unique_count = len(seen)
    return {
        "source_kind": source_kind,
        "sample_count": len(rows),
        "unique_encoding_count": unique_count,
        "min_unique_required": int(min_unique),
        "passed": unique_count >= int(min_unique),
        "rows": rows,
    }


def sample_config_diversity(
    config_path: Path,
    *,
    sample_count: int,
    min_unique: int,
    apply_pruning: bool = True,
) -> dict[str, Any]:
    config = load_config(str(config_path))
    dataset_cfg = config.get("dataset", {})
    search_cfg = config.get("search", {})
    constraints = build_constraints(config)
    hardware_spec = build_hardware_spec(config)
    search_space = build_search_space(
        config,
        image_size=int(dataset_cfg.get("image_size", 224)),
        input_channels=int(dataset_cfg.get("input_channels", 1)),
        num_classes=int(dataset_cfg.get("num_classes", 8)),
        constraints=constraints,
    )
    estimator = build_cost_estimator(
        config,
        hardware_spec=hardware_spec,
        constraints=constraints,
    )
    if apply_pruning:
        search_space = search_space.pre_prune(estimator)
    searcher = RLSearcher(
        search_space=search_space,
        cost_estimator=estimator,
        constraints=constraints,
        controller_hidden_dim=int(search_cfg.get("controller_hidden", 64)),
        controller_lr=float(search_cfg.get("controller_lr", 0.01)),
        train_epochs_per_arch=0,
        device="cpu",
        seed=int(config.get("project", {}).get("seed", 42)),
        reward_weights=dict(search_cfg.get("objective_weights") or {}),
        selection_metric=str(search_cfg.get("selection_metric", "macro_f1")),
        reward_cfg=dict(search_cfg.get("reward_cfg") or {}),
        controller_temperature=float(search_cfg.get("controller_temperature", 1.0)),
        entropy_coef=float(search_cfg.get("controller_entropy_coef", search_cfg.get("entropy_coef", 0.0))),
        exploration_epsilon_start=float(search_cfg.get("exploration_epsilon_start", 0.0)),
        exploration_epsilon_end=(
            None
            if search_cfg.get("exploration_epsilon_end") is None
            else float(search_cfg.get("exploration_epsilon_end"))
        ),
        exploration_epsilon_decay_episodes=int(search_cfg.get("exploration_epsilon_decay_episodes", 0)),
        exploration_bonus=float(search_cfg.get("exploration_bonus", 0.0)),
    )

    sampled = []
    for index in range(max(0, int(sample_count))):
        searcher.current_episode = index
        architecture = searcher.generate_architecture()
        estimate = estimator.estimate(architecture, search_space)
        encoding = architecture.to_dict()
        sampled.append((f"sample_{index:03d}", encoding, estimate))

    summary = summarize_encodings(
        ((arch_id, encoding) for arch_id, encoding, _estimate in sampled),
        source_kind="config_controller_sample",
        min_unique=min_unique,
    )
    summary["apply_pruning"] = bool(apply_pruning)
    estimate_by_signature = {
        encoding_signature(encoding): estimate
        for _arch_id, encoding, estimate in sampled
    }
    for row in summary["rows"]:
        estimate = estimate_by_signature.get(row["encoding_signature"])
        if estimate is None:
            continue
        row["metrics"] = {
            "latency_ms": estimate.latency_ms,
            "lut": estimate.resource_lut,
            "bram": estimate.resource_bram,
            "dsp": estimate.resource_dsp,
            "power_w": estimate.power_w,
            "violations": list(estimate.violations),
        }
    feasible_signatures = {
        encoding_signature(encoding)
        for _arch_id, encoding, estimate in sampled
        if not estimate.violations
    }
    feasible_sample_count = sum(1 for _arch_id, _encoding, estimate in sampled if not estimate.violations)
    summary["feasible_sample_count"] = feasible_sample_count
    summary["feasible_unique_encoding_count"] = len(feasible_signatures)
    summary["feasible_passed"] = len(feasible_signatures) >= int(min_unique)
    summary["passed"] = bool(summary["passed"] and summary["feasible_passed"])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Phase0 architecture diversity")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pareto-selection", default="")
    parser.add_argument("--candidates-jsonl", default="")
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--min-unique", type=int, default=3)
    parser.add_argument(
        "--no-apply-pruning",
        action="store_true",
        help="Sample the raw config without run_search.py hardware-driven pre-pruning.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    min_unique = int(args.min_unique)

    if args.pareto_selection:
        pareto_path = Path(args.pareto_selection).expanduser().resolve()
        source = summarize_encodings(
            iter_pareto_encodings(read_json(pareto_path)),
            source_kind="pareto_selection",
            min_unique=min_unique,
        )
        source["source_path"] = str(pareto_path)
    elif args.candidates_jsonl:
        candidates_path = Path(args.candidates_jsonl).expanduser().resolve()
        source = summarize_encodings(
            iter_jsonl_encodings(candidates_path),
            source_kind="candidates_jsonl",
            min_unique=min_unique,
        )
        source["source_path"] = str(candidates_path)
    else:
        source = sample_config_diversity(
            config_path,
            sample_count=int(args.sample_count),
            min_unique=min_unique,
            apply_pruning=not bool(args.no_apply_pruning),
        )
        source["source_path"] = str(config_path)

    payload = {
        "generated_at": timestamp(),
        "runs_server": False,
        "runs_vivado": False,
        "runs_com_measurement": False,
        "config": str(config_path),
        "status": "PASS" if source["passed"] else "FAIL",
        "diversity": source,
        "guardrails": [
            "Do not launch remote RL300 unless unique_encoding_count >= min_unique_required.",
            "This audit does not train candidates, run Vivado, or access COM5.",
        ],
    }
    if args.output:
        write_json(Path(args.output).expanduser().resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if source["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
