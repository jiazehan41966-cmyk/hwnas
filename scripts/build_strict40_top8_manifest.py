#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = REPO_ROOT / "results" / "strict40_expanded_rl_random_200_eval10_analysis.json"
DEFAULT_RL_CANDIDATES = (
    REPO_ROOT
    / "results"
    / "formal_lut_strict40_expanded_nksid_rl200_eval10_explore_v1"
    / "results"
    / "candidates.jsonl"
)
DEFAULT_RANDOM_CANDIDATES = (
    REPO_ROOT
    / "results"
    / "formal_lut_strict40_expanded_nksid_random_200_eval10_v1"
    / "results"
    / "candidates.jsonl"
)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "strict40_expanded_top8_manifest_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an exact Top-8 candidate manifest from the strict40 RL-vs-Random analysis."
    )
    parser.add_argument("--analysis", default=str(DEFAULT_ANALYSIS))
    parser.add_argument("--rl-candidates", default=str(DEFAULT_RL_CANDIDATES))
    parser.add_argument("--random-candidates", default=str(DEFAULT_RANDOM_CANDIDATES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _block_label(block: dict[str, Any]) -> str:
    return (
        f"{block.get('op')}_e{block.get('expand_ratio')}"
        f"_k{block.get('kernel_size')}_s{block.get('stride')}"
    )


def _arch_key(encoding: dict[str, Any]) -> str:
    return "|".join(
        f"s{stage_index}:{_block_label(stage['blocks'][0])}"
        for stage_index, stage in enumerate(encoding["stages"])
    )


def _metric(row: dict[str, Any], name: str) -> Any:
    return row.get("candidate", {}).get("metrics", {}).get(name)


def _candidate_brief(row: dict[str, Any], *, rank: int, source_key: dict[str, Any]) -> dict[str, Any]:
    candidate = row["candidate"]
    return {
        "rank": rank,
        "candidate_id": f"top8_{rank:02d}",
        "candidate_label": source_key["compact"],
        "candidate_key": source_key["key"],
        "source_method": row["_source_method"],
        "source_arch_id": candidate.get("arch_id"),
        "observed_macro_f1": _metric(row, "macro_f1"),
        "observed_top1": _metric(row, "top1"),
        "observed_latency_ms": _metric(row, "latency_ms"),
        "observed_lut": _metric(row, "lut"),
        "observed_dsp": _metric(row, "dsp"),
        "observed_bram": _metric(row, "bram"),
        "observed_power_w": _metric(row, "power_w"),
        "architecture": candidate["encoding"],
    }


def main() -> None:
    args = parse_args()
    analysis_path = Path(args.analysis).expanduser().resolve()
    rl_path = Path(args.rl_candidates).expanduser().resolve()
    random_path = Path(args.random_candidates).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    analysis = _load_json(analysis_path)
    source_top = analysis.get("union_top10_proxy_region", [])[: int(args.top_k)]
    if len(source_top) != int(args.top_k):
        raise ValueError(
            f"analysis has only {len(source_top)} union top records, expected {args.top_k}"
        )

    observed_rows: list[dict[str, Any]] = []
    for method, path in (("rl", rl_path), ("random", random_path)):
        for row in _load_jsonl(path):
            row["_source_method"] = method
            observed_rows.append(row)

    rows_by_key: dict[str, list[dict[str, Any]]] = {}
    for row in observed_rows:
        key = _arch_key(row["candidate"]["encoding"])
        rows_by_key.setdefault(key, []).append(row)

    candidates: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for rank, item in enumerate(source_top, start=1):
        key = item["key"]
        if key in seen_keys:
            raise ValueError(f"duplicate key in top region: {key}")
        seen_keys.add(key)
        matches = rows_by_key.get(key, [])
        if not matches:
            raise ValueError(f"top-region architecture not found in candidate logs: {key}")
        best_match = max(
            matches,
            key=lambda row: (
                float(_metric(row, "macro_f1") or float("-inf")),
                float(_metric(row, "top1") or float("-inf")),
            ),
        )
        candidates.append(_candidate_brief(best_match, rank=rank, source_key=item))

    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_analysis": str(analysis_path),
        "source_rl_candidates": str(rl_path),
        "source_random_candidates": str(random_path),
        "selection": "union_top8_proxy_region_observed_macro_f1",
        "candidate_count": len(candidates),
        "config": analysis["space"]["exhaustive_validation"]["config"],
        "seeds": [42, 43, 44],
        "eval_epochs": 10,
        "selection_metric": "macro_f1",
        "power_coverage_caveat": analysis["space"].get("power_coverage_caveat"),
        "operator_scope": analysis["space"].get("operator_scope"),
        "candidates": candidates,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(candidates)} candidates -> {output_path}")
    for candidate in candidates:
        print(
            f"{candidate['rank']}. {candidate['candidate_label']} "
            f"source={candidate['source_method']}:{candidate['source_arch_id']} "
            f"macro_f1={candidate['observed_macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
