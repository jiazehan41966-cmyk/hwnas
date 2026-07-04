#!/usr/bin/env python3
"""Compare the three frozen-protocol accuracy baselines."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.training.protocol_reporting import hierarchical_paired_bootstrap


def load_summary(run_dir: str | Path) -> dict:
    path = Path(run_dir) / "protocol_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimability = payload.get("claimability", {})
    if claimability.get("claimable") is not True:
        raise ValueError(f"{path} is not a complete claimable protocol run")
    return payload


def metric_rows(summary: dict, metric: str) -> list[dict]:
    return [
        {
            "fold": int(run["fold"]),
            "seed": int(run["seed"]),
            metric: float(run["outer_val"][metric]),
        }
        for run in summary.get("runs", [])
    ]


def compare(left: dict, right: dict, *, left_name: str, right_name: str) -> dict:
    metrics = {}
    for metric in ("macro_f1", "top1", "weighted_f1"):
        metrics[metric] = hierarchical_paired_bootstrap(
            metric_rows(left, metric),
            metric_rows(right, metric),
            metric=metric,
        )
    return {
        "left": left_name,
        "right": right_name,
        "direction": "left_minus_right",
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-run", required=True)
    parser.add_argument("--pretrained-run", required=True)
    parser.add_argument("--nas-run", required=True)
    parser.add_argument(
        "--output-dir",
        default="artifacts/protocol_baseline_analysis",
    )
    args = parser.parse_args()

    scratch = load_summary(args.scratch_run)
    pretrained = load_summary(args.pretrained_run)
    nas = load_summary(args.nas_run)
    pretrained_meta = pretrained.get("model") or {}
    if pretrained_meta.get("pretrained_loaded") is not True:
        raise ValueError("pretrained baseline did not record pretrained_loaded=true")

    comparisons = [
        compare(
            pretrained,
            scratch,
            left_name="mobilenet_v2_pretrained",
            right_name="mobilenet_v2_scratch",
        ),
        compare(
            pretrained,
            nas,
            left_name="mobilenet_v2_pretrained",
            right_name="rl_arch_135_legacy_selected",
        ),
    ]
    primary = comparisons[1]["metrics"]["macro_f1"]
    deployability_first = (
        primary["mean_difference"] >= 0.03 and primary["ci95_low"] > 0.0
    )
    payload = {
        "schema_version": 1,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "runs": {
            "scratch": str(Path(args.scratch_run).resolve()),
            "pretrained": str(Path(args.pretrained_run).resolve()),
            "nas": str(Path(args.nas_run).resolve()),
        },
        "comparisons": comparisons,
        "decision_rule": {
            "metric": "macro_f1",
            "practical_margin": 0.03,
            "requires_ci95_low_above_zero": True,
            "deployability_first_narrative": deployability_first,
        },
        "claim_boundary": (
            "rl_arch_135 was selected by the historical fold-0 workflow. "
            "The comparison benchmarks a frozen legacy-selected architecture; "
            "it does not establish unbiased NAS-method generalization."
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "baseline_comparison.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Frozen-protocol baseline comparison",
        "",
        f"- deployability-first narrative: `{deployability_first}`",
        f"- claim boundary: {payload['claim_boundary']}",
        "",
        "| comparison | metric | mean delta | 95% CI | paired n |",
        "|---|---|---:|---:|---:|",
    ]
    for comparison in comparisons:
        label = f"{comparison['left']} - {comparison['right']}"
        for metric, stats in comparison["metrics"].items():
            lines.append(
                f"| {label} | {metric} | {stats['mean_difference']:.4f} "
                f"| [{stats['ci95_low']:.4f}, {stats['ci95_high']:.4f}] "
                f"| {stats['paired_n']} |"
            )
    (output_dir / "baseline_comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(json_path), "decision": deployability_first}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
