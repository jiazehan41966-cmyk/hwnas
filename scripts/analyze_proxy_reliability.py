#!/usr/bin/env python3
"""Run the preregistered Gate 0 classification and hardware analyses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.analysis.proxy_reliability import (  # noqa: E402
    analyze_hardware_reliability,
    analyze_proxy_reliability,
    load_classification_observations,
    load_hardware_observations,
    write_proxy_reliability_bundle,
)


DEFAULT_PROTOCOL = REPO_ROOT / "configs" / "audit" / "proxy_reliability_gate0.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-csv", required=True)
    parser.add_argument("--hardware-csv", default=None)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=None)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    classification_path = Path(args.classification_csv).expanduser().resolve()
    hardware_path = (
        Path(args.hardware_csv).expanduser().resolve()
        if args.hardware_csv
        else None
    )
    bootstrap = protocol.get("bootstrap", {})
    iterations = int(
        args.bootstrap_iterations
        if args.bootstrap_iterations is not None
        else bootstrap.get("iterations", 2000)
    )
    seed = int(bootstrap.get("seed", 20260704))
    execution = protocol["execution"]
    classification = analyze_proxy_reliability(
        load_classification_observations(classification_path),
        truth_budget=int(execution["truth_budget"]),
        top_k=(5, 10),
        bootstrap_iterations=iterations,
        bootstrap_seed=seed,
        gate_config=protocol.get("classification_gate", {}),
    )
    hardware = None
    if hardware_path is not None:
        hardware_cfg = protocol.get("hardware_audit", {})
        hardware = analyze_hardware_reliability(
            load_hardware_observations(hardware_path),
            metrics=hardware_cfg.get(
                "metrics", ("latency_ms", "dsp", "bram", "lut")
            ),
            pareto_metrics=hardware_cfg.get(
                "pareto_metrics", ("latency_ms", "dsp", "bram", "lut")
            ),
            bootstrap_iterations=iterations,
            bootstrap_seed=seed,
        )
    outputs = write_proxy_reliability_bundle(
        args.output_dir,
        classification=classification,
        hardware=hardware,
        classification_source=str(classification_path),
        hardware_source=str(hardware_path) if hardware_path else None,
    )
    result = {
        "gate_status": classification["gate"]["status"],
        "protocol_complete": classification["gate"]["protocol_complete"],
        "earliest_usable_proxy": classification["gate"]["earliest_usable_proxy"],
        **outputs,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_pass and classification["gate"]["status"] != "pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
