#!/usr/bin/env python3
"""Evaluate base8 with the existing strict board LUT without route overclaiming."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.fourstage_operator import enumerate_base8  # noqa: E402
from hwnas_fpga.fourstage_selection import canonical_sha256  # noqa: E402
from hwnas_fpga.runtime import (  # noqa: E402
    build_constraints,
    build_cost_estimator,
    build_hardware_spec,
    build_search_space,
    load_config,
)
from hwnas_fpga.training.protocol_reporting import sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "configs/search/"
            "nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_"
            "cuda_av7k325.yaml"
        ),
    )
    parser.add_argument("--lut-path", required=True)
    parser.add_argument("--formal-lut-status-path", required=True)
    parser.add_argument(
        "--output",
        default=(
            "artifacts/sonar_fourstage_operator_v2/"
            "base8_strict_lut_proxy_audit.json"
        ),
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))
    config["hardware"]["lut_path"] = str(Path(args.lut_path).resolve())
    config["hardware"]["formal_lut_status_path"] = str(
        Path(args.formal_lut_status_path).resolve()
    )
    constraints = build_constraints(config)
    hardware_spec = build_hardware_spec(config)
    search_space = build_search_space(
        config,
        image_size=int(config["dataset"]["image_size"]),
        input_channels=int(config["dataset"]["input_channels"]),
        num_classes=int(config["dataset"]["num_classes"]),
        constraints=constraints,
    )
    estimator = build_cost_estimator(
        config,
        hardware_spec=hardware_spec,
        constraints=constraints,
    )

    rows = []
    all_covered = True
    for row in enumerate_base8():
        estimator.reset_lut_stats()
        cost = estimator.estimate(row.architecture, search_space)
        stats = estimator.get_lut_stats()
        strict_violations = [
            value
            for value in cost.violations
            if value.startswith("strict board LUT")
        ]
        covered = (
            int(stats["true_misses"]) == 0
            and int(stats["deferred_hits"]) == 0
            and not strict_violations
        )
        all_covered = all_covered and covered
        cost_payload = asdict(cost)
        cost_payload["power_w_field_semantics"] = (
            "analytical_power_proxy_w; not Vivado report_power and not "
            "external-meter board measurement"
        )
        rows.append(
            {
                "arch_id": row.arch_id,
                "factors": row.factors(),
                "strict_lut_covered": covered,
                "strict_lut_stats": stats,
                "strict_lut_violations": strict_violations,
                "cost": cost_payload,
            }
        )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all_covered else "FAIL",
        "evidence_layer": "strict_lut_proxy",
        "full_network_route_completed": False,
        "operator_micro_harness_route_completed": False,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "lut": {
            "path": str(Path(args.lut_path).resolve()),
            "sha256": sha256_file(args.lut_path),
        },
        "formal_lut_status": {
            "path": str(Path(args.formal_lut_status_path).resolve()),
            "sha256": sha256_file(args.formal_lut_status_path),
        },
        "candidate_count": len(rows),
        "all_candidates_strict_lut_covered": all_covered,
        "rows": rows,
        "claim_boundary": (
            "Strict LUT values mix retained measured operator entries into a "
            "network cost estimate. They do not establish current full-network "
            "HLS, place-and-route, COM5, bitstream, or measured power evidence."
        ),
        "power_status": "NOT_MEASURED",
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if all_covered else 2


if __name__ == "__main__":
    raise SystemExit(main())
