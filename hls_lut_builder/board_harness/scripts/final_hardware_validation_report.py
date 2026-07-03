#!/usr/bin/env python3
"""Write a strict final hardware validation report from route and COM5 evidence."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

sys.path.insert(0, str(REPO_ROOT / "src"))

from hwnas_fpga.hardware.validation_report import build_final_hardware_validation_report  # noqa: E402


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final HW-NAS board validation report")
    parser.add_argument("--candidate-path", required=True, help="best_candidate.json or candidate snapshot")
    parser.add_argument("--e2e-summary", help="rl_best_e2e_summary.json with build and measurement paths")
    parser.add_argument("--route-build-result", help="Single build_result.json / route payload")
    parser.add_argument("--measurement-json", help="COM5 measurement JSON")
    parser.add_argument("--run-name", help="Run name to select from --e2e-summary")
    parser.add_argument("--expected-serial-port", default="COM5")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def select_route_payload(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.route_build_result:
        path = Path(args.route_build_result).expanduser().resolve()
        return read_json(path), str(path)

    if not args.e2e_summary:
        raise ValueError("Either --route-build-result or --e2e-summary is required")

    summary_path = Path(args.e2e_summary).expanduser().resolve()
    summary = read_json(summary_path)
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"Missing runs[] in e2e summary: {summary_path}")

    candidates = [row for row in runs if isinstance(row, dict)]
    if args.run_name:
        candidates = [row for row in candidates if row.get("run_name") == args.run_name]
        if not candidates:
            raise ValueError(f"Run name not found in e2e summary: {args.run_name}")
    else:
        usable = [row for row in candidates if row.get("strict_latency_usable") is True]
        candidates = usable or candidates

    selected = candidates[0] if candidates else {}
    return dict(selected), str(summary_path)


def main() -> int:
    args = parse_args()
    candidate_path = Path(args.candidate_path).expanduser().resolve()
    route_payload, route_source = select_route_payload(args)
    measurement_path = (
        Path(args.measurement_json).expanduser().resolve()
        if args.measurement_json
        else Path(str(route_payload.get("measurement_json_path") or ""))
    )
    measurement_payload = read_json(measurement_path) if str(measurement_path) else {}
    report = build_final_hardware_validation_report(
        candidate_payload=read_json(candidate_path),
        route_payload=route_payload,
        measurement_payload=measurement_payload,
        measurement_json_path=str(measurement_path) if measurement_payload else "",
        expected_serial_port=args.expected_serial_port,
    )
    report.update(
        {
            "generated_at": timestamp(),
            "candidate_source": str(candidate_path),
            "route_source": route_source,
        }
    )
    output = Path(args.output).expanduser().resolve()
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
