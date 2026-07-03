#!/usr/bin/env python3
"""Generate a machine-checkable audit for the HW-NAS FPGA route-clean gap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hwnas_fpga.hardware.acceptance_audit import (  # noqa: E402
    GapAcceptanceAuditInputs,
    build_gap_acceptance_audit,
    write_json,
)


DEFAULT_PHYSICAL_CONFIG = (
    REPO_ROOT
    / "configs"
    / "search"
    / "nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_av7k325.yaml"
)
DEFAULT_LEGACY_PARETO = (
    REPO_ROOT
    / "results_port57244"
    / "nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_port57244_v1"
    / "results"
    / "pareto_selection.json"
)
DEFAULT_LUT_STATS = (
    REPO_ROOT
    / "results_port57244"
    / "nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_port57244_v1"
    / "results"
    / "lut_stats.json"
)
DEFAULT_PARETO_PROOF = (
    REPO_ROOT
    / "results"
    / "phase1_pareto_physical_risk_minirepro"
    / "pareto_selection.json"
)
DEFAULT_PARETO_REANALYSIS = (
    REPO_ROOT
    / "results"
    / "phase1_pareto_physical_risk_reanalysis_port57244"
)
DEFAULT_PARETO_RANKED_CSV = (
    DEFAULT_PARETO_REANALYSIS
    / "pareto_ranked_candidates.csv"
)
DEFAULT_FULL_PHASE0_RESULTS = (
    REPO_ROOT
    / "results"
    / "remote_phase0_full_rl300_search"
    / "results"
)
DEFAULT_FULL_PHASE0_STATUS = (
    REPO_ROOT
    / "results"
    / "remote_phase0_full_rl300_status"
    / "remote_status_latest.json"
)
DEFAULT_FULL_PHASE0_FETCH_ERROR = (
    REPO_ROOT
    / "results"
    / "remote_phase0_full_rl300_status"
    / "fetch_error_latest.json"
)
DEFAULT_LEGACY_ROUTE_GATE_ROOT = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "pareto_route_gate_with_default_quick_lowdsp_full"
)
DEFAULT_FULL_PHASE0_ROUTE_GATE_ROOT = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "pareto_route_gate_phase0_full_rl300_pending"
)
DEFAULT_ROUTE_GATE_ROOT = DEFAULT_FULL_PHASE0_ROUTE_GATE_ROOT
DEFAULT_LEGACY_FINAL_VALIDATION = (
    DEFAULT_LEGACY_ROUTE_GATE_ROOT
    / "reports"
    / "001_rl_arch_299.final_hardware_validation_report.json"
)
DEFAULT_LEGACY_MEASUREMENT = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "sonar_classifier_rl_best_e2e_board"
    / "port57244_rl_arch_299_lowdsp_p8_v1"
    / "measurements"
    / "sonar_classifier_rl_best_e2e__port57244_rl_arch_299_lowdsp_p8_v1.measurement.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "hwnas_gap_acceptance_audit"
    / "acceptance_audit.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW-NAS FPGA gap acceptance audit")
    parser.add_argument("--physical-config", default=str(DEFAULT_PHYSICAL_CONFIG))
    parser.add_argument("--legacy-candidate-source", default=str(DEFAULT_LEGACY_PARETO))
    parser.add_argument("--strict-lut-stats", default=str(DEFAULT_LUT_STATS))
    parser.add_argument("--pareto-selection", default=str(DEFAULT_PARETO_REANALYSIS / "pareto_selection.json"))
    parser.add_argument("--pareto-proof-selection", default=str(DEFAULT_PARETO_PROOF))
    parser.add_argument("--pareto-ranked-csv", default=str(DEFAULT_PARETO_RANKED_CSV))
    parser.add_argument("--fresh-search-pareto-selection", default=str(DEFAULT_FULL_PHASE0_RESULTS / "pareto_selection.json"))
    parser.add_argument("--fresh-search-ranked-csv", default=str(DEFAULT_FULL_PHASE0_RESULTS / "pareto_ranked_candidates.csv"))
    parser.add_argument("--fresh-search-summary", default=str(DEFAULT_FULL_PHASE0_RESULTS / "summary.json"))
    parser.add_argument("--fresh-search-status", default=str(DEFAULT_FULL_PHASE0_STATUS))
    parser.add_argument("--fresh-fetch-error", default=str(DEFAULT_FULL_PHASE0_FETCH_ERROR))
    parser.add_argument("--fresh-route-gate-summary", default=str(DEFAULT_FULL_PHASE0_ROUTE_GATE_ROOT / "route_gate_summary.json"))
    parser.add_argument("--route-gate-summary", default=str(DEFAULT_ROUTE_GATE_ROOT / "route_gate_summary.json"))
    parser.add_argument("--final-candidate-manifest", default=str(DEFAULT_ROUTE_GATE_ROOT / "final_candidate_manifest.json"))
    parser.add_argument("--final-validation-report", default="")
    parser.add_argument("--measurement-json", default="")
    parser.add_argument("--expected-serial-port", default="COM5")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = GapAcceptanceAuditInputs(
        physical_config_path=args.physical_config,
        legacy_candidate_source_path=args.legacy_candidate_source,
        strict_lut_stats_path=args.strict_lut_stats,
        pareto_selection_path=args.pareto_selection,
        pareto_proof_selection_path=args.pareto_proof_selection,
        pareto_ranked_csv_path=args.pareto_ranked_csv,
        fresh_search_pareto_selection_path=args.fresh_search_pareto_selection,
        fresh_search_ranked_csv_path=args.fresh_search_ranked_csv,
        fresh_search_summary_path=args.fresh_search_summary,
        fresh_search_status_path=args.fresh_search_status,
        fresh_fetch_error_path=args.fresh_fetch_error,
        fresh_route_gate_summary_path=args.fresh_route_gate_summary,
        route_gate_summary_path=args.route_gate_summary,
        final_candidate_manifest_path=args.final_candidate_manifest,
        final_validation_report_path=args.final_validation_report,
        measurement_json_path=args.measurement_json,
        expected_serial_port=args.expected_serial_port,
    )
    audit = build_gap_acceptance_audit(inputs)
    write_json(args.output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
