#!/usr/bin/env python3
"""Prepare Phase0 v4 sonar Stage3 k3 launch-readiness artifacts.

This helper is local-only. It does not launch server RL300, does not run
Vivado full-network route, and does not access COM ports. It verifies the
already generated admission evidence and writes a command bundle for the next
manual decision point.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = REPO_ROOT / "hls_lut_builder" / "results" / "phase0_v4_sonar_stage3_k3_pilot"
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "search"
    / "nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_v4_sonar_stage3_k3_lowdsp_draft_av7k325.yaml"
)
DEFAULT_OUTPUT_DIR = PILOT_ROOT / "launch_readiness"
DEFAULT_TIMING_GATE = PILOT_ROOT / "sonar_stage3_k3_timing_gate.json"
DEFAULT_MERGED_LUT = PILOT_ROOT / "merged_current84_arch84_plus_sonar_stage3_k3_lut.json"
DEFAULT_MERGED_STATUS = PILOT_ROOT / "merged_current84_arch84_plus_sonar_stage3_k3_status.json"
DEFAULT_PRECHECK_DIR = PILOT_ROOT / "precheck_merged_with_lowdsp"
DEFAULT_DIVERSITY = PILOT_ROOT / "diversity_audit_v4_sonar_stage3_k3_lowdsp_draft.json"
DEFAULT_REMOTE_OUTPUT_DIR = REPO_ROOT / "results" / "remote_phase0_v4_sonar_stage3_k3_rl300_search" / "results"
DEFAULT_ROUTE_OUTPUT_ROOT = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "pareto_route_gate_phase0_v4_sonar_stage3_k3_pending"
)
V3_LOWDSP_CATALOG = (
    REPO_ROOT / "hls_lut_builder" / "board_harness" / "configs" / "phase0_v3_lowdsp_override_catalog.yaml"
)
V4_SONAR_LOWDSP_CATALOG = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "configs"
    / "phase0_v4_sonar_stage3_lowdsp_override_catalog.yaml"
)


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


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def run_command(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def catalog_args(paths: list[Path]) -> list[str]:
    args: list[str] = []
    for path in paths:
        args.extend(["--lowdsp-override-catalog", rel(path)])
    return args


def precheck_catalog_args(paths: list[Path]) -> list[str]:
    args: list[str] = []
    for path in paths:
        args.extend(["--lowdsp-catalog", rel(path)])
    return args


def precheck_command(args: argparse.Namespace, lowdsp_catalogs: list[Path]) -> list[str]:
    return [
        args.python,
        "scripts/audit_phase0_v4_sonar_op_precheck.py",
        "--config",
        rel(Path(args.config)),
        "--lut",
        rel(Path(args.merged_lut)),
        "--status",
        rel(Path(args.merged_status)),
        *precheck_catalog_args(lowdsp_catalogs[1:]),
        "--operators",
        "denoise,edge",
        "--kernels",
        "3",
        "--stages",
        "3",
        "--output-dir",
        rel(Path(args.precheck_dir)),
    ]


def diversity_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        "scripts/audit_phase0_search_diversity.py",
        "--config",
        rel(Path(args.config)),
        "--sample-count",
        str(args.sample_count),
        "--min-unique",
        str(args.min_unique),
        "--output",
        rel(Path(args.diversity_output)),
    ]


def run_local_audits(args: argparse.Namespace, lowdsp_catalogs: list[Path]) -> dict[str, Any]:
    return {
        "precheck": run_command(precheck_command(args, lowdsp_catalogs)),
        "diversity": run_command(diversity_command(args)),
    }


def summarize_config(config_path: Path) -> dict[str, Any]:
    config = read_yaml(config_path)
    project = config.get("project") if isinstance(config.get("project"), dict) else {}
    search_space = config.get("search_space") if isinstance(config.get("search_space"), dict) else {}
    notes = config.get("notes") if isinstance(config.get("notes"), dict) else {}
    stage_block_choices = search_space.get("stage_block_choices")
    stage3_choices = []
    if isinstance(stage_block_choices, list) and len(stage_block_choices) > 3:
        stage3_choices = list(stage_block_choices[3] or [])
    return {
        "run_name": project.get("run_name"),
        "phase0_v4_status": notes.get("phase0_v4_status"),
        "server_launch_status": notes.get("server_launch_status"),
        "stage3_choices": stage3_choices,
        "physical_risk_weight": (
            (config.get("search") or {}).get("objective_weights") or {}
        ).get("physical_risk"),
        "early_expand_pressure_weight": (
            (config.get("search") or {}).get("objective_weights") or {}
        ).get("early_expand_pressure"),
    }


def count_entries(path: Path) -> int:
    payload = read_json(path)
    entries = payload.get("entries")
    return len(entries) if isinstance(entries, list) else 0


def evaluate_readiness(
    *,
    args: argparse.Namespace,
    config_summary: Mapping[str, Any],
    timing_gate: Mapping[str, Any],
    precheck: Mapping[str, Any],
    diversity: Mapping[str, Any],
) -> tuple[str, list[str]]:
    blockers: list[str] = []

    required_paths = {
        "config": Path(args.config),
        "timing_gate": Path(args.timing_gate),
        "merged_lut": Path(args.merged_lut),
        "merged_status": Path(args.merged_status),
        "precheck": Path(args.precheck_dir) / "sonar_op_coverage.json",
        "diversity": Path(args.diversity_output),
        "v3_lowdsp_catalog": V3_LOWDSP_CATALOG,
        "v4_sonar_lowdsp_catalog": V4_SONAR_LOWDSP_CATALOG,
    }
    for label, path in required_paths.items():
        if not path.exists():
            blockers.append(f"missing required artifact: {label} -> {path}")

    if timing_gate.get("status") != "PASS":
        blockers.append("timing gate is not PASS")
    timing_summary = timing_gate.get("summary") if isinstance(timing_gate.get("summary"), dict) else {}
    if timing_summary.get("passed_cases") != 2 or timing_summary.get("blocked_cases") != 0:
        blockers.append("timing gate does not report 2/2 passed cases")

    precheck_summary = precheck.get("summary") if isinstance(precheck.get("summary"), dict) else {}
    if precheck.get("status") != "PASS":
        blockers.append("targeted sonar precheck is not PASS")
    if (
        precheck_summary.get("strict_lut_hits") != 2
        or precheck_summary.get("true_misses") != 0
        or precheck_summary.get("deferred_hits") != 0
        or precheck_summary.get("lowdsp_override_covered_rows") != 2
    ):
        blockers.append("targeted sonar precheck summary is not 2/2 clean")

    diversity_summary = diversity.get("diversity") if isinstance(diversity.get("diversity"), dict) else {}
    if diversity.get("status") != "PASS":
        blockers.append("diversity audit is not PASS")
    if int(diversity_summary.get("unique_encoding_count") or 0) < int(args.min_unique):
        blockers.append("unique_encoding_count is below the launch threshold")
    if int(diversity_summary.get("feasible_unique_encoding_count") or 0) < int(args.min_unique):
        blockers.append("feasible_unique_encoding_count is below the launch threshold")

    if config_summary.get("phase0_v4_status") != "inactive_draft_local_only":
        blockers.append("draft config is not marked inactive_draft_local_only")
    if config_summary.get("server_launch_status") != "not_launched":
        blockers.append("draft config server_launch_status is not not_launched")

    stage3 = config_summary.get("stage3_choices")
    sonar_choices = {
        (choice.get("op"), int(choice.get("kernel_size")), int(choice.get("expand_ratio")))
        for choice in stage3
        if isinstance(choice, dict) and choice.get("op") in {"denoise", "edge"}
    }
    if sonar_choices != {("denoise", 3, 1), ("edge", 3, 1)}:
        blockers.append("stage3 sonar choices are not exactly denoise k3 e1 and edge k3 e1")

    return ("READY_FOR_MANUAL_SERVER_LAUNCH" if not blockers else "BLOCKED", blockers)


def build_command_bundle(
    *,
    args: argparse.Namespace,
    config_summary: Mapping[str, Any],
    lowdsp_catalogs: list[Path],
) -> dict[str, Any]:
    run_name = str(config_summary.get("run_name") or Path(args.config).stem)
    remote_output_dir = Path(args.remote_output_dir)
    route_output_root = Path(args.route_output_root)
    pareto_selection = remote_output_dir / run_name / "results" / "pareto_selection.json"
    lowdsp_args = catalog_args(lowdsp_catalogs)
    route_common = [
        "--pareto-selection",
        rel(pareto_selection),
        "--output-root",
        rel(route_output_root),
        "--topk",
        "24",
        "--full-topk",
        "8",
        "--full-max-actual-dsp",
        "700",
        *lowdsp_args,
    ]
    return {
        "local_admission_precheck": precheck_command(args, lowdsp_catalogs),
        "local_diversity_dry_run": diversity_command(args),
        "manual_server_rl300_launch": [
            args.python,
            "run_search.py",
            "--config",
            rel(Path(args.config)),
            "--output-dir",
            rel(remote_output_dir),
            "--run-name",
            run_name,
            "--device",
            "cuda",
        ],
        "manual_server_rl300_resume": [
            args.python,
            "run_search.py",
            "--config",
            rel(Path(args.config)),
            "--output-dir",
            rel(remote_output_dir),
            "--run-name",
            run_name,
            "--device",
            "cuda",
            "--resume",
        ],
        "prepare_route_gate_after_search": [
            args.python,
            "scripts/prepare_phase0_full_route_gate.py",
            *route_common,
        ],
        "route_gate_quick_after_manual_approval": [
            args.python,
            "hls_lut_builder/board_harness/scripts/pareto_route_gate.py",
            "quick",
            *route_common,
            "--execute",
        ],
        "route_gate_full_after_quick_pass": [
            args.python,
            "hls_lut_builder/board_harness/scripts/pareto_route_gate.py",
            "full",
            *route_common,
            "--execute",
        ],
        "route_gate_refresh_after_route_or_measurement": [
            args.python,
            "hls_lut_builder/board_harness/scripts/pareto_route_gate.py",
            "refresh",
            *route_common,
        ],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Phase0 v4 Sonar Stage3 k3 Launch Readiness",
        "",
        f"Generated: `{payload['generated_at']}`",
        f"Status: `{payload['status']}`",
        "",
        "## Conclusion",
        "",
    ]
    if payload["status"] == "READY_FOR_MANUAL_SERVER_LAUNCH":
        lines.append(
            "The minimal Stage3 k3 sonar admission package is ready for a separate manual server RL300 launch decision."
        )
    else:
        lines.append("The package is blocked and must not be used to launch server RL300.")
    lines.extend(
        [
            "",
            "## Evidence Summary",
            "",
            f"- Timing gate: `{payload['evidence']['timing_gate']['status']}`",
            f"- Targeted precheck: `{payload['evidence']['precheck']['status']}`",
            f"- Diversity audit: `{payload['evidence']['diversity']['status']}`",
            f"- Merged LUT entries: `{payload['evidence']['merged_lut_entries']}`",
            f"- Merged status entries: `{payload['evidence']['merged_status_entries']}`",
            "",
            "## Blocking Items",
            "",
        ]
    )
    blockers = payload.get("blocking_items") or []
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- None for the admitted Stage3 k3 sonar rows.")
    lines.extend(["", "## Command Bundle", ""])
    for name, command in (payload.get("commands") or {}).items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("```powershell")
        lines.append(" ".join(str(part) for part in command))
        lines.append("```")
        lines.append("")
    lines.extend(
        [
            "## Guardrails",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload.get("guardrails") or [])
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Phase0 v4 sonar launch-readiness artifacts")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timing-gate", default=str(DEFAULT_TIMING_GATE))
    parser.add_argument("--merged-lut", default=str(DEFAULT_MERGED_LUT))
    parser.add_argument("--merged-status", default=str(DEFAULT_MERGED_STATUS))
    parser.add_argument("--precheck-dir", default=str(DEFAULT_PRECHECK_DIR))
    parser.add_argument("--diversity-output", default=str(DEFAULT_DIVERSITY))
    parser.add_argument("--remote-output-dir", default=str(DEFAULT_REMOTE_OUTPUT_DIR))
    parser.add_argument("--route-output-root", default=str(DEFAULT_ROUTE_OUTPUT_ROOT))
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--min-unique", type=int, default=3)
    parser.add_argument(
        "--rerun-local-audits",
        action="store_true",
        help="Rerun the local precheck and diversity dry-run before writing readiness artifacts.",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Return non-zero if the readiness status is BLOCKED.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    lowdsp_catalogs = [V3_LOWDSP_CATALOG, V4_SONAR_LOWDSP_CATALOG]

    local_audit_results: dict[str, Any] = {}
    if args.rerun_local_audits:
        local_audit_results = run_local_audits(args, lowdsp_catalogs)

    config_summary = summarize_config(Path(args.config))
    timing_gate = read_json(Path(args.timing_gate))
    precheck = read_json(Path(args.precheck_dir) / "sonar_op_coverage.json")
    diversity = read_json(Path(args.diversity_output))
    status, blockers = evaluate_readiness(
        args=args,
        config_summary=config_summary,
        timing_gate=timing_gate,
        precheck=precheck,
        diversity=diversity,
    )
    for audit_name, audit_result in local_audit_results.items():
        if int(audit_result.get("returncode", 1)) != 0:
            blockers.append(
                f"local audit rerun failed: {audit_name} returned {audit_result.get('returncode')}"
            )
    if blockers:
        status = "BLOCKED"
    commands = build_command_bundle(
        args=args,
        config_summary=config_summary,
        lowdsp_catalogs=lowdsp_catalogs,
    )

    timing_summary = timing_gate.get("summary") if isinstance(timing_gate.get("summary"), dict) else {}
    precheck_summary = precheck.get("summary") if isinstance(precheck.get("summary"), dict) else {}
    diversity_summary = diversity.get("diversity") if isinstance(diversity.get("diversity"), dict) else {}
    payload = {
        "generated_at": timestamp(),
        "status": status,
        "runs_server": False,
        "runs_training": False,
        "runs_vivado": False,
        "runs_com_measurement": False,
        "config": str(Path(args.config).resolve()),
        "config_summary": config_summary,
        "evidence": {
            "timing_gate": {
                "path": str(Path(args.timing_gate).resolve()),
                "status": timing_gate.get("status"),
                "summary": timing_summary,
            },
            "precheck": {
                "path": str((Path(args.precheck_dir) / "sonar_op_coverage.json").resolve()),
                "status": precheck.get("status"),
                "summary": precheck_summary,
            },
            "diversity": {
                "path": str(Path(args.diversity_output).resolve()),
                "status": diversity.get("status"),
                "summary": {
                    "sample_count": diversity_summary.get("sample_count"),
                    "unique_encoding_count": diversity_summary.get("unique_encoding_count"),
                    "feasible_unique_encoding_count": diversity_summary.get("feasible_unique_encoding_count"),
                    "min_unique_required": diversity_summary.get("min_unique_required"),
                },
            },
            "merged_lut": str(Path(args.merged_lut).resolve()),
            "merged_status": str(Path(args.merged_status).resolve()),
            "merged_lut_entries": count_entries(Path(args.merged_lut)),
            "merged_status_entries": count_entries(Path(args.merged_status)),
            "lowdsp_catalogs": [str(path.resolve()) for path in lowdsp_catalogs],
        },
        "blocking_items": blockers,
        "commands": commands,
        "local_audit_rerun_results": local_audit_results,
        "guardrails": [
            "This readiness package does not launch server RL300.",
            "The draft config remains inactive until a separate manual decision is made.",
            "Do not run COM5 until a later full-network route gate PASS exists with actual Vivado DSP <= 700.",
            "Use both the Phase0 v3 low-DSP catalog and the Phase0 v4 sonar Stage3 catalog for later route-gate planning.",
            "Stage1/stage2 sonar rows, k5 sonar rows, and mixconv remain excluded.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "phase0_v4_sonar_stage3_k3_launch_readiness.json"
    md_path = output_dir / "phase0_v4_sonar_stage3_k3_launch_readiness.md"
    write_json(json_path, payload)
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status == "READY_FOR_MANUAL_SERVER_LAUNCH" or not args.require_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
