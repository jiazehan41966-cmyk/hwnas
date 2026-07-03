#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import build_cases, load_config, resolve_workspace_path, select_pilot_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the end-to-end HLS + Vivado downstream pilot pipeline")
    parser.add_argument("--config", required=True, help="Path to builder config YAML")
    parser.add_argument("--project-root", default=None, help="Optional override for generated project root")
    parser.add_argument("--operator", action="append", default=None, help="Limit the pilot to specific operators")
    parser.add_argument("--case", action="append", default=None, help="Limit the pilot to specific cases")
    parser.add_argument("--vitis-hls", default=None, help="Override Vitis HLS executable path")
    parser.add_argument("--vivado", default=None, help="Override Vivado executable path")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--force", action="store_true", help="Re-generate and re-run pilot cases")
    parser.add_argument("--hls-timeout-minutes", type=int, default=30, help="Per-case HLS timeout")
    parser.add_argument("--downstream-timeout-minutes", type=int, default=60, help="Per-case Vivado timeout")
    parser.add_argument("--summary-json", default=None, help="Optional pipeline summary JSON override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pilot_cases = select_pilot_cases(
        build_cases(
            config,
            operator_filter=args.operator,
            case_filter=args.case,
            project_root_override=args.project_root,
            sort_cases=False,
        )
    )
    if not pilot_cases:
        raise SystemExit("No pilot cases matched the provided filters.")

    case_names = [case.case_name for case in pilot_cases]
    python_exe = sys.executable
    summary_path = resolve_workspace_path(
        config,
        args.summary_json or "../results/pilot_pipeline_summary.json",
    )

    hls_csv = resolve_workspace_path(config, "../results/pilot_hls.csv")
    hls_manifest = resolve_workspace_path(config, "../results/pilot_hls_manifest.yaml")
    hls_archive = resolve_workspace_path(config, "../results/pilot_r")
    hls_synth_summary = resolve_workspace_path(config, "../results/pilot_hls_synth_summary.json")
    hls_parse_summary = resolve_workspace_path(config, "../results/pilot_hls_parse_summary.json")
    downstream_summary = resolve_workspace_path(config, "../results/pilot_vivado_downstream_summary.json")
    downstream_csv = resolve_workspace_path(config, "../results/pilot_vivado_downstream.csv")
    downstream_manifest = resolve_workspace_path(config, "../results/pilot_vivado_downstream_manifest.yaml")
    downstream_parse_summary = resolve_workspace_path(config, "../results/pilot_vivado_downstream_parse_summary.json")

    commands = [
        [
            python_exe,
            str((SCRIPT_DIR / "gen_project.py").resolve()),
            "--config",
            str(Path(args.config).resolve()),
            "--pilot",
            *(["--project-root", args.project_root] if args.project_root else []),
            *(["--force"] if args.force else []),
        ],
        [
            python_exe,
            str((SCRIPT_DIR / "run_synthesis.py").resolve()),
            "--config",
            str(Path(args.config).resolve()),
            "--pilot",
            "--summary-json",
            str(hls_synth_summary),
            "--timeout-minutes",
            str(args.hls_timeout_minutes),
            *(["--project-root", args.project_root] if args.project_root else []),
            *(["--vitis-hls", args.vitis_hls] if args.vitis_hls else []),
            *(["--force"] if args.force else []),
        ],
        [
            python_exe,
            str((SCRIPT_DIR / "parse_reports.py").resolve()),
            "--config",
            str(Path(args.config).resolve()),
            "--pilot",
            "--csv",
            str(hls_csv),
            "--manifest",
            str(hls_manifest),
            "--archive-dir",
            str(hls_archive),
            "--summary-json",
            str(hls_parse_summary),
            *(["--project-root", args.project_root] if args.project_root else []),
        ],
        [
            python_exe,
            str((SCRIPT_DIR / "run_vivado_downstream.py").resolve()),
            "--config",
            str(Path(args.config).resolve()),
            "--pilot",
            "--summary-json",
            str(downstream_summary),
            "--timeout-minutes",
            str(args.downstream_timeout_minutes),
            *(["--project-root", args.project_root] if args.project_root else []),
            *(["--vivado", args.vivado] if args.vivado else []),
            *(["--force"] if args.force else []),
        ],
        [
            python_exe,
            str((SCRIPT_DIR / "parse_vivado_downstream.py").resolve()),
            "--config",
            str(Path(args.config).resolve()),
            "--pilot",
            "--csv",
            str(downstream_csv),
            "--manifest",
            str(downstream_manifest),
            "--summary-json",
            str(downstream_parse_summary),
            *(["--project-root", args.project_root] if args.project_root else []),
        ],
    ]

    summary: dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pilot_cases": case_names,
        "commands": [" ".join(command) for command in commands],
        "status": "dry_run" if args.dry_run else "running",
    }

    if args.dry_run:
        for command in commands:
            print("[dry-run]", " ".join(command))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote pilot pipeline summary: {summary_path}")
        return

    command_results: list[dict[str, object]] = []
    for command in commands:
        print("[run]", " ".join(command))
        result = subprocess.run(command, cwd=Path.cwd(), text=True, capture_output=True, check=False)
        command_results.append(
            {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            summary["status"] = "failed"
            summary["command_results"] = command_results
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            raise SystemExit(result.returncode)

    summary["status"] = "success"
    summary["command_results"] = command_results
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote pilot pipeline summary: {summary_path}")


if __name__ == "__main__":
    main()
