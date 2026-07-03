#!/usr/bin/env python3
"""Export an existing Vitis HLS case into an IP catalog package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export existing HLS cases as Vivado IP")
    parser.add_argument(
        "--case-dir",
        action="append",
        required=True,
        help="Absolute or relative path to an existing HLS case directory. Repeat for multiple cases.",
    )
    parser.add_argument(
        "--vitis-hls",
        required=True,
        help="Path to vitis_hls executable or batch launcher.",
    )
    parser.add_argument(
        "--solution",
        default="solution1",
        help="HLS solution name to export. Default: solution1",
    )
    return parser.parse_args()


def write_export_tcl(case_dir: Path, solution: str) -> Path:
    tcl_path = case_dir / "export_ip.tcl"
    tcl_path.write_text(
        "\n".join(
            [
                "open_project project",
                f"open_solution {solution}",
                "export_design -rtl verilog -format ip_catalog",
                "exit",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return tcl_path


def build_command(vitis_hls: Path, tcl_path: Path) -> list[str]:
    if vitis_hls.suffix.lower() in {".bat", ".cmd"}:
        return ["cmd", "/c", str(vitis_hls), "-f", str(tcl_path)]
    return [str(vitis_hls), "-f", str(tcl_path)]


def export_case(case_dir: Path, vitis_hls: Path, solution: str) -> dict[str, object]:
    script_path = write_export_tcl(case_dir, solution)
    log_path = case_dir / "export_ip.log"
    export_zip = case_dir / "project" / solution / "impl" / "export.zip"
    component_xml = case_dir / "project" / solution / "impl" / "ip" / "component.xml"
    command = build_command(vitis_hls, script_path)
    completed = subprocess.run(
        command,
        cwd=case_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60 * 60,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    status = {
        "case_dir": str(case_dir),
        "script": str(script_path),
        "log": str(log_path),
        "return_code": completed.returncode,
        "export_zip": str(export_zip) if export_zip.exists() else None,
        "component_xml": str(component_xml) if component_xml.exists() else None,
        "success": completed.returncode == 0 and export_zip.exists() and component_xml.exists(),
    }
    (case_dir / "export_ip_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return status


def main() -> None:
    args = parse_args()
    vitis_hls = Path(args.vitis_hls).expanduser().resolve()
    if not vitis_hls.exists():
        raise FileNotFoundError(f"vitis_hls not found: {vitis_hls}")

    statuses = []
    for raw_case_dir in args.case_dir:
        case_dir = Path(raw_case_dir).expanduser().resolve()
        if not case_dir.exists():
            raise FileNotFoundError(f"case directory not found: {case_dir}")
        statuses.append(export_case(case_dir, vitis_hls, args.solution))

    print(json.dumps(statuses, ensure_ascii=False, indent=2))
    if not all(status["success"] for status in statuses):
        sys.exit(1)


if __name__ == "__main__":
    main()
