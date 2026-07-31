#!/usr/bin/env python3
"""Run synthetic INT8 reference, HLS C-sim and exact-shape HLS synthesis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.dir_int8_reference import (  # noqa: E402
    DIR_INT8_CONTRACT,
    deterministic_weights,
    dir_mbconv3_split11_e3_v1_int8,
)
from hwnas_fpga.fourstage_selection import canonical_sha256  # noqa: E402
from hwnas_fpga.training.protocol_reporting import sha256_file  # noqa: E402


def save_vector(path: Path, values: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(values, dtype=np.int8).tofile(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_csynth(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()

    def text_at(xpath: str) -> str | None:
        node = root.find(xpath)
        return None if node is None else node.text

    return {
        "report_path": str(path.resolve()),
        "report_sha256": sha256_file(path),
        "part": text_at(".//UserAssignments/Part"),
        "target_clock_ns": text_at(".//UserAssignments/TargetClockPeriod"),
        "estimated_clock_ns": text_at(
            ".//PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod"
        ),
        "latency_cycles_best": text_at(
            ".//PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency"
        ),
        "latency_cycles_worst": text_at(
            ".//PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency"
        ),
        "resources": {
            name: text_at(f".//AreaEstimates/Resources/{name}")
            for name in ("BRAM_18K", "DSP", "FF", "LUT", "URAM")
        },
        "available": {
            name: text_at(f".//AreaEstimates/AvailableResources/{name}")
            for name in ("BRAM_18K", "DSP", "FF", "LUT", "URAM")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vitis-hls",
        default=r"F:\vivado\Vitis_HLS\2023.2\bin\vitis_hls.bat",
    )
    parser.add_argument("--part", default="xc7k325t-ffg676-2")
    parser.add_argument("--clock-ns", type=float, default=5.0)
    parser.add_argument(
        "--work-dir",
        default=(
            "results/sonar_fourstage_operator_v2/"
            "dir_int8_hls_pretrain_gate"
        ),
    )
    parser.add_argument(
        "--hls-project-dir",
        default=None,
        help=(
            "Optional short Windows path for the transient Vitis project. "
            "Use this only to avoid the Vitis-generated RTL path limit."
        ),
    )
    parser.add_argument(
        "--summary",
        default=(
            "artifacts/sonar_fourstage_operator_v2/"
            "dir_int8_hls_pretrain_gate.json"
        ),
    )
    args = parser.parse_args()
    work_dir = Path(args.work_dir).resolve()
    vector_dir = work_dir / "vectors"
    project_dir = (
        Path(args.hls_project_dir).resolve()
        if args.hls_project_dir
        else work_dir / "vitis_project"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    weights = deterministic_weights()
    rng = np.random.default_rng(20260731)
    cases = {
        "random": rng.integers(
            -128, 128, size=(32, 28, 28), dtype=np.int8
        ),
        "negative_full_scale": np.full((32, 28, 28), -128, dtype=np.int8),
        "positive_full_scale": np.full((32, 28, 28), 127, dtype=np.int8),
    }
    vector_evidence = {}
    for name, inputs in cases.items():
        expected = dir_mbconv3_split11_e3_v1_int8(inputs, weights)
        case_dir = vector_dir / name
        vector_evidence[name] = {
            "input": save_vector(case_dir / "input.bin", inputs),
            "expand_w": save_vector(case_dir / "expand_w.bin", weights.expand),
            "dw_1x3_w": save_vector(
                case_dir / "dw_1x3_w.bin", weights.dw_1x3
            ),
            "dw_3x1_w": save_vector(
                case_dir / "dw_3x1_w.bin", weights.dw_3x1
            ),
            "project_w": save_vector(
                case_dir / "project_w.bin", weights.project
            ),
            "expected": save_vector(case_dir / "expected.bin", expected),
        }

    vitis = Path(args.vitis_hls).resolve()
    if not vitis.is_file():
        raise FileNotFoundError(vitis)
    tcl = ROOT / "hls_dir_mbconv3_split11_e3_v1" / "run_hls.tcl"
    command_parts = [
        str(vitis),
        "-f",
        str(tcl),
    ]
    command_string = subprocess.list2cmdline(command_parts)
    environment = os.environ.copy()
    environment.update(
        {
            "DIR_V1_PROJECT_DIR": str(project_dir),
            "DIR_V1_VECTOR_DIR": str(vector_dir),
            "DIR_V1_PART": str(args.part),
            "DIR_V1_CLOCK_NS": str(args.clock_ns),
        }
    )
    completed = subprocess.run(
        ["cmd.exe", "/d", "/s", "/c", command_string],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
        env=environment,
    )
    stdout_path = work_dir / "vitis_hls.stdout.log"
    stderr_path = work_dir / "vitis_hls.stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    mismatch_rows = {}
    for name in cases:
        marker = f"{name} mismatches="
        count = None
        for line in completed.stdout.splitlines():
            if marker in line:
                count = int(line.rsplit("=", 1)[1].strip())
        mismatch_rows[name] = count
    reports = list(
        project_dir.glob("solution1/syn/report/*_csynth.xml")
    )
    synthesis = parse_csynth(reports[0]) if reports else None
    synthesis_pass = bool(completed.returncode == 0 and synthesis is not None)
    passed = bool(
        completed.returncode == 0
        and all(value == 0 for value in mismatch_rows.values())
        and synthesis is not None
    )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "operator": "dir_mbconv3_split11_e3_v1",
        "shape": {
            "input_resolution": 28,
            "in_channels": 32,
            "expanded_channels": 96,
            "out_channels": 32,
            "stride": 1,
        },
        "integer_contract": DIR_INT8_CONTRACT,
        "random_and_boundary_vectors": vector_evidence,
        "hls_csim": {
            "status": (
                "PASS"
                if all(value == 0 for value in mismatch_rows.values())
                else "FAIL"
            ),
            "mismatches": mismatch_rows,
        },
        "hls_synthesis": {
            "status": "PASS" if synthesis_pass else "FAIL",
            "report": synthesis,
            "transient_project_path": str(project_dir),
            "windows_path_limit_workaround": bool(args.hls_project_dir),
        },
        "tool": {
            "path": str(vitis),
            "returncode": completed.returncode,
            "command": command_parts,
            "stdout": {
                "path": str(stdout_path),
                "sha256": sha256_file(stdout_path),
            },
            "stderr": {
                "path": str(stderr_path),
                "sha256": sha256_file(stderr_path),
            },
        },
        "claim_boundary": (
            "Synthetic random/boundary integer parity and operator-level HLS "
            "synthesis only. This is not checkpoint-calibrated parity, RTL "
            "co-sim, micro-route, full-network HLS/route, board, or power."
        ),
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    target = Path(args.summary).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
