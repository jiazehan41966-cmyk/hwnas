#!/usr/bin/env python3
"""Freeze the exact-shape Stage4 MBConv-k5-e3 HLS/micro-route gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.fourstage_selection import canonical_sha256  # noqa: E402


EXPECTED_SHAPE = {
    "input_resolution": 28,
    "in_channels": 32,
    "out_channels": 32,
    "stride": 1,
    "kernel_size": 5,
    "expand_ratio": 3,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _xml_int(root: ET.Element, path: str) -> int:
    value = root.findtext(path)
    if value is None:
        raise ValueError(f"Missing XML value: {path}")
    return int(value)


def _xml_float(root: ET.Element, path: str) -> float:
    value = root.findtext(path)
    if value is None:
        raise ValueError(f"Missing XML value: {path}")
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        default=(
            "results/sonar_fourstage_operator_v2/"
            "stage4_k5_exact_shape_hardware_gate"
        ),
    )
    parser.add_argument(
        "--hls-config",
        default=(
            "hls_lut_builder/configs/"
            "candidate_kernels_sonar_stage4_k5_exact_shape_av7k325.yaml"
        ),
    )
    parser.add_argument(
        "--micro-route-config",
        default=(
            "hls_lut_builder/configs/candidate_kernels_strict40_expansion.yaml"
        ),
    )
    parser.add_argument(
        "--route-tcl",
        default="hls_lut_builder/scripts/run_vivado_downstream.tcl",
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/sonar_fourstage_operator_v2/"
            "stage4_k5_exact_shape_hardware_gate.json"
        ),
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    hls_config = Path(args.hls_config).resolve()
    micro_route_config = Path(args.micro_route_config).resolve()
    route_tcl = Path(args.route_tcl).resolve()
    output = Path(args.output).resolve()

    metadata_path = raw_dir / "metadata.json"
    hls_status_path = raw_dir / "synthesis_status.json"
    hls_report_path = raw_dir / "mbconv_e3_k5_kernel_csynth.xml"
    route_summary_path = raw_dir / "stage4_k5_exact_shape_route676_summary.json"
    board_package_attempt_path = (
        raw_dir / "stage4_k5_exact_shape_route_summary.json"
    )
    board_package_log_path = raw_dir / "vivado_ffg900_failed.log"

    metadata = _read_json(metadata_path)
    hls_status = _read_json(hls_status_path)
    route_rows = _read_json(route_summary_path)
    board_package_rows = _read_json(board_package_attempt_path)
    if not isinstance(metadata, dict) or not isinstance(hls_status, dict):
        raise TypeError("HLS metadata/status must be JSON objects")
    if not isinstance(route_rows, list) or len(route_rows) != 1:
        raise ValueError("Expected exactly one micro-route result")
    if not isinstance(board_package_rows, list) or len(board_package_rows) != 1:
        raise ValueError("Expected exactly one board-package route attempt")

    parameters = metadata.get("parameters", {})
    actual_shape = {
        "input_resolution": int(parameters["feature_h"]),
        "in_channels": int(parameters["in_channels"]),
        "out_channels": int(parameters["out_channels"]),
        "stride": int(parameters["stride"]),
        "kernel_size": int(parameters["kernel_size"]),
        "expand_ratio": int(parameters["expand_ratio"]),
    }
    if actual_shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Exact-shape gate mismatch: {actual_shape} != {EXPECTED_SHAPE}"
        )
    if float(metadata["clock_period_ns"]) != 5.0:
        raise ValueError("HLS target must be exactly 5 ns")
    if metadata["part"] != "xc7k325t-ffg900-2":
        raise ValueError("HLS must target the AV7K325 board package")

    report = ET.parse(hls_report_path).getroot()
    hls_resources = {
        "lut": _xml_int(report, ".//AreaEstimates/Resources/LUT"),
        "ff": _xml_int(report, ".//AreaEstimates/Resources/FF"),
        "bram_18k": _xml_int(
            report, ".//AreaEstimates/Resources/BRAM_18K"
        ),
        "dsp": _xml_int(report, ".//AreaEstimates/Resources/DSP"),
    }
    hls_metrics = {
        "estimated_clock_period_ns": _xml_float(
            report,
            ".//PerformanceEstimates/SummaryOfTimingAnalysis/"
            "EstimatedClockPeriod",
        ),
        "latency_cycles": _xml_int(
            report,
            ".//PerformanceEstimates/SummaryOfOverallLatency/"
            "Worst-caseLatency",
        ),
        **hls_resources,
    }

    route = route_rows[0]
    board_package_attempt = board_package_rows[0]
    post_route = route.get("metrics", {}).get("post_route", {})
    hls_pass = (
        hls_status.get("status") == "success"
        and hls_status.get("returncode") == 0
    )
    route_pass = (
        route.get("status") == "success"
        and route.get("returncode") == 0
        and float(post_route.get("setup_wns_ns", -999.0)) >= 0.0
        and int(post_route.get("dsp", 999999)) <= 700
    )
    board_package_log = board_package_log_path.read_text(
        encoding="utf-8", errors="replace"
    )
    board_package_failure_class = (
        "VIVADO_RUNTIME_TCL_READ_FAILURE"
        if "unimacro_vhdl.tcl" in board_package_log
        and "synth_design failed" in board_package_log
        else "UNCLASSIFIED_FAILURE"
    )

    evidence_paths = [
        hls_config,
        micro_route_config,
        route_tcl,
        *sorted(
            path
            for path in raw_dir.rglob("*")
            if path.is_file() and path.name != "summary_stdout.json"
        ),
    ]
    evidence = [
        {
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in evidence_paths
    ]
    payload = {
        "schema_version": 1,
        "gate_id": "stage4_mbconv_k5_e3_exact_shape_hardware_v1",
        "status": "PASS" if hls_pass and route_pass else "FAIL",
        "shape": EXPECTED_SHAPE,
        "hls_synthesis": "PASS" if hls_pass else "FAIL",
        "micro_harness_route": "PASS" if route_pass else "FAIL",
        "hls": {
            "target_part": metadata["part"],
            "target_clock_ns": metadata["clock_period_ns"],
            "top_function": metadata["top_function"],
            "metrics": hls_metrics,
            "report_path": str(hls_report_path),
        },
        "micro_harness": {
            "target_part": "xc7k325t-ffg676-2",
            "flow": "out_of_context_operator_micro_harness",
            "target_clock_ns": 5.0,
            "post_route": {
                "wns_ns": post_route.get("setup_wns_ns"),
                "tns_ns": post_route.get("setup_tns_ns"),
                "lut": post_route.get("lut"),
                "ff": post_route.get("ff"),
                "block_ram_tile": post_route.get("block_ram_tile"),
                "dsp": post_route.get("dsp"),
            },
            "report_dir": route.get("report_dir"),
        },
        "av7k325_ffg900_micro_route_attempt": {
            "status": board_package_attempt.get("status"),
            "returncode": board_package_attempt.get("returncode"),
            "failure_class": board_package_failure_class,
            "contributes_to_gate": False,
        },
        "power": {
            "status": "NOT_MEASURED",
            "reason": (
                "The route flow emitted a vectorless Vivado estimate only; "
                "no external-instrument board measurement was executed."
            ),
        },
        "evidence": evidence,
        "claim_boundary": (
            "PASS opens MBConv-k5-e3 as a Stage4 accuracy candidate only. "
            "The successful route is a 5 ns xc7k325t-ffg676-2 out-of-context "
            "operator micro-harness on the same XC7K325T speed grade. It is "
            "not the AV7K325 ffg900 complete-network route, RTL co-sim, "
            "bitstream, board, or power evidence. The ffg900 micro-route "
            "attempt failed in the Vivado runtime Tcl loader and is preserved "
            "as non-gating negative environment evidence."
        ),
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
