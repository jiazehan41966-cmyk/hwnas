#!/usr/bin/env python3
"""Run the next full-network HLS gate with external scratch tensors.

The static full-buffer probe in ``run_fourstage_hls_synthesis_gate.py`` reached
Vitis scheduling/binding but timed out without a csynth report.  This gate keeps
the frozen four-stage architecture and the same INT8 arithmetic contract, but
moves large intermediate activation tensors out of local HLS arrays and into
explicit AXI scratch buffers.

This is a deliberately modest implementation step:

* it is full-network and consumes the real checkpoint/calibration/reference
  artifacts;
* it runs a generated C-sim testbench before synthesis;
* it is layer-serial / pixel-tiled and uses external scratch tensors, so it is
  not yet a final streaming line-buffer or board ABI claim.

Passing this gate still does not imply RTL co-sim, route, bitstream, COM5 board
execution, or external-meter power.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from textwrap import dedent
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

from hwnas_fpga.deploy.inference import load_checkpoint_model  # noqa: E402
from hwnas_fpga.fourstage_selection import canonical_sha256  # noqa: E402
from hwnas_fpga.hardware import parse_hls_report  # noqa: E402
from hwnas_fpga.training.protocol_reporting import sha256_file  # noqa: E402
from run_fourstage_csim_gate import (  # noqa: E402
    CALIBRATION_CONTRACT,
    DEFAULT_VITIS_HLS,
    INPUT_SIZE,
    OUTPUT_CLASSES,
    array_initializer,
    cpp_string,
    evidence,
    read_json,
    rebuild_inputs_from_summary_if_needed,
    safe_name,
    write_json,
)
from run_fourstage_hls_synthesis_gate import (  # noqa: E402
    BRAM18_CAPACITY_BITS,
    DEFAULT_CSIM_SUMMARY,
    SynthesizableEmitter,
    find_csynth_report,
)


DEFAULT_SUMMARY = (
    ROOT
    / "artifacts"
    / "sonar_fourstage_operator_v2"
    / "fourstage_external_scratch_hls_summary.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "zcu104_full_campaign_v1" / "hls"
DEFAULT_PORTABLE_INPUT_ROOT = (
    ROOT / "artifacts" / "zcu104_full_campaign_v1" / "inputs" / "hls_reference"
)
ZCU104_TARGET_PART = "xczu7ev-ffvc1156-2-e"
ZCU104_TARGET_CLOCK_NS = 5.0
ZCU104_RESOURCE_LIMITS = {
    "lut": 230_400,
    "ff": 460_800,
    "dsp": 1_728,
    "bram_18k": 624,
}


class ExternalScratchEmitter(SynthesizableEmitter):
    """Reuse the audited layer arithmetic with external scratch activations."""

    def __init__(
        self,
        model: torch.nn.Module,
        calibration: dict[str, Any],
        *,
        io_interface: str = "axi_dma",
    ):
        super().__init__(model, calibration)
        if io_interface not in {"m_axi", "axi_dma"}:
            raise ValueError(f"unsupported external-scratch I/O interface: {io_interface}")
        self.io_interface = io_interface

    def preflight(self) -> dict[str, Any]:
        self.emit_synth_ops()
        external_scratch_bytes = (
            int(self.max_tensor_size) * 2
            + int(self.max_residual_size) * 2
            + int(self.max_residual_size)
        )
        weight_bytes = 0
        bias_bytes = 0
        for layer in self.conv_layers:
            weight_bytes += len(layer.weights)
            bias_bytes += len(layer.bias) * 4
        for layer in self.linear_layers:
            weight_bytes += len(layer.weights)
            bias_bytes += len(layer.bias) * 4
        weight_bias_bits = (weight_bytes + bias_bytes) * 8
        weight_bias_bram18_lower_bound = int(
            math.ceil(weight_bias_bits / BRAM18_CAPACITY_BITS)
        )
        return {
            "implementation_style": (
                "external_scratch_axi_dma_stream_v2"
                if self.io_interface == "axi_dma"
                else "external_scratch_pixel_tiled_v1"
            ),
            "io_interface": self.io_interface,
            "max_tensor_elements": int(self.max_tensor_size),
            "max_residual_elements": int(self.max_residual_size),
            "external_scratch_bytes": int(external_scratch_bytes),
            "local_activation_buffer_bytes": 0,
            "weight_bytes": int(weight_bytes),
            "bias_bytes": int(bias_bytes),
            "weight_bias_bram18_lower_bound": weight_bias_bram18_lower_bound,
            "claim_boundary": (
                "Large activations are modeled as explicit AXI scratch "
                "buffers. This reduces local activation BRAM pressure but is "
                "not a final line-buffered or routed deployment datapath."
            ),
        }

    def top_cpp(self) -> str:
        top = super().top_cpp()
        old_start = (
            f'extern "C" void fourstage_top_hls(const i8 input[{INPUT_SIZE}], '
            f"i8 output[{OUTPUT_CLASSES}]) {{"
        )
        start = top.find(old_start)
        if start < 0:
            raise RuntimeError("static-buffer top signature was not found for replacement")
        old_end_marker = f"copy_tensor(input, buf0, {INPUT_SIZE});\n"
        old_end = top.find(old_end_marker, start)
        if old_end < 0:
            raise RuntimeError("static-buffer input copy was not found for replacement")
        old_end += len(old_end_marker)
        new = dedent(
            f"""
            extern "C" void fourstage_top_hls(
                const i8 input[{INPUT_SIZE}],
                i8 scratch0[MAX_TENSOR_SIZE],
                i8 scratch1[MAX_TENSOR_SIZE],
                i8 residual[MAX_RESIDUAL_SIZE],
                i8 identity[MAX_RESIDUAL_SIZE],
                i8 output[{OUTPUT_CLASSES}]) {{
            #pragma HLS INTERFACE m_axi port=input depth={INPUT_SIZE} offset=slave bundle=gmem0
            #pragma HLS INTERFACE m_axi port=scratch0 depth=MAX_TENSOR_SIZE offset=slave bundle=gmem1
            #pragma HLS INTERFACE m_axi port=scratch1 depth=MAX_TENSOR_SIZE offset=slave bundle=gmem2
            #pragma HLS INTERFACE m_axi port=residual depth=MAX_RESIDUAL_SIZE offset=slave bundle=gmem3
            #pragma HLS INTERFACE m_axi port=identity depth=MAX_RESIDUAL_SIZE offset=slave bundle=gmem4
            #pragma HLS INTERFACE m_axi port=output depth={OUTPUT_CLASSES} offset=slave bundle=gmem5
            #pragma HLS INTERFACE s_axilite port=input bundle=control
            #pragma HLS INTERFACE s_axilite port=scratch0 bundle=control
            #pragma HLS INTERFACE s_axilite port=scratch1 bundle=control
            #pragma HLS INTERFACE s_axilite port=residual bundle=control
            #pragma HLS INTERFACE s_axilite port=identity bundle=control
            #pragma HLS INTERFACE s_axilite port=output bundle=control
            #pragma HLS INTERFACE s_axilite port=return bundle=control
              i8* buf0 = scratch0;
              i8* buf1 = scratch1;
              copy_tensor(input, buf0, {INPUT_SIZE});
            """
        ).lstrip()
        rewritten = top[:start] + new + top[old_end:]
        if self.io_interface == "m_axi":
            return rewritten

        axis_signature = dedent(
            f"""
            extern "C" void fourstage_top_hls(
                hls::stream<axis_i8>& input_stream,
                i8 scratch0[MAX_TENSOR_SIZE],
                i8 scratch1[MAX_TENSOR_SIZE],
                i8 residual[MAX_RESIDUAL_SIZE],
                i8 identity[MAX_RESIDUAL_SIZE],
                hls::stream<axis_i8>& output_stream) {{
            #pragma HLS INTERFACE axis port=input_stream
            #pragma HLS INTERFACE m_axi port=scratch0 depth=MAX_TENSOR_SIZE offset=slave bundle=gmem0
            #pragma HLS INTERFACE m_axi port=scratch1 depth=MAX_TENSOR_SIZE offset=slave bundle=gmem1
            #pragma HLS INTERFACE m_axi port=residual depth=MAX_RESIDUAL_SIZE offset=slave bundle=gmem2
            #pragma HLS INTERFACE m_axi port=identity depth=MAX_RESIDUAL_SIZE offset=slave bundle=gmem3
            #pragma HLS INTERFACE axis port=output_stream
            #pragma HLS INTERFACE s_axilite port=scratch0 bundle=control
            #pragma HLS INTERFACE s_axilite port=scratch1 bundle=control
            #pragma HLS INTERFACE s_axilite port=residual bundle=control
            #pragma HLS INTERFACE s_axilite port=identity bundle=control
            #pragma HLS INTERFACE s_axilite port=return bundle=control
              i8* buf0 = scratch0;
              i8* buf1 = scratch1;
            input_stream_loop:
              for (int index = 0; index < {INPUT_SIZE}; ++index) {{
                axis_i8 word = input_stream.read();
                buf0[index] = static_cast<i8>(word.data.to_int());
              }}
            """
        ).lstrip()
        if new not in rewritten:
            raise RuntimeError("external-scratch memory signature was not found")
        rewritten = rewritten.replace(new, axis_signature, 1)
        axis_output = dedent(
            f"""
            output_stream_loop:
              for (int index = 0; index < {OUTPUT_CLASSES}; ++index) {{
                axis_i8 word;
                word.data = buf1[index];
                word.keep = 1;
                word.strb = 1;
                word.last = (index == {OUTPUT_CLASSES - 1});
                output_stream.write(word);
              }}
            """
        ).strip()
        output_start = rewritten.rfind("output_loop:")
        function_close = rewritten.rfind("}")
        if output_start < 0 or function_close <= output_start:
            raise RuntimeError("external-scratch memory output loop was not found")
        rewritten = rewritten[:output_start] + axis_output + "\n" + rewritten[function_close:]
        return (
            '#include "ap_axi_sdata.h"\n#include "hls_stream.h"\n'
            "using axis_i8 = ap_axis<8, 0, 0, 0>;\n"
            + rewritten
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csim-summary", default=str(DEFAULT_CSIM_SUMMARY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--vitis-hls", default=str(DEFAULT_VITIS_HLS))
    parser.add_argument("--portable-input-root", default=str(DEFAULT_PORTABLE_INPUT_ROOT))
    parser.add_argument("--target-part", default=ZCU104_TARGET_PART)
    parser.add_argument("--target-clock-ns", type=float, default=ZCU104_TARGET_CLOCK_NS)
    parser.add_argument("--max-lut", type=int, default=ZCU104_RESOURCE_LIMITS["lut"])
    parser.add_argument("--max-ff", type=int, default=ZCU104_RESOURCE_LIMITS["ff"])
    parser.add_argument("--max-dsp", type=int, default=ZCU104_RESOURCE_LIMITS["dsp"])
    parser.add_argument("--max-bram18", type=int, default=ZCU104_RESOURCE_LIMITS["bram_18k"])
    parser.add_argument("--expected-toolchain-version", default="2022.2")
    parser.add_argument(
        "--io-interface",
        choices=("axi_dma", "m_axi"),
        default="axi_dma",
        help="Use AXI4-Stream input/output for AXI DMA or legacy memory-mapped I/O.",
    )
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument(
        "--role",
        action="append",
        default=[],
        help="Candidate role to include; repeatable. Defaults to all candidates.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help=(
            "Limit the generated C/RTL testbench to the first N real INT8 "
            "reference samples. This is a co-sim feasibility knob and makes "
            "the summary non-formal when set."
        ),
    )
    parser.add_argument(
        "--run-hls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Vitis HLS csim_design followed by csynth_design.",
    )
    parser.add_argument(
        "--export-ip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export the synthesized kernel as an IP-catalog package for ZCU104 block design.",
    )
    parser.add_argument("--timeout-minutes", type=int, default=90)
    return parser.parse_args()


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def git_status_excluding(*paths: Path) -> list[str]:
    """Return provenance dirt while excluding files produced by this invocation."""
    excluded = {
        path.resolve().relative_to(ROOT).as_posix()
        for path in paths
        if path.resolve().is_relative_to(ROOT)
    }
    rows = []
    for row in git_output("status", "--porcelain=v1").splitlines():
        status_path = row[3:].replace("\\", "/") if len(row) > 3 else ""
        if status_path not in excluded:
            rows.append(row)
    return rows


def probe_vitis_hls_version(vitis_hls: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(vitis_hls), "-version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
    match = re.search(r"(?:Vitis HLS|Vivado HLS)\s+v?(\d{4}\.\d)", combined, re.IGNORECASE)
    if match is None:
        match = re.search(r"\b(\d{4}\.\d)\b", combined)
    return {
        "path": str(vitis_hls),
        "returncode": int(completed.returncode),
        "version": match.group(1) if match else "",
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
    }


def tcl_script(
    *, target_part: str, target_clock_ns: float, export_ip: bool = True
) -> str:
    export_command = "export_design -format ip_catalog" if export_ip else ""
    return dedent(
        f"""
        open_project -reset project
        set_top fourstage_top_hls
        add_files top.cpp -cflags {{-std=c++14 -O2}}
        add_files -tb tb.cpp -cflags {{-std=c++14 -O2}}
        open_solution -reset solution1 -flow_target vivado
        set_part {{{target_part}}}
        create_clock -period {target_clock_ns:.3f} -name default
        if {{[catch {{csim_design}} result]}} {{
          puts "CSIM_FAILED $result"
          exit 1
        }}
        csynth_design
        {export_command}
        exit
        """
    ).lstrip()


def _bound_path(record: dict[str, Any]) -> Path:
    path = Path(str(record["path"])).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _verified_bound_path(record: dict[str, Any], label: str) -> Path:
    path = _bound_path(record)
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    if sha256_file(path) != str(record["sha256"]):
        raise RuntimeError(f"{label} SHA256 mismatch: {path}")
    return path


def load_portable_inputs(
    candidate: dict[str, Any], portable_input_root: Path
) -> dict[str, Any] | None:
    manifest_path = portable_input_root / safe_name(str(candidate["role"])) / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = read_json(manifest_path)
    if manifest.get("arch_id") != candidate.get("arch_id"):
        raise RuntimeError(f"portable input arch_id mismatch: {manifest_path}")
    checkpoint = _verified_bound_path(manifest["checkpoint"], "portable checkpoint")
    calibration = _verified_bound_path(
        manifest["activation_calibration"], "portable activation calibration"
    )
    reference = _verified_bound_path(
        manifest["python_int8_reference"], "portable Python INT8 reference"
    )
    vector_records = manifest["test_vectors"]
    inputs_path = _verified_bound_path(
        vector_records["inputs"], "portable input vectors"
    )
    expected_path = _verified_bound_path(
        vector_records["expected"], "portable expected vectors"
    )
    inputs = np.load(inputs_path, allow_pickle=False).astype(np.int8, copy=False).tolist()
    expected = np.load(expected_path, allow_pickle=False).astype(np.int8, copy=False).tolist()
    if len(inputs) != int(manifest.get("reference_sample_count") or -1):
        raise RuntimeError(f"portable input sample count mismatch: {manifest_path}")
    return {
        "manifest": evidence(manifest_path),
        "checkpoint": checkpoint,
        "calibration": calibration,
        "reference": reference,
        "inputs": inputs,
        "expected": expected,
    }


def check_hls_resources(
    metrics: dict[str, Any] | None,
    limits: dict[str, int],
) -> dict[str, Any]:
    values = {
        key: (None if not isinstance(metrics, dict) else metrics.get(key))
        for key in limits
    }
    violations: dict[str, dict[str, int | None]] = {}
    for key, limit in limits.items():
        value = values[key]
        if value is None or int(value) > int(limit):
            violations[key] = {
                "value": None if value is None else int(value),
                "limit": int(limit),
            }
    return {
        "status": "PASS" if not violations else "FAIL",
        "accounting": {
            "bram": "BRAM18",
            "bram36": "reported separately when present",
            "uram": "separate; not folded into BRAM18",
        },
        "limits": limits,
        "values": values,
        "violations": violations,
        "claim_boundary": "HLS estimates are a pre-route resource gate only.",
    }


def run_vitis_hls(
    *, vitis_hls: Path, hls_dir: Path, timeout_minutes: int
) -> tuple[int, Path, bool]:
    log_path = hls_dir / "vitis_hls_external_scratch.log"
    command = [str(vitis_hls), "-f", "run_synthesis.tcl"]
    process = subprocess.Popen(
        command,
        cwd=hls_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(
            timeout=(None if timeout_minutes <= 0 else int(timeout_minutes) * 60)
        )
    except subprocess.TimeoutExpired:
        timed_out = True
        if sys.platform.startswith("win"):
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                check=False,
                capture_output=True,
                text=True,
            )
        else:  # pragma: no cover - current execution is Windows.
            process.kill()
        stdout, stderr = process.communicate()
    log_path.write_text(
        "COMMAND\n"
        + " ".join(command)
        + ("\nTIMEOUT\n" if timed_out else "\n")
        + "\nSTDOUT\n"
        + (stdout or "")
        + "\nSTDERR\n"
        + (stderr or "")
        + "\n",
        encoding="utf-8",
        errors="replace",
    )
    return (-1 if timed_out else int(process.returncode)), log_path, timed_out


def classify_failure(
    *,
    status: str,
    timed_out: bool,
    csim_pass: bool,
    returncode: int,
    report_path: Path | None,
    log_path: Path,
) -> str | None:
    if status == "PASS":
        return None
    if timed_out:
        return "EXTERNAL_SCRATCH_HLS_TIMEOUT"
    if not csim_pass:
        return "EXTERNAL_SCRATCH_CSIM_FAILURE"
    if report_path is None:
        return "EXTERNAL_SCRATCH_HLS_SYNTHESIS_NO_REPORT"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "file name(s)" in log_text and "too long" in log_text:
        return "EXTERNAL_SCRATCH_HLS_ARTIFACT_PATH_TOO_LONG"
    if returncode != 0:
        return "EXTERNAL_SCRATCH_HLS_RETURNCODE_NONZERO_WITH_REPORT"
    return "EXTERNAL_SCRATCH_HLS_FAILURE"


def testbench_cpp(
    *,
    input_samples: list[list[int]],
    expected_outputs: list[list[int]],
    result_path: Path,
    max_tensor_size: int,
    max_residual_size: int,
    io_interface: str = "m_axi",
) -> str:
    if io_interface == "axi_dma":
        return testbench_cpp_axi_dma(
            input_samples=input_samples,
            expected_outputs=expected_outputs,
            result_path=result_path,
            max_tensor_size=max_tensor_size,
            max_residual_size=max_residual_size,
        )
    samples = ",\n".join(
        "{\n" + array_initializer(sample) + "\n}" for sample in input_samples
    )
    expected = ",\n".join(
        "{ " + ", ".join(str(int(value)) for value in row) + " }"
        for row in expected_outputs
    )
    sample_count = len(input_samples)
    return dedent(
        f"""
        #include <fstream>
        #include <iostream>

        using i8 = signed char;

        static const int MAX_TENSOR_SIZE = {int(max_tensor_size)};
        static const int MAX_RESIDUAL_SIZE = {int(max_residual_size)};

        extern "C" void fourstage_top_hls(
            const i8 input[{INPUT_SIZE}],
            i8 scratch0[MAX_TENSOR_SIZE],
            i8 scratch1[MAX_TENSOR_SIZE],
            i8 residual[MAX_RESIDUAL_SIZE],
            i8 identity[MAX_RESIDUAL_SIZE],
            i8 output[{OUTPUT_CLASSES}]);

        static const i8 SAMPLE_INPUTS[{sample_count}][{INPUT_SIZE}] = {{
        {samples}
        }};

        static const i8 EXPECTED[{sample_count}][{OUTPUT_CLASSES}] = {{
        {expected}
        }};

        static i8 SCRATCH0[MAX_TENSOR_SIZE];
        static i8 SCRATCH1[MAX_TENSOR_SIZE];
        static i8 RESIDUAL[MAX_RESIDUAL_SIZE];
        static i8 IDENTITY[MAX_RESIDUAL_SIZE];

        int main() {{
          const char* result_path = "{cpp_string(result_path)}";
          int mismatch_count = 0;
          int checked_outputs = 0;
          int max_abs_mismatch = 0;
          for (int sample = 0; sample < {sample_count}; ++sample) {{
            i8 output[{OUTPUT_CLASSES}] = {{0}};
            fourstage_top_hls(
              SAMPLE_INPUTS[sample],
              SCRATCH0,
              SCRATCH1,
              RESIDUAL,
              IDENTITY,
              output);
            for (int index = 0; index < {OUTPUT_CLASSES}; ++index) {{
              ++checked_outputs;
              const int diff =
                static_cast<int>(output[index]) - static_cast<int>(EXPECTED[sample][index]);
              const int abs_diff = diff < 0 ? -diff : diff;
              if (diff != 0) {{
                ++mismatch_count;
                if (abs_diff > max_abs_mismatch) {{
                  max_abs_mismatch = abs_diff;
                }}
                std::cout << "MISMATCH sample=" << sample
                          << " output=" << index
                          << " got=" << static_cast<int>(output[index])
                          << " expected=" << static_cast<int>(EXPECTED[sample][index])
                          << "\\n";
              }}
            }}
          }}
          std::ofstream json(result_path);
          json << "{{\\n";
          json << "  \\"schema_version\\": 1,\\n";
          json << "  \\"status\\": \\"" << (mismatch_count == 0 ? "PASS" : "FAIL") << "\\",\\n";
          json << "  \\"sample_count\\": {sample_count},\\n";
          json << "  \\"output_count\\": " << checked_outputs << ",\\n";
          json << "  \\"mismatch_count\\": " << mismatch_count << ",\\n";
          json << "  \\"max_abs_mismatch\\": " << max_abs_mismatch << "\\n";
          json << "}}\\n";
          json.close();
          std::cout << "CSIM_STATUS " << (mismatch_count == 0 ? "PASS" : "FAIL")
                    << " mismatch_count=" << mismatch_count
                    << " checked_outputs=" << checked_outputs << "\\n";
          return mismatch_count == 0 ? 0 : 1;
        }}
        """
    ).lstrip()


def testbench_cpp_axi_dma(
    *,
    input_samples: list[list[int]],
    expected_outputs: list[list[int]],
    result_path: Path,
    max_tensor_size: int,
    max_residual_size: int,
) -> str:
    samples = ",\n".join(
        "{\n" + array_initializer(sample) + "\n}" for sample in input_samples
    )
    expected = ",\n".join(
        "{ " + ", ".join(str(int(value)) for value in row) + " }"
        for row in expected_outputs
    )
    sample_count = len(input_samples)
    return dedent(
        f"""
        #include "ap_axi_sdata.h"
        #include "hls_stream.h"
        #include <fstream>
        #include <iostream>

        using i8 = signed char;
        using axis_i8 = ap_axis<8, 0, 0, 0>;

        static const int MAX_TENSOR_SIZE = {int(max_tensor_size)};
        static const int MAX_RESIDUAL_SIZE = {int(max_residual_size)};

        extern "C" void fourstage_top_hls(
            hls::stream<axis_i8>& input_stream,
            i8 scratch0[MAX_TENSOR_SIZE],
            i8 scratch1[MAX_TENSOR_SIZE],
            i8 residual[MAX_RESIDUAL_SIZE],
            i8 identity[MAX_RESIDUAL_SIZE],
            hls::stream<axis_i8>& output_stream);

        static const i8 SAMPLE_INPUTS[{sample_count}][{INPUT_SIZE}] = {{
        {samples}
        }};

        static const i8 EXPECTED[{sample_count}][{OUTPUT_CLASSES}] = {{
        {expected}
        }};

        static i8 SCRATCH0[MAX_TENSOR_SIZE];
        static i8 SCRATCH1[MAX_TENSOR_SIZE];
        static i8 RESIDUAL[MAX_RESIDUAL_SIZE];
        static i8 IDENTITY[MAX_RESIDUAL_SIZE];

        int main() {{
          const char* result_path = "{cpp_string(result_path)}";
          int mismatch_count = 0;
          int checked_outputs = 0;
          int max_abs_mismatch = 0;
          for (int sample = 0; sample < {sample_count}; ++sample) {{
            hls::stream<axis_i8> input_stream;
            hls::stream<axis_i8> output_stream;
            for (int index = 0; index < {INPUT_SIZE}; ++index) {{
              axis_i8 word;
              word.data = SAMPLE_INPUTS[sample][index];
              word.keep = 1;
              word.strb = 1;
              word.last = (index == {INPUT_SIZE - 1});
              input_stream.write(word);
            }}
            fourstage_top_hls(
              input_stream,
              SCRATCH0,
              SCRATCH1,
              RESIDUAL,
              IDENTITY,
              output_stream);
            for (int index = 0; index < {OUTPUT_CLASSES}; ++index) {{
              axis_i8 word = output_stream.read();
              const int got = word.data.to_int();
              ++checked_outputs;
              const int diff = got - static_cast<int>(EXPECTED[sample][index]);
              const int abs_diff = diff < 0 ? -diff : diff;
              if (diff != 0) {{
                ++mismatch_count;
                if (abs_diff > max_abs_mismatch) {{
                  max_abs_mismatch = abs_diff;
                }}
                std::cout << "MISMATCH sample=" << sample
                          << " output=" << index
                          << " got=" << got
                          << " expected=" << static_cast<int>(EXPECTED[sample][index])
                          << "\\n";
              }}
              if ((index == {OUTPUT_CLASSES - 1}) != static_cast<bool>(word.last)) {{
                ++mismatch_count;
              }}
            }}
          }}
          std::ofstream json(result_path);
          json << "{{\\n";
          json << "  \\"schema_version\\": 1,\\n";
          json << "  \\"status\\": \\"" << (mismatch_count == 0 ? "PASS" : "FAIL") << "\\",\\n";
          json << "  \\"sample_count\\": {sample_count},\\n";
          json << "  \\"output_count\\": " << checked_outputs << ",\\n";
          json << "  \\"mismatch_count\\": " << mismatch_count << ",\\n";
          json << "  \\"max_abs_mismatch\\": " << max_abs_mismatch << "\\n";
          json << "}}\\n";
          json.close();
          std::cout << "CSIM_STATUS " << (mismatch_count == 0 ? "PASS" : "FAIL")
                    << " mismatch_count=" << mismatch_count
                    << " checked_outputs=" << checked_outputs << "\\n";
          return mismatch_count == 0 ? 0 : 1;
        }}
        """
    ).lstrip()


def run_one_candidate(
    candidate: dict[str, Any],
    *,
    output_root: Path,
    vitis_hls: Path,
    run_hls: bool,
    timeout_minutes: int,
    int8_summary: dict[str, Any],
    sample_limit: int | None,
    portable_input_root: Path,
    target_part: str,
    target_clock_ns: float,
    resource_limits: dict[str, int],
    io_interface: str,
    export_ip: bool,
) -> dict[str, Any]:
    portable = load_portable_inputs(candidate, portable_input_root)
    if portable is not None:
        checkpoint = portable["checkpoint"]
        calibration_path = portable["calibration"]
        reference_path = portable["reference"]
    else:
        checkpoint = _verified_bound_path(
            candidate["source_checkpoint"], f"{candidate['arch_id']} checkpoint"
        )
        calibration_path = _verified_bound_path(
            candidate["activation_calibration"],
            f"{candidate['arch_id']} activation calibration",
        )
        reference_path = _verified_bound_path(
            candidate["python_int8_reference"],
            f"{candidate['arch_id']} Python INT8 reference",
        )

    calibration = read_json(calibration_path)
    reference_payload = read_json(reference_path)
    model, _architecture, _payload, _class_names = load_checkpoint_model(
        checkpoint, device="cpu"
    )
    emitter = ExternalScratchEmitter(model, calibration, io_interface=io_interface)
    preflight = emitter.preflight()
    if portable is not None:
        input_samples = portable["inputs"]
        expected_outputs = portable["expected"]
    else:
        input_samples = rebuild_inputs_from_summary_if_needed(
            reference_payload,
            calibration,
            candidate_row=candidate,
            int8_summary=int8_summary,
        )
        expected_outputs = [
            [int(value) for value in row["logits_int8"]]
            for row in reference_payload["reference_records"]
        ]
    reference_sample_count = len(input_samples)
    if sample_limit is not None:
        if int(sample_limit) <= 0:
            raise RuntimeError("--sample-limit must be positive when set")
        input_samples = input_samples[: int(sample_limit)]
        expected_outputs = expected_outputs[: int(sample_limit)]
        if not input_samples:
            raise RuntimeError("sample-limit removed all testbench samples")

    hls_dir = (
        output_root
        / safe_name(candidate["role"])
    )
    hls_dir.mkdir(parents=True, exist_ok=True)
    top_path = hls_dir / "top.cpp"
    tb_path = hls_dir / "tb.cpp"
    tcl_path = hls_dir / "run_synthesis.tcl"
    result_path = hls_dir / "csim_result.json"
    top_path.write_text(emitter.top_cpp(), encoding="utf-8")
    tb_path.write_text(
        testbench_cpp(
            input_samples=input_samples,
            expected_outputs=expected_outputs,
            result_path=result_path,
            max_tensor_size=int(preflight["max_tensor_elements"]),
            max_residual_size=int(preflight["max_residual_elements"]),
            io_interface=io_interface,
        ),
        encoding="utf-8",
    )
    tcl_path.write_text(
        tcl_script(
            target_part=target_part,
            target_clock_ns=target_clock_ns,
            export_ip=export_ip,
        ),
        encoding="utf-8",
    )
    if result_path.exists():
        result_path.unlink()

    generated = {
        "top": evidence(top_path),
        "testbench": evidence(tb_path),
        "tcl": evidence(tcl_path),
    }
    if not run_hls:
        return {
            "role": candidate["role"],
            "arch_id": candidate["arch_id"],
            "status": "NOT_RUN",
            "hls_project_dir": str(hls_dir.resolve()),
            "generated": generated,
            "preflight": preflight,
            "target_part": target_part,
            "target_clock_ns": target_clock_ns,
            "io_interface": io_interface,
            "export_ip": export_ip,
            "portable_input_manifest": portable.get("manifest") if portable else None,
            "sample_count": len(input_samples),
            "reference_sample_count": int(reference_sample_count),
            "sample_limit": int(sample_limit) if sample_limit is not None else None,
            "output_count": len(input_samples) * OUTPUT_CLASSES,
            "csim_result": None,
            "vitis_returncode": None,
            "timeout_minutes": int(timeout_minutes),
            "failure_category": None,
            "vitis_log": None,
            "csynth_report": None,
            "metrics": None,
            "hls_resource_check": None,
            "claim_boundary": "HLS source generated only; C-sim/synthesis was not run.",
        }

    returncode, log_path, timed_out = run_vitis_hls(
        vitis_hls=vitis_hls,
        hls_dir=hls_dir,
        timeout_minutes=timeout_minutes,
    )
    result_payload = read_json(result_path) if result_path.is_file() else None
    csim_pass = (
        isinstance(result_payload, dict)
        and result_payload.get("status") == "PASS"
        and int(result_payload.get("mismatch_count", -1)) == 0
    )
    report_path = find_csynth_report(hls_dir)
    metrics = None
    if report_path is not None and report_path.is_file():
        try:
            metrics = parse_hls_report(report_path)
        except Exception as exc:  # pragma: no cover - evidence path still recorded.
            metrics = {"parse_error": repr(exc)}
    resource_check = check_hls_resources(metrics, resource_limits)
    status = (
        "PASS"
        if returncode == 0
        and csim_pass
        and report_path is not None
        and resource_check["status"] == "PASS"
        else "FAIL"
    )
    if timed_out:
        status = "TIMEOUT"
    failure_category = classify_failure(
        status=status,
        timed_out=timed_out,
        csim_pass=csim_pass,
        returncode=returncode,
        report_path=report_path,
        log_path=log_path,
    )
    if resource_check["status"] != "PASS":
        failure_category = "EXTERNAL_SCRATCH_HLS_RESOURCE_INFEASIBLE"
    return {
        "role": candidate["role"],
        "arch_id": candidate["arch_id"],
        "status": status,
        "hls_project_dir": str(hls_dir.resolve()),
        "generated": generated,
        "preflight": preflight,
        "target_part": target_part,
        "target_clock_ns": target_clock_ns,
        "io_interface": io_interface,
        "export_ip": export_ip,
        "portable_input_manifest": portable.get("manifest") if portable else None,
        "reference_sample_count": int(reference_sample_count),
        "sample_limit": int(sample_limit) if sample_limit is not None else None,
        "sample_count": (
            int(result_payload["sample_count"])
            if isinstance(result_payload, dict) and "sample_count" in result_payload
            else len(input_samples)
        ),
        "output_count": (
            int(result_payload["output_count"])
            if isinstance(result_payload, dict) and "output_count" in result_payload
            else len(input_samples) * OUTPUT_CLASSES
        ),
        "mismatch_count": (
            int(result_payload["mismatch_count"])
            if isinstance(result_payload, dict) and "mismatch_count" in result_payload
            else None
        ),
        "max_abs_mismatch": (
            int(result_payload["max_abs_mismatch"])
            if isinstance(result_payload, dict) and "max_abs_mismatch" in result_payload
            else None
        ),
        "csim_result": evidence(result_path) if result_path.is_file() else None,
        "vitis_returncode": int(returncode),
        "timeout_minutes": int(timeout_minutes),
        "failure_category": failure_category,
        "vitis_log": evidence(log_path),
        "csynth_report": evidence(report_path) if report_path is not None else None,
        "metrics": metrics,
        "hls_resource_check": resource_check,
        "claim_boundary": (
            "External-scratch full-network HLS C-sim/synthesis evidence only. "
            "This is not RTL co-sim, place-and-route, bitstream, board transport, or "
            "measured power."
        ),
    }


def main() -> int:
    args = parse_args()
    csim_summary_path = Path(args.csim_summary).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve()
    vitis_hls = Path(args.vitis_hls).expanduser().resolve()
    portable_input_root = Path(args.portable_input_root).expanduser().resolve()
    target_part = str(args.target_part)
    target_clock_ns = float(args.target_clock_ns)
    implementation_style = (
        "external_scratch_axi_dma_stream_v2"
        if args.io_interface == "axi_dma"
        else "external_scratch_pixel_tiled_v1"
    )
    resource_limits = {
        "lut": int(args.max_lut),
        "ff": int(args.max_ff),
        "dsp": int(args.max_dsp),
        "bram_18k": int(args.max_bram18),
    }
    toolchain_probe = None
    csim_summary = read_json(csim_summary_path)
    if csim_summary.get("status") != "PASS" or not csim_summary.get("zero_mismatch"):
        raise RuntimeError("C-sim zero-mismatch gate must PASS before HLS synthesis")
    if csim_summary.get("contract") != CALIBRATION_CONTRACT:
        raise RuntimeError("unexpected C-sim calibration contract")
    int8_record = csim_summary["int8_reference_summary"]
    portable_int8_summary_path = (
        ROOT
        / "artifacts"
        / "sonar_fourstage_operator_v2"
        / "fourstage_int8_reference_summary.json"
    )
    int8_summary_path = (
        portable_int8_summary_path
        if portable_int8_summary_path.is_file()
        else Path(str(int8_record["path"])).expanduser()
    )
    int8_summary_path = int8_summary_path.resolve()
    if sha256_file(int8_summary_path) != str(int8_record["sha256"]):
        raise RuntimeError("Python INT8 reference summary SHA256 mismatch")
    int8_summary = read_json(int8_summary_path)
    if int8_summary.get("status") != "PASS":
        raise RuntimeError("Python INT8 reference summary must PASS before HLS synthesis")
    csim_passed_ids = {
        str(row["arch_id"])
        for row in csim_summary.get("candidates", [])
        if row.get("status") == "PASS" and row.get("mismatch_count") == 0
    }
    candidates = [
        row for row in int8_summary["candidates"] if str(row["arch_id"]) in csim_passed_ids
    ]
    if len(candidates) != len(int8_summary["candidates"]):
        raise RuntimeError("not all INT8 candidates have matching C-sim PASS evidence")
    formal_candidate_count = len(candidates)
    roles = {str(role) for role in args.role}
    if roles:
        candidates = [row for row in candidates if str(row.get("role")) in roles]
    if args.candidate_limit is not None:
        candidates = candidates[: int(args.candidate_limit)]
    if not candidates:
        raise RuntimeError("no candidates selected for external-scratch HLS gate")
    if args.run_hls and not vitis_hls.is_file():
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "BLOCKED",
            "gate": "full_network_external_scratch_hls",
            "implementation_style": implementation_style,
            "io_interface": str(args.io_interface),
            "target_part": target_part,
            "target_clock_ns": target_clock_ns,
            "resource_limits": resource_limits,
            "blocker": f"Vitis HLS executable not found: {vitis_hls}",
            "csim_zero_mismatch_summary": evidence(csim_summary_path),
            "int8_reference_summary": evidence(int8_summary_path),
            "candidate_count": len(candidates),
            "formal_candidate_count": formal_candidate_count,
            "candidates": [],
        }
        payload["payload_sha256"] = canonical_sha256(payload)
        write_json(summary_path, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 2
    if args.run_hls:
        toolchain_probe = probe_vitis_hls_version(vitis_hls)
        if toolchain_probe["version"] != str(args.expected_toolchain_version):
            payload = {
                "schema_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "status": "BLOCKED_TOOLCHAIN_VERSION_MISMATCH",
                "gate": "full_network_external_scratch_hls",
                "expected_toolchain_version": str(args.expected_toolchain_version),
                "vitis_hls": toolchain_probe,
                "candidate_count": len(candidates),
                "formal_candidate_count": formal_candidate_count,
                "candidates": [],
                "claim_boundary": "No HLS result was produced under a mismatched toolchain.",
            }
            payload["payload_sha256"] = canonical_sha256(payload)
            write_json(summary_path, payload)
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 2

    rows = [
        run_one_candidate(
            candidate,
            output_root=output_root,
            vitis_hls=vitis_hls,
            run_hls=bool(args.run_hls),
            timeout_minutes=int(args.timeout_minutes),
            int8_summary=int8_summary,
            sample_limit=args.sample_limit,
            portable_input_root=portable_input_root,
            target_part=target_part,
            target_clock_ns=target_clock_ns,
            resource_limits=resource_limits,
            io_interface=str(args.io_interface),
            export_ip=bool(args.export_ip),
        )
        for candidate in candidates
    ]
    formal_scope = (
        args.candidate_limit is None
        and not roles
        and args.sample_limit is None
        and len(rows) == formal_candidate_count
    )
    rows_pass = rows and all(row["status"] == "PASS" for row in rows)
    if not bool(args.run_hls):
        status = "NOT_RUN"
    elif formal_scope and rows_pass:
        status = "PASS"
    elif rows_pass:
        status = "PARTIAL_PASS_NOT_FORMAL"
    elif any(row["status"] == "TIMEOUT" for row in rows):
        status = "TIMEOUT"
    else:
        status = "FAIL"
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "gate": "full_network_external_scratch_hls",
        "contract": CALIBRATION_CONTRACT,
        "implementation_style": implementation_style,
        "io_interface": str(args.io_interface),
        "export_ip": bool(args.export_ip),
        "target_part": target_part,
        "target_clock_ns": target_clock_ns,
        "resource_limits": resource_limits,
        "resource_accounting": {
            "bram_gate_unit": "BRAM18",
            "bram36_and_uram": "reported separately; not silently folded into BRAM18",
        },
        "csim_zero_mismatch_summary": evidence(csim_summary_path),
        "int8_reference_summary": evidence(int8_summary_path),
        "vitis_hls": (
            toolchain_probe
            if toolchain_probe is not None
            else {"path": str(vitis_hls), "exists": vitis_hls.is_file(), "version": None}
        ),
        "expected_toolchain_version": str(args.expected_toolchain_version),
        "git": {
            "head": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "status_porcelain": git_status_excluding(summary_path),
        },
        "candidate_count": len(rows),
        "formal_candidate_count": formal_candidate_count,
        "formal_scope": formal_scope,
        "candidate_limit": args.candidate_limit,
        "selected_roles": [row.get("role") for row in candidates],
        "sample_limit": args.sample_limit,
        "candidates": rows,
        "downstream_gates": {
            "rtl_cosim": (
                "PENDING"
                if status in {"PASS", "PARTIAL_PASS_NOT_FORMAL"}
                else "NOT_RUN_HLS_SYNTHESIS_NOT_PASSED"
            ),
            "place_and_route_5ns": (
                "NOT_RUN_RTL_COSIM_NOT_PASSED"
                if status in {"PASS", "PARTIAL_PASS_NOT_FORMAL"}
                else "NOT_RUN_HLS_SYNTHESIS_NOT_PASSED"
            ),
            "bitstream": "NOT_GENERATED",
            "board_transport_latency": "NOT_RUN",
            "external_meter_power": "NOT_MEASURED",
        },
        "claim_boundary": (
            (
                "Complete-network AXI-DMA HLS source was generated, but Vitis HLS "
                "C-sim and synthesis were not run. This is portability preflight "
                "evidence only."
            )
            if status == "NOT_RUN"
            else (
                "This is complete-network Vitis HLS C-sim/synthesis evidence for "
                "an external-scratch AXI-DMA probe. It is distinct from RTL co-sim, "
                "full route, bitstream, board execution, and external-meter power; "
                "it is also not a final optimized streaming line-buffer datapath."
            )
        ),
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    write_json(summary_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if status in {"PASS", "PARTIAL_PASS_NOT_FORMAL", "NOT_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
