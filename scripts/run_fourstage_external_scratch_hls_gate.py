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
import subprocess
import sys
from textwrap import dedent
from typing import Any

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
    AV7K325_BRAM18_EQUIV,
    BRAM18_CAPACITY_BITS,
    DEFAULT_CSIM_SUMMARY,
    TARGET_CLOCK_NS,
    TARGET_PART,
    SynthesizableEmitter,
    find_csynth_report,
)


DEFAULT_SUMMARY = (
    ROOT
    / "artifacts"
    / "sonar_fourstage_operator_v2"
    / "fourstage_external_scratch_hls_summary.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "sonar_fourstage_operator_v2" / "hxs"


class ExternalScratchEmitter(SynthesizableEmitter):
    """Reuse the audited layer arithmetic with external scratch activations."""

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
            "implementation_style": "external_scratch_pixel_tiled_v1",
            "max_tensor_elements": int(self.max_tensor_size),
            "max_residual_elements": int(self.max_residual_size),
            "external_scratch_bytes": int(external_scratch_bytes),
            "local_activation_buffer_bytes": 0,
            "weight_bytes": int(weight_bytes),
            "bias_bytes": int(bias_bytes),
            "weight_bias_bram18_lower_bound": weight_bias_bram18_lower_bound,
            "av7k325_bram18_equivalent_capacity": AV7K325_BRAM18_EQUIV,
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
        return top[:start] + new + top[old_end:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csim-summary", default=str(DEFAULT_CSIM_SUMMARY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--vitis-hls", default=str(DEFAULT_VITIS_HLS))
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument(
        "--run-hls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Vitis HLS csim_design followed by csynth_design.",
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


def tcl_script() -> str:
    return dedent(
        f"""
        open_project -reset project
        set_top fourstage_top_hls
        add_files top.cpp -cflags {{-std=c++14 -O2}}
        add_files -tb tb.cpp -cflags {{-std=c++14 -O2}}
        open_solution -reset solution1 -flow_target vivado
        set_part {{{TARGET_PART}}}
        create_clock -period {TARGET_CLOCK_NS:.3f} -name default
        if {{[catch {{csim_design}} result]}} {{
          puts "CSIM_FAILED $result"
          exit 1
        }}
        csynth_design
        exit
        """
    ).lstrip()


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


def run_one_candidate(
    candidate: dict[str, Any],
    *,
    output_root: Path,
    vitis_hls: Path,
    run_hls: bool,
    timeout_minutes: int,
    int8_summary: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = Path(candidate["source_checkpoint"]["path"]).resolve()
    calibration_path = Path(candidate["activation_calibration"]["path"]).resolve()
    reference_path = Path(candidate["python_int8_reference"]["path"]).resolve()
    if sha256_file(checkpoint) != candidate["source_checkpoint"]["sha256"]:
        raise RuntimeError(f"{candidate['arch_id']}: checkpoint SHA mismatch")
    if sha256_file(calibration_path) != candidate["activation_calibration"]["sha256"]:
        raise RuntimeError(f"{candidate['arch_id']}: activation calibration SHA mismatch")
    if sha256_file(reference_path) != candidate["python_int8_reference"]["sha256"]:
        raise RuntimeError(f"{candidate['arch_id']}: Python INT8 reference SHA mismatch")

    calibration = read_json(calibration_path)
    reference_payload = read_json(reference_path)
    model, _architecture, _payload, _class_names = load_checkpoint_model(
        checkpoint, device="cpu"
    )
    emitter = ExternalScratchEmitter(model, calibration)
    preflight = emitter.preflight()
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
        ),
        encoding="utf-8",
    )
    tcl_path.write_text(tcl_script(), encoding="utf-8")
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
            "sample_count": len(input_samples),
            "output_count": len(input_samples) * OUTPUT_CLASSES,
            "csim_result": None,
            "vitis_returncode": None,
            "timeout_minutes": int(timeout_minutes),
            "failure_category": None,
            "vitis_log": None,
            "csynth_report": None,
            "metrics": None,
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
    status = "PASS" if returncode == 0 and csim_pass and report_path is not None else "FAIL"
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
    return {
        "role": candidate["role"],
        "arch_id": candidate["arch_id"],
        "status": status,
        "hls_project_dir": str(hls_dir.resolve()),
        "generated": generated,
        "preflight": preflight,
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
        "claim_boundary": (
            "External-scratch full-network HLS C-sim/synthesis evidence only. "
            "This is not RTL co-sim, place-and-route, bitstream, COM5, or "
            "measured power."
        ),
    }


def main() -> int:
    args = parse_args()
    csim_summary_path = Path(args.csim_summary).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve()
    vitis_hls = Path(args.vitis_hls).expanduser().resolve()
    csim_summary = read_json(csim_summary_path)
    if csim_summary.get("status") != "PASS" or not csim_summary.get("zero_mismatch"):
        raise RuntimeError("C-sim zero-mismatch gate must PASS before HLS synthesis")
    if csim_summary.get("contract") != CALIBRATION_CONTRACT:
        raise RuntimeError("unexpected C-sim calibration contract")
    int8_summary_path = Path(csim_summary["int8_reference_summary"]["path"]).resolve()
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
    if args.candidate_limit is not None:
        candidates = candidates[: int(args.candidate_limit)]
    if args.run_hls and not vitis_hls.is_file():
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "BLOCKED",
            "gate": "full_network_external_scratch_hls",
            "implementation_style": "external_scratch_pixel_tiled_v1",
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

    rows = [
        run_one_candidate(
            candidate,
            output_root=output_root,
            vitis_hls=vitis_hls,
            run_hls=bool(args.run_hls),
            timeout_minutes=int(args.timeout_minutes),
            int8_summary=int8_summary,
        )
        for candidate in candidates
    ]
    formal_scope = args.candidate_limit is None and len(rows) == formal_candidate_count
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
        "implementation_style": "external_scratch_pixel_tiled_v1",
        "target_part": TARGET_PART,
        "target_clock_ns": TARGET_CLOCK_NS,
        "csim_zero_mismatch_summary": evidence(csim_summary_path),
        "int8_reference_summary": evidence(int8_summary_path),
        "vitis_hls": {"path": str(vitis_hls), "exists": vitis_hls.is_file()},
        "git": {
            "head": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "status_porcelain": git_output("status", "--porcelain=v1").splitlines(),
        },
        "candidate_count": len(rows),
        "formal_candidate_count": formal_candidate_count,
        "formal_scope": formal_scope,
        "candidate_limit": args.candidate_limit,
        "candidates": rows,
        "downstream_gates": {
            "rtl_cosim": "PENDING" if status == "PASS" else "NOT_RUN_HLS_SYNTHESIS_NOT_PASSED",
            "place_and_route_5ns": (
                "NOT_RUN_RTL_COSIM_NOT_PASSED"
                if status == "PASS"
                else "NOT_RUN_HLS_SYNTHESIS_NOT_PASSED"
            ),
            "bitstream": "NOT_GENERATED",
            "com5_board_latency": "NOT_RUN",
            "external_meter_power": "NOT_MEASURED",
        },
        "claim_boundary": (
            "This is complete-network Vitis HLS C-sim/synthesis evidence for "
            "an external-scratch pixel-tiled probe. It is distinct from RTL "
            "co-sim, full route, bitstream, COM5 board execution, and "
            "external-meter power; it is also not a final optimized streaming "
            "line-buffer datapath."
        ),
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    write_json(summary_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if status in {"PASS", "PARTIAL_PASS_NOT_FORMAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
