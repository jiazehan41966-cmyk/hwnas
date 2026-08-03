#!/usr/bin/env python3
"""Run the next full-network HLS synthesis gate for four-stage candidates.

The previous gate, ``run_fourstage_csim_gate.py``, proves integer parity with a
C-simulation harness.  That harness uses ``std::vector`` and is not a
synthesizable datapath.  This script generates a separate static-buffer C++
implementation that Vitis HLS can attempt to synthesize.

The generated implementation is intentionally conservative and auditable:

* it reuses the same quantized weights, scales, rounding and saturation contract
  as the Python INT8 reference and C-sim gate;
* it uses fixed-size local activation buffers so Vitis HLS can synthesize it;
* it records a static-buffer BRAM preflight estimate before invoking synthesis.

Passing this gate is still not a route, bitstream, COM5, or power result.
Failing this gate is a real downstream blocker for this implementation style,
not a failure of the already-completed accuracy/C-sim evidence.
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

from hwnas_fpga.fourstage_selection import canonical_sha256  # noqa: E402
from hwnas_fpga.hardware import parse_hls_report  # noqa: E402
from hwnas_fpga.models.builder import ConvBlock, MBConvBlock, SkipBlock  # noqa: E402
from hwnas_fpga.training.protocol_reporting import sha256_file  # noqa: E402
from run_fourstage_csim_gate import (  # noqa: E402
    CALIBRATION_CONTRACT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_VITIS_HLS,
    INPUT_SIZE,
    OUTPUT_CLASSES,
    CppNetworkEmitter,
    evidence,
    read_json,
    safe_name,
    write_json,
)


DEFAULT_CSIM_SUMMARY = (
    ROOT
    / "artifacts"
    / "sonar_fourstage_operator_v2"
    / "fourstage_csim_zero_mismatch_summary.json"
)
DEFAULT_SUMMARY = (
    ROOT
    / "artifacts"
    / "sonar_fourstage_operator_v2"
    / "fourstage_hls_synthesis_summary.json"
)
TARGET_PART = "xc7k325tffg900-2"
TARGET_CLOCK_NS = 5.0
BRAM18_CAPACITY_BITS = 18 * 1024
AV7K325_BRAM18_EQUIV = 890


class SynthesizableEmitter(CppNetworkEmitter):
    """Build static-buffer HLS C++ from the same layer quantization specs."""

    def __init__(self, model: torch.nn.Module, calibration: dict[str, Any]):
        super().__init__(model, calibration)
        self.synth_ops: list[str] = []
        self.active_buffer = 0
        self.max_tensor_size = self.current_shape.size
        self.max_residual_size = 1
        self._emitted = False

    def inactive_buffer(self) -> int:
        return 1 - self.active_buffer

    def active_name(self) -> str:
        return f"buf{self.active_buffer}"

    def inactive_name(self) -> str:
        return f"buf{self.inactive_buffer()}"

    def swap_buffers(self) -> None:
        self.active_buffer = self.inactive_buffer()

    def _track_current_tensor(self) -> None:
        self.max_tensor_size = max(int(self.max_tensor_size), int(self.current_shape.size))

    def add_conv_synth(
        self,
        *,
        name: str,
        conv: torch.nn.Conv2d,
        bn: torch.nn.BatchNorm2d,
        output_scale: float,
        activation: str | None,
    ) -> None:
        super().add_conv(
            name=name,
            conv=conv,
            bn=bn,
            output_scale=output_scale,
            activation=activation,
        )
        layer = self.conv_layers[-1]
        self.synth_ops.append(
            f"  conv2d_layer({self.active_name()}, {self.inactive_name()}, L_{layer.identifier});"
        )
        self.swap_buffers()
        self._track_current_tensor()

    def add_gap_synth(self, *, output_scale: float) -> None:
        shape = self.current_shape
        self.synth_ops.append(
            f"  avg_pool_global({self.active_name()}, {self.inactive_name()}, "
            f"{shape.channels}, {shape.height}, {shape.width});"
        )
        self.swap_buffers()
        self.current_shape = type(self.current_shape)(shape.channels, 1, 1)
        if not self._same_scale(self.current_scale, output_scale):
            numerator, denominator = self._rescale_ratio(self.current_scale, output_scale) or (1, 1)
            self.synth_ops.append(
                f"  rescale_tensor({self.active_name()}, {self.inactive_name()}, "
                f"{self.current_shape.size}, {int(numerator)}LL, {int(denominator)}LL);"
            )
            self.swap_buffers()
        self.current_scale = float(output_scale)
        self._track_current_tensor()

    def add_linear_synth(
        self, *, name: str, linear: torch.nn.Linear, output_scale: float
    ) -> None:
        super().add_linear(name=name, linear=linear, output_scale=output_scale)
        layer = self.linear_layers[-1]
        self.synth_ops.append(
            f"  linear_layer({self.active_name()}, {self.inactive_name()}, L_{layer.identifier});"
        )
        self.swap_buffers()
        self._track_current_tensor()

    def copy_residual(self) -> tuple[float, int]:
        residual_scale = float(self.current_scale)
        residual_size = int(self.current_shape.size)
        self.max_residual_size = max(self.max_residual_size, residual_size)
        self.synth_ops.append(
            f"  copy_tensor({self.active_name()}, residual, {residual_size});"
        )
        return residual_scale, residual_size

    def add_residual_synth(
        self, *, residual_scale: float, output_scale: float, residual_size: int
    ) -> None:
        ratio = self._rescale_ratio(residual_scale, output_scale)
        if ratio is None:
            self.synth_ops.append(
                f"  residual_add_tensor({self.active_name()}, residual, "
                f"{self.inactive_name()}, {residual_size});"
            )
        else:
            numerator, denominator = ratio
            self.synth_ops.append(
                f"  rescale_tensor(residual, identity, {residual_size}, "
                f"{int(numerator)}LL, {int(denominator)}LL);"
            )
            self.synth_ops.append(
                f"  residual_add_tensor({self.active_name()}, identity, "
                f"{self.inactive_name()}, {residual_size});"
            )
        self.swap_buffers()
        self.current_scale = float(output_scale)
        self._track_current_tensor()

    def emit_synth_ops(self) -> None:
        if self._emitted:
            return
        stem_block = self.model.stem.conv
        if not isinstance(stem_block, ConvBlock):
            raise ValueError("four-stage stem must contain ConvBlock stem.conv")
        self.add_conv_synth(
            name="stem_conv",
            conv=stem_block.conv,
            bn=stem_block.bn,
            output_scale=float(self.scales["stem.conv"]),
            activation="relu",
        )
        for stage_index, stage in enumerate(self.model.stages):
            for block_index, block in enumerate(stage):
                name = f"stages.{stage_index}.{block_index}"
                if isinstance(block, ConvBlock):
                    self.add_conv_synth(
                        name=name,
                        conv=block.conv,
                        bn=block.bn,
                        output_scale=float(self.scales[name]),
                        activation="relu",
                    )
                elif isinstance(block, MBConvBlock):
                    residual_scale = 0.0
                    residual_size = 0
                    if block.use_residual:
                        residual_scale, residual_size = self.copy_residual()
                    if block.use_expand:
                        self.add_conv_synth(
                            name=f"{name}.expand",
                            conv=block.expand_conv,
                            bn=block.expand_bn,
                            output_scale=float(self.scales[f"{name}.expand_relu"]),
                            activation="relu6",
                        )
                    self.add_conv_synth(
                        name=f"{name}.dw",
                        conv=block.dw_conv,
                        bn=block.dw_bn,
                        output_scale=float(self.scales[f"{name}.dw_relu"]),
                        activation="relu6",
                    )
                    self.add_conv_synth(
                        name=f"{name}.project",
                        conv=block.project_conv,
                        bn=block.project_bn,
                        output_scale=float(self.scales[name]),
                        activation=None,
                    )
                    if block.use_residual:
                        self.add_residual_synth(
                            residual_scale=residual_scale,
                            output_scale=float(self.scales[name]),
                            residual_size=residual_size,
                        )
                elif isinstance(block, SkipBlock):
                    if not block.use_conv:
                        continue
                    self.add_conv_synth(
                        name=name,
                        conv=block.conv,
                        bn=block.bn,
                        output_scale=float(self.scales[name]),
                        activation=None,
                    )
                else:  # pragma: no cover
                    raise ValueError(f"unsupported block: {block.__class__.__name__}")
        self.add_gap_synth(output_scale=float(self.scales["head.gap"]))
        self.add_linear_synth(
            name="head.fc",
            linear=self.model.head.fc,
            output_scale=float(self.scales["head.fc"]),
        )
        self._emitted = True

    def preflight(self) -> dict[str, Any]:
        self.emit_synth_ops()
        activation_buffer_bytes = int(self.max_tensor_size) * 2 + int(self.max_residual_size) * 2
        activation_buffer_bits = activation_buffer_bytes * 8
        bram18_lower_bound = int(math.ceil(activation_buffer_bits / BRAM18_CAPACITY_BITS))
        return {
            "implementation_style": "static_full_buffer_hls_probe",
            "max_tensor_elements": int(self.max_tensor_size),
            "max_residual_elements": int(self.max_residual_size),
            "activation_buffer_bytes": int(activation_buffer_bytes),
            "activation_buffer_bits": int(activation_buffer_bits),
            "bram18_lower_bound_for_activation_buffers": bram18_lower_bound,
            "av7k325_bram18_equivalent_capacity": AV7K325_BRAM18_EQUIV,
            "buffer_capacity_warning": bram18_lower_bound > AV7K325_BRAM18_EQUIV,
            "claim_boundary": (
                "Lower bound covers only the generated static activation "
                "buffers, not full routed utilization. A streaming/tiled "
                "datapath may have different memory behavior."
            ),
        }

    def top_cpp(self) -> str:
        self.emit_synth_ops()
        arrays = self._emit_layer_arrays()
        structs = self._emit_layer_structs()
        ops = "\n".join(self.synth_ops)
        return dedent(
            f"""
            #include <climits>
            #include <cstdint>

            using i8 = signed char;
            using i32 = int;
            using i64 = long long;

            static const int MAX_TENSOR_SIZE = {int(self.max_tensor_size)};
            static const int MAX_RESIDUAL_SIZE = {int(self.max_residual_size)};

            struct ConvLayer {{
              int in_c;
              int in_h;
              int in_w;
              int out_c;
              int out_h;
              int out_w;
              int kernel_h;
              int kernel_w;
              int stride;
              int pad_h;
              int pad_w;
              int groups;
              const i8* weights;
              const i32* bias;
              i64 multiplier_numerator;
              i64 multiplier_denominator;
              int activation;
              int activation_upper;
            }};

            struct LinearLayer {{
              int in_features;
              int out_features;
              const i8* weights;
              const i32* bias;
              i64 multiplier_numerator;
              i64 multiplier_denominator;
            }};

            static i64 clamp_i32(i64 value) {{
              if (value < static_cast<i64>(INT32_MIN)) return static_cast<i64>(INT32_MIN);
              if (value > static_cast<i64>(INT32_MAX)) return static_cast<i64>(INT32_MAX);
              return value;
            }}

            static i8 clamp_i8_symmetric(i64 value) {{
              if (value < -127) return static_cast<i8>(-127);
              if (value > 127) return static_cast<i8>(127);
              return static_cast<i8>(value);
            }}

            static i64 round_divide_nearest_even(i64 numerator, i64 denominator) {{
              if (denominator <= 0) return 0;
              const int sign = numerator < 0 ? -1 : 1;
              unsigned long long absolute = numerator < 0
                ? static_cast<unsigned long long>(-numerator)
                : static_cast<unsigned long long>(numerator);
              const unsigned long long divisor = static_cast<unsigned long long>(denominator);
              unsigned long long quotient = absolute / divisor;
              const unsigned long long remainder = absolute % divisor;
              const unsigned long long twice = remainder * 2ULL;
              const bool increment =
                (twice > divisor) || ((twice == divisor) && ((quotient & 1ULL) == 1ULL));
              if (increment) quotient += 1ULL;
              const i64 rounded = static_cast<i64>(quotient);
              return sign < 0 ? -rounded : rounded;
            }}

            static i8 requantize_i8(i64 accumulator, i64 numerator, i64 denominator) {{
              const i64 scaled = round_divide_nearest_even(accumulator * numerator, denominator);
              return clamp_i8_symmetric(scaled);
            }}

            static i8 apply_activation(i8 value, int activation, int activation_upper) {{
              int v = static_cast<int>(value);
              if (activation == 1) {{
                if (v < 0) v = 0;
              }} else if (activation == 2) {{
                if (v < 0) v = 0;
                if (v > activation_upper) v = activation_upper;
              }}
              return static_cast<i8>(v);
            }}

            static int index3(int c, int h, int w, int height, int width) {{
              return (c * height + h) * width + w;
            }}

            static void copy_tensor(const i8* input, i8* output, int size) {{
            copy_loop:
              for (int index = 0; index < size; ++index) {{
            #pragma HLS LOOP_TRIPCOUNT min=1 max={int(self.max_tensor_size)}
                output[index] = input[index];
              }}
            }}

            static void conv2d_layer(const i8* input, i8* output, const ConvLayer& layer) {{
            #pragma HLS INLINE off
            conv_oc:
              for (int oc = 0; oc < layer.out_c; ++oc) {{
            #pragma HLS LOOP_TRIPCOUNT min=8 max=96
              conv_oh:
                for (int oh = 0; oh < layer.out_h; ++oh) {{
            #pragma HLS LOOP_TRIPCOUNT min=1 max=112
                conv_ow:
                  for (int ow = 0; ow < layer.out_w; ++ow) {{
            #pragma HLS LOOP_TRIPCOUNT min=1 max=112
                    const int output_per_group = layer.out_c / layer.groups;
                    const int channels_per_group = layer.in_c / layer.groups;
                    const int group = oc / output_per_group;
                    const int in_start = group * channels_per_group;
                    i64 acc = static_cast<i64>(layer.bias[oc]);
                  conv_ic:
                    for (int ic_local = 0; ic_local < channels_per_group; ++ic_local) {{
            #pragma HLS LOOP_TRIPCOUNT min=1 max=96
                    conv_kh:
                      for (int kh = 0; kh < layer.kernel_h; ++kh) {{
            #pragma HLS LOOP_TRIPCOUNT min=1 max=5
                        const int ih = oh * layer.stride + kh - layer.pad_h;
                        if (ih < 0 || ih >= layer.in_h) continue;
                      conv_kw:
                        for (int kw = 0; kw < layer.kernel_w; ++kw) {{
            #pragma HLS LOOP_TRIPCOUNT min=1 max=5
                          const int iw = ow * layer.stride + kw - layer.pad_w;
                          if (iw < 0 || iw >= layer.in_w) continue;
                          const int ic = in_start + ic_local;
                          const int input_index = index3(ic, ih, iw, layer.in_h, layer.in_w);
                          const int weight_index =
                            ((oc * channels_per_group + ic_local) * layer.kernel_h + kh)
                              * layer.kernel_w + kw;
                          acc += static_cast<i64>(input[input_index])
                            * static_cast<i64>(layer.weights[weight_index]);
                        }}
                      }}
                    }}
                    acc = clamp_i32(acc);
                    i8 value = requantize_i8(
                      acc, layer.multiplier_numerator, layer.multiplier_denominator);
                    value = apply_activation(value, layer.activation, layer.activation_upper);
                    output[index3(oc, oh, ow, layer.out_h, layer.out_w)] = value;
                  }}
                }}
              }}
            }}

            static void rescale_tensor(
                const i8* input,
                i8* output,
                int size,
                i64 numerator,
                i64 denominator) {{
            rescale_loop:
              for (int index = 0; index < size; ++index) {{
            #pragma HLS LOOP_TRIPCOUNT min=1 max={int(self.max_tensor_size)}
                output[index] = requantize_i8(
                  static_cast<i64>(input[index]), numerator, denominator);
              }}
            }}

            static void residual_add_tensor(
                const i8* left,
                const i8* right,
                i8* output,
                int size) {{
            residual_loop:
              for (int index = 0; index < size; ++index) {{
            #pragma HLS LOOP_TRIPCOUNT min=1 max={int(self.max_residual_size)}
                const i64 value =
                  static_cast<i64>(left[index]) + static_cast<i64>(right[index]);
                output[index] = clamp_i8_symmetric(value);
              }}
            }}

            static void avg_pool_global(
                const i8* input,
                i8* output,
                int channels,
                int height,
                int width) {{
            gap_c:
              for (int c = 0; c < channels; ++c) {{
            #pragma HLS LOOP_TRIPCOUNT min=16 max=32
                i64 sum = 0;
              gap_h:
                for (int h = 0; h < height; ++h) {{
            #pragma HLS LOOP_TRIPCOUNT min=28 max=28
                gap_w:
                  for (int w = 0; w < width; ++w) {{
            #pragma HLS LOOP_TRIPCOUNT min=28 max=28
                    sum += static_cast<i64>(input[index3(c, h, w, height, width)]);
                  }}
                }}
                const i64 denominator = static_cast<i64>(height) * static_cast<i64>(width);
                output[c] = clamp_i8_symmetric(round_divide_nearest_even(sum, denominator));
              }}
            }}

            static void linear_layer(const i8* input, i8* output, const LinearLayer& layer) {{
            linear_out:
              for (int out = 0; out < layer.out_features; ++out) {{
            #pragma HLS LOOP_TRIPCOUNT min=8 max=8
                i64 acc = static_cast<i64>(layer.bias[out]);
              linear_in:
                for (int in = 0; in < layer.in_features; ++in) {{
            #pragma HLS LOOP_TRIPCOUNT min=32 max=32
                  const int weight_index = out * layer.in_features + in;
                  acc += static_cast<i64>(input[in]) * static_cast<i64>(layer.weights[weight_index]);
                }}
                acc = clamp_i32(acc);
                output[out] = requantize_i8(
                  acc, layer.multiplier_numerator, layer.multiplier_denominator);
              }}
            }}

            {arrays}

            {structs}

            extern "C" void fourstage_top_hls(const i8 input[{INPUT_SIZE}], i8 output[{OUTPUT_CLASSES}]) {{
            #pragma HLS INTERFACE m_axi port=input depth={INPUT_SIZE} offset=slave bundle=gmem0
            #pragma HLS INTERFACE m_axi port=output depth={OUTPUT_CLASSES} offset=slave bundle=gmem1
            #pragma HLS INTERFACE s_axilite port=input bundle=control
            #pragma HLS INTERFACE s_axilite port=output bundle=control
            #pragma HLS INTERFACE s_axilite port=return bundle=control
              static i8 buf0[MAX_TENSOR_SIZE];
              static i8 buf1[MAX_TENSOR_SIZE];
              static i8 residual[MAX_RESIDUAL_SIZE];
              static i8 identity[MAX_RESIDUAL_SIZE];
              copy_tensor(input, buf0, {INPUT_SIZE});
            {ops}
            output_loop:
              for (int index = 0; index < {OUTPUT_CLASSES}; ++index) {{
                output[index] = {self.active_name()}[index];
              }}
            }}
            """
        ).lstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csim-summary", default=str(DEFAULT_CSIM_SUMMARY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--vitis-hls", default=str(DEFAULT_VITIS_HLS))
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--run-synthesis", action=argparse.BooleanOptionalAction, default=True)
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
        add_files fourstage_top_hls.cpp -cflags {{-std=c++14 -O2}}
        open_solution -reset solution1 -flow_target vivado
        set_part {{{TARGET_PART}}}
        create_clock -period {TARGET_CLOCK_NS:.3f} -name default
        csynth_design
        exit
        """
    ).lstrip()


def run_vitis_synthesis(
    *, vitis_hls: Path, synth_dir: Path, timeout_minutes: int
) -> tuple[int, Path, bool]:
    log_path = synth_dir / "vitis_hls_synthesis.log"
    command = ["cmd", "/c", str(vitis_hls), "-f", "run_synthesis.tcl"]
    process = subprocess.Popen(
        command,
        cwd=synth_dir,
        capture_output=True,
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


def find_csynth_report(synth_dir: Path) -> Path | None:
    candidates = [
        synth_dir / "project" / "solution1" / "syn" / "report" / "fourstage_top_hls_csynth.xml",
        synth_dir / "project" / "solution1" / "syn" / "report" / "fourstage_top_hls_csynth.rpt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    reports = sorted((synth_dir / "project").glob("**/*csynth.xml"))
    if reports:
        return reports[0]
    reports = sorted((synth_dir / "project").glob("**/*csynth.rpt"))
    return reports[0] if reports else None


def run_one_candidate(
    candidate: dict[str, Any],
    *,
    output_root: Path,
    vitis_hls: Path,
    run_synthesis: bool,
    timeout_minutes: int,
) -> dict[str, Any]:
    from hwnas_fpga.deploy.inference import load_checkpoint_model  # noqa: PLC0415

    checkpoint = Path(candidate["source_checkpoint"]["path"]).resolve()
    calibration_path = Path(candidate["activation_calibration"]["path"]).resolve()
    reference_path = Path(candidate["python_int8_reference"]["path"]).resolve()
    if sha256_file(checkpoint) != candidate["source_checkpoint"]["sha256"]:
        raise RuntimeError(f"{candidate['arch_id']}: checkpoint SHA mismatch")
    if sha256_file(calibration_path) != candidate["activation_calibration"]["sha256"]:
        raise RuntimeError(f"{candidate['arch_id']}: calibration SHA mismatch")
    if sha256_file(reference_path) != candidate["python_int8_reference"]["sha256"]:
        raise RuntimeError(f"{candidate['arch_id']}: reference SHA mismatch")

    calibration = read_json(calibration_path)
    model, _architecture, _payload, _class_names = load_checkpoint_model(checkpoint, device="cpu")
    emitter = SynthesizableEmitter(model, calibration)
    preflight = emitter.preflight()

    synth_dir = (
        output_root
        / f"{safe_name(candidate['role'])}__{safe_name(candidate['arch_id'])}"
        / "fold0_seed42"
        / "hls_synthesis"
    )
    synth_dir.mkdir(parents=True, exist_ok=True)
    top_path = synth_dir / "fourstage_top_hls.cpp"
    tcl_path = synth_dir / "run_synthesis.tcl"
    top_path.write_text(emitter.top_cpp(), encoding="utf-8")
    tcl_path.write_text(tcl_script(), encoding="utf-8")

    generated = {
        "top": evidence(top_path),
        "tcl": evidence(tcl_path),
    }
    if not run_synthesis:
        return {
            "role": candidate["role"],
            "arch_id": candidate["arch_id"],
            "status": "NOT_RUN",
            "synth_project_dir": str(synth_dir.resolve()),
            "generated": generated,
            "preflight": preflight,
            "vitis_returncode": None,
            "vitis_log": None,
            "csynth_report": None,
            "metrics": None,
            "claim_boundary": "HLS source generated only; synthesis was not run.",
        }

    returncode, log_path, timed_out = run_vitis_synthesis(
        vitis_hls=vitis_hls,
        synth_dir=synth_dir,
        timeout_minutes=timeout_minutes,
    )

    report_path = find_csynth_report(synth_dir)
    metrics = None
    if report_path is not None and report_path.is_file():
        try:
            metrics = parse_hls_report(report_path)
        except Exception as exc:  # pragma: no cover - evidence path still recorded.
            metrics = {"parse_error": repr(exc)}
    status = "PASS" if returncode == 0 and report_path is not None else "FAIL"
    if timed_out:
        status = "TIMEOUT"
    return {
        "role": candidate["role"],
        "arch_id": candidate["arch_id"],
        "status": status,
        "synth_project_dir": str(synth_dir.resolve()),
        "generated": generated,
        "preflight": preflight,
        "vitis_returncode": int(returncode),
        "timeout_minutes": int(timeout_minutes),
        "failure_category": (
            "STATIC_FULL_BUFFER_HLS_SYNTHESIS_TIMEOUT"
            if status == "TIMEOUT"
            else ("HLS_SYNTHESIS_NO_REPORT" if status == "FAIL" else None)
        ),
        "vitis_log": evidence(log_path),
        "csynth_report": evidence(report_path) if report_path is not None else None,
        "metrics": metrics,
        "claim_boundary": (
            "Vitis HLS csynth evidence only. This is not RTL co-sim, "
            "place-and-route, bitstream, COM5, or measured power."
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
        row
        for row in int8_summary["candidates"]
        if str(row["arch_id"]) in csim_passed_ids
    ]
    if len(candidates) != len(int8_summary["candidates"]):
        raise RuntimeError("not all INT8 candidates have matching C-sim PASS evidence")
    formal_candidate_count = len(candidates)
    if args.candidate_limit is not None:
        candidates = candidates[: int(args.candidate_limit)]
    if args.run_synthesis and not vitis_hls.is_file():
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "BLOCKED",
            "gate": "full_network_hls_synthesis",
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
            run_synthesis=bool(args.run_synthesis),
            timeout_minutes=int(args.timeout_minutes),
        )
        for candidate in candidates
    ]
    formal_scope = args.candidate_limit is None and len(rows) == formal_candidate_count
    if not bool(args.run_synthesis):
        status = "NOT_RUN"
    elif formal_scope and rows and all(row["status"] == "PASS" for row in rows):
        status = "PASS"
    elif rows and all(row["status"] == "PASS" for row in rows):
        status = "PARTIAL_PASS_NOT_FORMAL"
    elif any(row["status"] == "TIMEOUT" for row in rows):
        status = "TIMEOUT"
    else:
        status = "FAIL"
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "gate": "full_network_hls_synthesis",
        "contract": CALIBRATION_CONTRACT,
        "implementation_style": "static_full_buffer_hls_probe",
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
            "place_and_route_5ns": "PENDING" if status == "PASS" else "NOT_RUN_HLS_SYNTHESIS_NOT_PASSED",
            "bitstream": "NOT_GENERATED",
            "com5_board_latency": "NOT_RUN",
            "external_meter_power": "NOT_MEASURED",
        },
        "claim_boundary": (
            "This is full-network Vitis HLS synthesis evidence for a "
            "static-buffer probe implementation. It is distinct from C-sim, "
            "RTL co-sim, full route, bitstream, COM5 board execution, and "
            "external-meter power. A failure may require a streaming/tiled "
            "datapath rather than changing the frozen architecture."
        ),
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    write_json(summary_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if status in {"PASS", "PARTIAL_PASS_NOT_FORMAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
