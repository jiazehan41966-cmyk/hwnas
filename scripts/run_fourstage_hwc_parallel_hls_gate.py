#!/usr/bin/env python3
"""Run a co-sim-oriented full-network HLS gate for four-stage sonar candidates.

The previous ``external_scratch_pixel_tiled_v1`` design is full-network and
bit-exact at C-sim/HLS synthesis, but its RTL co-simulation timed out because
the generated datapath is essentially one-MAC-at-a-time and AXI-scratch serial.

This gate keeps the frozen four-stage architecture, real checkpoints, activation
calibration, INT8 arithmetic contract, and full-network sample set.  The only
implementation change is the HLS datapath:

* internal activations are stored in HWC order so channels are contiguous;
* local scratch buffers are cyclically banked across the channel-parallel lane;
* convolution output/depthwise channels are computed in parallel.

It is still not a board result by itself.  Route and bitstream may only be
claimed after RTL co-sim and Vivado implementation pass their own gates.
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
    CppNetworkEmitter,
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


ARTIFACT_ROOT = ROOT / "artifacts" / "sonar_fourstage_operator_v2"
DEFAULT_SUMMARY = ARTIFACT_ROOT / "fourstage_hwc_parallel_hls_summary.json"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "sonar_fourstage_operator_v2" / "hwcpar32"
IMPLEMENTATION_STYLE = "hwc_channel_parallel_v1"


def _run_process(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_minutes: int,
) -> tuple[int, Path, bool]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
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


class HwcParallelEmitter(SynthesizableEmitter):
    """Emit a local-banked HWC-layout HLS datapath for the frozen network."""

    def __init__(
        self,
        model: torch.nn.Module,
        calibration: dict[str, Any],
        *,
        channel_parallel: int,
    ):
        super().__init__(model, calibration)
        self.channel_parallel = int(channel_parallel)

    def add_conv_synth(
        self,
        *,
        name: str,
        conv: torch.nn.Conv2d,
        bn: torch.nn.BatchNorm2d,
        output_scale: float,
        activation: str | None,
    ) -> None:
        CppNetworkEmitter.add_conv(
            self,
            name=name,
            conv=conv,
            bn=bn,
            output_scale=output_scale,
            activation=activation,
        )
        layer = self.conv_layers[-1]
        if layer.groups == layer.input_shape.channels and layer.groups == layer.output_shape.channels:
            fn_name = "conv2d_depthwise_hwc"
        elif layer.groups == 1:
            fn_name = "conv2d_standard_hwc"
        else:
            raise ValueError(f"{name}: grouped convolution other than depthwise is unsupported")
        self.synth_ops.append(
            f"  {fn_name}({self.active_name()}, {self.inactive_name()}, L_{layer.identifier});"
        )
        self.swap_buffers()
        self._track_current_tensor()

    def add_gap_synth(self, *, output_scale: float) -> None:
        shape = self.current_shape
        self.synth_ops.append(
            f"  avg_pool_global_hwc({self.active_name()}, {self.inactive_name()}, "
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
        CppNetworkEmitter.add_linear(
            self,
            name=name,
            linear=linear,
            output_scale=output_scale,
        )
        layer = self.linear_layers[-1]
        self.synth_ops.append(
            f"  linear_layer({self.active_name()}, {self.inactive_name()}, L_{layer.identifier});"
        )
        self.swap_buffers()
        self._track_current_tensor()

    def preflight(self) -> dict[str, Any]:
        self.emit_synth_ops()
        bank_factor = int(self.channel_parallel)
        activation_buffer_bytes = (
            int(self.max_tensor_size) * 2
            + int(self.max_residual_size) * 2
            + int(self.max_residual_size)
        )
        banked_bits = int(math.ceil(self.max_tensor_size / bank_factor)) * 8
        banked_residual_bits = int(math.ceil(self.max_residual_size / bank_factor)) * 8
        activation_bram18_estimate = (
            2 * bank_factor * int(math.ceil(banked_bits / BRAM18_CAPACITY_BITS))
            + 2
            * bank_factor
            * int(math.ceil(banked_residual_bits / BRAM18_CAPACITY_BITS))
            + int(math.ceil((self.max_residual_size * 8) / BRAM18_CAPACITY_BITS))
        )
        return {
            "implementation_style": IMPLEMENTATION_STYLE,
            "layout": "internal_hwc_channel_contiguous",
            "channel_parallel": bank_factor,
            "max_tensor_elements": int(self.max_tensor_size),
            "max_residual_elements": int(self.max_residual_size),
            "local_activation_buffer_bytes": int(activation_buffer_bytes),
            "banked_activation_bram18_estimate": int(activation_bram18_estimate),
            "av7k325_bram18_equivalent_capacity": AV7K325_BRAM18_EQUIV,
            "activation_bram_capacity_warning": (
                int(activation_bram18_estimate) > AV7K325_BRAM18_EQUIV
            ),
            "claim_boundary": (
                "Full-network HLS implementation with local banked activation "
                "scratch. Passing HLS still does not imply RTL co-sim, route, "
                "bitstream, COM5, or measured power."
            ),
        }

    def _weight_partition_pragmas(self) -> str:
        lines: list[str] = []
        for layer in self.conv_layers:
            lines.append(f"#pragma HLS ARRAY_PARTITION variable=W_{layer.identifier} complete dim=1")
            lines.append(f"#pragma HLS ARRAY_PARTITION variable=B_{layer.identifier} complete dim=1")
        for layer in self.linear_layers:
            lines.append(f"#pragma HLS ARRAY_PARTITION variable=W_{layer.identifier} complete dim=1")
            lines.append(f"#pragma HLS ARRAY_PARTITION variable=B_{layer.identifier} complete dim=1")
        return "\n".join("  " + line for line in lines)

    def top_cpp(self) -> str:
        self.emit_synth_ops()
        arrays = self._emit_layer_arrays()
        structs = self._emit_layer_structs()
        ops = "\n".join(self.synth_ops)
        weight_pragmas = self._weight_partition_pragmas()
        channel_parallel = int(self.channel_parallel)
        return dedent(
            f"""
            #include <climits>
            #include <cstdint>

            using i8 = signed char;
            using i32 = int;
            using i64 = long long;

            static const int MAX_TENSOR_SIZE = {int(self.max_tensor_size)};
            static const int MAX_RESIDUAL_SIZE = {int(self.max_residual_size)};
            static const int CHANNEL_PARALLEL = {channel_parallel};

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

            static int index_hwc(int c, int h, int w, int height, int width, int channels) {{
              return (h * width + w) * channels + c;
            }}

            static void copy_tensor(const i8* input, i8* output, int size) {{
            copy_loop:
              for (int index = 0; index < size; ++index) {{
            #pragma HLS LOOP_TRIPCOUNT min=1 max={int(self.max_tensor_size)}
            #pragma HLS PIPELINE II=1
                output[index] = input[index];
              }}
            }}

            static void conv2d_standard_hwc(
                const i8* input,
                i8* output,
                const ConvLayer& layer) {{
            #pragma HLS INLINE off
            standard_oh:
              for (int oh = 0; oh < layer.out_h; ++oh) {{
            #pragma HLS LOOP_TRIPCOUNT min=1 max=112
              standard_ow:
                for (int ow = 0; ow < layer.out_w; ++ow) {{
            #pragma HLS LOOP_TRIPCOUNT min=1 max=112
                standard_oc_tile:
                  for (int oc_base = 0; oc_base < layer.out_c; oc_base += CHANNEL_PARALLEL) {{
            #pragma HLS LOOP_TRIPCOUNT min=1 max=3
                    i64 acc[CHANNEL_PARALLEL];
            #pragma HLS ARRAY_PARTITION variable=acc complete dim=1
                  standard_init:
                    for (int lane = 0; lane < CHANNEL_PARALLEL; ++lane) {{
            #pragma HLS UNROLL
                      const int oc = oc_base + lane;
                      acc[lane] = oc < layer.out_c ? static_cast<i64>(layer.bias[oc]) : 0;
                    }}
                  standard_ic:
                    for (int ic = 0; ic < layer.in_c; ++ic) {{
            #pragma HLS LOOP_TRIPCOUNT min=1 max=96
                    standard_kh:
                      for (int kh = 0; kh < layer.kernel_h; ++kh) {{
            #pragma HLS LOOP_TRIPCOUNT min=1 max=5
                        const int ih = oh * layer.stride + kh - layer.pad_h;
                        if (ih < 0 || ih >= layer.in_h) continue;
                      standard_kw:
                        for (int kw = 0; kw < layer.kernel_w; ++kw) {{
            #pragma HLS LOOP_TRIPCOUNT min=1 max=5
            #pragma HLS PIPELINE II=1
                          const int iw = ow * layer.stride + kw - layer.pad_w;
                          if (iw < 0 || iw >= layer.in_w) continue;
                          const i8 in_value = input[index_hwc(
                            ic, ih, iw, layer.in_h, layer.in_w, layer.in_c)];
                        standard_lane:
                          for (int lane = 0; lane < CHANNEL_PARALLEL; ++lane) {{
            #pragma HLS UNROLL
                            const int oc = oc_base + lane;
                            if (oc < layer.out_c) {{
                              const int weight_index =
                                ((oc * layer.in_c + ic) * layer.kernel_h + kh)
                                  * layer.kernel_w + kw;
                              acc[lane] += static_cast<i64>(in_value)
                                * static_cast<i64>(layer.weights[weight_index]);
                            }}
                          }}
                        }}
                      }}
                    }}
                  standard_store:
                    for (int lane = 0; lane < CHANNEL_PARALLEL; ++lane) {{
            #pragma HLS UNROLL
                      const int oc = oc_base + lane;
                      if (oc < layer.out_c) {{
                        i64 clamped = clamp_i32(acc[lane]);
                        i8 value = requantize_i8(
                          clamped, layer.multiplier_numerator, layer.multiplier_denominator);
                        value = apply_activation(value, layer.activation, layer.activation_upper);
                        output[index_hwc(oc, oh, ow, layer.out_h, layer.out_w, layer.out_c)] =
                          value;
                      }}
                    }}
                  }}
                }}
              }}
            }}

            static void conv2d_depthwise_hwc(
                const i8* input,
                i8* output,
                const ConvLayer& layer) {{
            #pragma HLS INLINE off
            depthwise_oh:
              for (int oh = 0; oh < layer.out_h; ++oh) {{
            #pragma HLS LOOP_TRIPCOUNT min=1 max=56
              depthwise_ow:
                for (int ow = 0; ow < layer.out_w; ++ow) {{
            #pragma HLS LOOP_TRIPCOUNT min=1 max=56
                depthwise_c_tile:
                  for (int c_base = 0; c_base < layer.out_c; c_base += CHANNEL_PARALLEL) {{
            #pragma HLS LOOP_TRIPCOUNT min=1 max=3
                    i64 acc[CHANNEL_PARALLEL];
            #pragma HLS ARRAY_PARTITION variable=acc complete dim=1
                  depthwise_init:
                    for (int lane = 0; lane < CHANNEL_PARALLEL; ++lane) {{
            #pragma HLS UNROLL
                      const int c = c_base + lane;
                      acc[lane] = c < layer.out_c ? static_cast<i64>(layer.bias[c]) : 0;
                    }}
                  depthwise_kh:
                    for (int kh = 0; kh < layer.kernel_h; ++kh) {{
            #pragma HLS LOOP_TRIPCOUNT min=1 max=5
                      const int ih = oh * layer.stride + kh - layer.pad_h;
                      if (ih < 0 || ih >= layer.in_h) continue;
                    depthwise_kw:
                      for (int kw = 0; kw < layer.kernel_w; ++kw) {{
            #pragma HLS LOOP_TRIPCOUNT min=1 max=5
            #pragma HLS PIPELINE II=1
                        const int iw = ow * layer.stride + kw - layer.pad_w;
                        if (iw < 0 || iw >= layer.in_w) continue;
                      depthwise_lane:
                        for (int lane = 0; lane < CHANNEL_PARALLEL; ++lane) {{
            #pragma HLS UNROLL
                          const int c = c_base + lane;
                          if (c < layer.out_c) {{
                            const int input_index = index_hwc(
                              c, ih, iw, layer.in_h, layer.in_w, layer.in_c);
                            const int weight_index =
                              (c * layer.kernel_h + kh) * layer.kernel_w + kw;
                            acc[lane] += static_cast<i64>(input[input_index])
                              * static_cast<i64>(layer.weights[weight_index]);
                          }}
                        }}
                      }}
                    }}
                  depthwise_store:
                    for (int lane = 0; lane < CHANNEL_PARALLEL; ++lane) {{
            #pragma HLS UNROLL
                      const int c = c_base + lane;
                      if (c < layer.out_c) {{
                        i64 clamped = clamp_i32(acc[lane]);
                        i8 value = requantize_i8(
                          clamped, layer.multiplier_numerator, layer.multiplier_denominator);
                        value = apply_activation(value, layer.activation, layer.activation_upper);
                        output[index_hwc(c, oh, ow, layer.out_h, layer.out_w, layer.out_c)] =
                          value;
                      }}
                    }}
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
            #pragma HLS PIPELINE II=1
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
            #pragma HLS PIPELINE II=1
                const i64 value =
                  static_cast<i64>(left[index]) + static_cast<i64>(right[index]);
                output[index] = clamp_i8_symmetric(value);
              }}
            }}

            static void avg_pool_global_hwc(
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
            #pragma HLS PIPELINE II=1
                    sum += static_cast<i64>(input[index_hwc(c, h, w, height, width, channels)]);
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
            #pragma HLS PIPELINE II=1
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
            #pragma HLS BIND_STORAGE variable=buf0 type=ram_t2p impl=bram
            #pragma HLS BIND_STORAGE variable=buf1 type=ram_t2p impl=bram
            #pragma HLS BIND_STORAGE variable=residual type=ram_t2p impl=bram
            #pragma HLS BIND_STORAGE variable=identity type=ram_t2p impl=bram
            #pragma HLS ARRAY_PARTITION variable=buf0 cyclic factor={channel_parallel} dim=1
            #pragma HLS ARRAY_PARTITION variable=buf1 cyclic factor={channel_parallel} dim=1
            #pragma HLS ARRAY_PARTITION variable=residual cyclic factor={channel_parallel} dim=1
            #pragma HLS ARRAY_PARTITION variable=identity cyclic factor={channel_parallel} dim=1
            {weight_pragmas}
              copy_tensor(input, buf0, {INPUT_SIZE});
            {ops}
            output_loop:
              for (int index = 0; index < {OUTPUT_CLASSES}; ++index) {{
            #pragma HLS PIPELINE II=1
                output[index] = {self.active_name()}[index];
              }}
            }}
            """
        ).lstrip()


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


def testbench_cpp(
    *,
    input_samples: list[list[int]],
    expected_outputs: list[list[int]],
    result_path: Path,
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

        extern "C" void fourstage_top_hls(
            const i8 input[{INPUT_SIZE}],
            i8 output[{OUTPUT_CLASSES}]);

        static const i8 SAMPLE_INPUTS[{sample_count}][{INPUT_SIZE}] = {{
        {samples}
        }};

        static const i8 EXPECTED[{sample_count}][{OUTPUT_CLASSES}] = {{
        {expected}
        }};

        int main() {{
          const char* result_path = "{cpp_string(result_path)}";
          int mismatch_count = 0;
          int checked_outputs = 0;
          int max_abs_mismatch = 0;
          for (int sample = 0; sample < {sample_count}; ++sample) {{
            i8 output[{OUTPUT_CLASSES}] = {{0}};
            fourstage_top_hls(SAMPLE_INPUTS[sample], output);
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


def run_vitis_hls(
    *, vitis_hls: Path, hls_dir: Path, timeout_minutes: int
) -> tuple[int, Path, bool]:
    log_path = hls_dir / "vitis_hls_hwc_parallel.log"
    command = ["cmd", "/c", str(vitis_hls), "-f", "run_synthesis.tcl"]
    return _run_process(
        command,
        cwd=hls_dir,
        log_path=log_path,
        timeout_minutes=timeout_minutes,
    )


def classify_failure(
    *,
    status: str,
    timed_out: bool,
    csim_pass: bool,
    returncode: int,
    report_path: Path | None,
) -> str | None:
    if status == "PASS":
        return None
    if timed_out:
        return "HWC_PARALLEL_HLS_TIMEOUT"
    if not csim_pass:
        return "HWC_PARALLEL_CSIM_FAILURE"
    if report_path is None:
        return "HWC_PARALLEL_HLS_SYNTHESIS_NO_REPORT"
    if returncode != 0:
        return "HWC_PARALLEL_HLS_RETURNCODE_NONZERO_WITH_REPORT"
    return "HWC_PARALLEL_HLS_FAILURE"


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_candidates(csim_summary: dict[str, Any], int8_summary: dict[str, Any]) -> list[dict[str, Any]]:
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
    return candidates


def run_one_candidate(
    candidate: dict[str, Any],
    *,
    output_root: Path,
    vitis_hls: Path,
    run_hls: bool,
    timeout_minutes: int,
    int8_summary: dict[str, Any],
    channel_parallel: int,
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
    emitter = HwcParallelEmitter(
        model,
        calibration,
        channel_parallel=channel_parallel,
    )
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

    hls_dir = output_root / safe_name(candidate["role"])
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
            "HWC channel-parallel full-network HLS C-sim/synthesis evidence only. "
            "This is not RTL co-sim, place-and-route, bitstream, COM5, or measured power."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csim-summary", default=str(DEFAULT_CSIM_SUMMARY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--vitis-hls", default=str(DEFAULT_VITIS_HLS))
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument(
        "--role",
        action="append",
        default=[],
        help="Candidate role to include; repeatable. Defaults to all candidates.",
    )
    parser.add_argument("--channel-parallel", type=int, default=32)
    parser.add_argument("--run-hls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-minutes", type=int, default=120)
    return parser.parse_args()


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
    candidates = load_candidates(csim_summary, int8_summary)
    formal_candidate_count = len(candidates)
    roles = {str(role) for role in args.role}
    if roles:
        candidates = [row for row in candidates if str(row.get("role")) in roles]
    if args.candidate_limit is not None:
        candidates = candidates[: int(args.candidate_limit)]
    if not candidates:
        raise RuntimeError("no candidates selected for HWC parallel HLS gate")

    if args.run_hls and not vitis_hls.is_file():
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "BLOCKED",
            "gate": "full_network_hwc_parallel_hls",
            "implementation_style": IMPLEMENTATION_STYLE,
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

    results = []
    for candidate in candidates:
        results.append(
            run_one_candidate(
                candidate,
                output_root=output_root,
                vitis_hls=vitis_hls,
                run_hls=bool(args.run_hls),
                timeout_minutes=int(args.timeout_minutes),
                int8_summary=int8_summary,
                channel_parallel=int(args.channel_parallel),
            )
        )

    if not bool(args.run_hls):
        status = "NOT_RUN"
    elif all(row["status"] == "PASS" for row in results):
        status = "PASS"
    elif any(row["status"] == "TIMEOUT" for row in results):
        status = "TIMEOUT"
    else:
        status = "FAIL"

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "gate": "full_network_hwc_parallel_hls",
        "implementation_style": IMPLEMENTATION_STYLE,
        "target_part": TARGET_PART,
        "target_clock_ns": TARGET_CLOCK_NS,
        "channel_parallel": int(args.channel_parallel),
        "csim_zero_mismatch_summary": evidence(csim_summary_path),
        "int8_reference_summary": evidence(int8_summary_path),
        "vitis_hls": {"path": str(vitis_hls), "exists": vitis_hls.is_file()},
        "git": {
            "branch": git_output("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": git_output("rev-parse", "HEAD"),
            "dirty": bool(git_output("status", "--short")),
        },
        "candidate_count": len(results),
        "formal_candidate_count": formal_candidate_count,
        "selected_roles": [row.get("role") for row in candidates],
        "candidates": results,
        "downstream_gates": {
            "rtl_cosim": "PENDING" if status == "PASS" else "NOT_RUN_HLS_NOT_PASSED",
            "place_and_route_5ns": "NOT_RUN_RTL_COSIM_NOT_PASSED",
            "bitstream": "NOT_GENERATED",
            "com5_board_latency": "NOT_RUN",
            "external_meter_power": "NOT_MEASURED",
        },
        "claim_boundary": (
            "This summary records full-network HWC channel-parallel HLS "
            "C-sim/synthesis only. It must not be used as RTL co-sim, full "
            "route, bitstream, COM5, or measured-power evidence."
        ),
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    write_json(summary_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if status in {"PASS", "NOT_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
