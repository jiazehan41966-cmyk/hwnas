# Kernel Specification

This document defines the formal kernel contract for the MobileNetV2-derived
operator screening flow in `hls_lut_builder`.

## Goal

Each candidate operator is materialized as an independent HLS kernel under one
shared external contract so that synthesis and downstream Vivado results are
comparable across operators.

## Formal Contract

The current formal contract is `packed_stream_v1`.

- feature input: packed `AXI-Stream`, width = `packed_input_channels * 8`
- feature output: packed `AXI-Stream`, width = `packed_output_channels * 8`
- control: `AXI-Lite`
- parameter storage: `BRAM`
- activation type: `ap_int<8>`
- weight type: `ap_int<8>`
- accumulator type: `ap_int<32>`
- target clock: `5 ns` (`200 MHz`)

All operator templates must include [common_defs.h](/E:/1/hwnas/hwnas/hls_lut_builder/include/common_defs.h).
[common_types.h](/E:/1/hwnas/hwnas/hls_lut_builder/include/common_types.h) is
deprecated as a direct include for operator kernels.

## Interface Pattern

Each kernel derives its stream widths from the operator shape. When a single
channel vector would exceed the Vitis HLS AXI payload limit, the kernel keeps
the same packed-stream semantics but splits that vector across multiple
consecutive input tokens.

```cpp
typedef MAKE_AXIS_TYPE(IN_CH * 8) input_axis_t;
typedef MAKE_AXIS_TYPE(OUT_CH * 8) output_axis_t;

void operator_kernel(
    hls::stream<input_axis_t> &input_stream,
    hls::stream<output_axis_t> &output_stream,
    const weight_t weights[...],
    const acc_t biases[...]
);
```

Required interface declarations:

```cpp
DECLARE_AXIS_INTERFACE();
DECLARE_BRAM_WEIGHTS();
DECLARE_CONTROL_INTERFACE();
```

## Datapath Policy

- Primitive kernels should avoid whole-feature-map input/output buffers.
- Packed-stream kernels should begin consuming tokens from `input_stream`
  immediately and emit `output_stream` tokens as soon as the operator semantics
  allow.
- Channel-sensitive local state uses `ARRAY_PARTITION`.
- `PIPELINE II=1` remains the baseline loop target.
- Composite kernels may use internal staging or `DATAFLOW` where needed, but
  the external interface contract must stay unchanged.

## Latency Scope

`latency` is defined from:

1. the first valid token entering `input_stream`
2. to the last valid token leaving `output_stream`

The measurement includes only the operator body and excludes host launch
overhead, AXI interconnect glue, DMA, and board-level transport.

The authoritative latency wording lives in
[LATENCY_SCOPE.md](/E:/1/hwnas/hwnas/hls_lut_builder/LATENCY_SCOPE.md); this
document is expected to stay aligned with it.

## Acceptance

The kernel-packaging stage is complete when:

1. all 11 core operator kernels render from the config
2. every core operator template passes the packed-stream contract test
3. each core operator runs at least one HLS synthesis case without crashing
4. `kernel_specification.md` and `LATENCY_SCOPE.md` state the same packed-stream
   measurement policy
