# Latency Scope

This document defines the measurement contract for the formal operator-level
LUT build in `hls_lut_builder`. The current formal contract id is
`packed_stream_v1`.

## Operator Coverage

Core kernels:

- `stem_conv_k3_s2`
- `dw_conv_k3`
- `dw_conv_k5`
- `pw_conv`
- `mbconv_e3_k3`
- `mbconv_e3_k5`
- `mbconv_e6_k3`
- `mbconv_e6_k5`
- `skip`
- `global_avg_pool`
- `fc_layer`

Extension kernels are disabled by default until v1 screening confirms them:

- `fused_mbconv_e3_k3`
- `fused_mbconv_e6_k3`
- `mixconv`
- `denoise`
- `edge`

Excluded from this LUT round:

- `7x7` and larger kernels
- `SE`
- `Swish/GELU`
- dilated convolution

## Unified Interface

The formal LUT contract is now a packed-stream contract:

- feature input: `AXI-Stream`, width = `packed_input_channels * 8` bits
- feature output: `AXI-Stream`, width = `packed_output_channels * 8` bits
- control: `AXI-Lite`
- parameters: `BRAM`

When a full channel vector would exceed the Vitis HLS AXI payload limit, the
same packed-stream rule is applied across multiple consecutive tokens for that
spatial position instead of forcing one oversized packet.

In code, the stream element type is derived per kernel rather than fixed to a
single global `axis_t` width:

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

Interface pragmas remain aligned across kernels:

- `#pragma HLS INTERFACE axis port=input_stream`
- `#pragma HLS INTERFACE axis port=output_stream`
- `#pragma HLS INTERFACE bram port=weights`
- `#pragma HLS INTERFACE bram port=biases`
- `#pragma HLS INTERFACE s_axilite port=return bundle=control`

## Unified Datapath And Pragma Rules

- activation and weight type: `ap_int<8>`
- accumulator type: `ap_int<32>`
- clock target: `5 ns` (`200 MHz`)
- baseline loop pragma: `#pragma HLS PIPELINE II=1`
- baseline storage pragma: `ARRAY_PARTITION` on channel-sensitive local buffers
- composite kernels may use internal staging or `DATAFLOW`, but the external
  packed-stream contract must remain unchanged

These rules are meant to keep operator-to-operator comparisons meaningful. The
packed-stream contract has been validated on the `stem_conv_k3_s2` pilot and is
the intended LUT contract for the broader operator migration. If a future
experiment needs a different interface or pragma policy, it should live in a
separate LUT config instead of mixing with this one.

## Latency Definition

`latency` is measured as:

- the cycle interval starting from the first valid token entering
  `input_stream`
- and ending at the last valid token leaving `output_stream`

The measurement includes the full operator body:

- `stem_conv_k3_s2`: convolution + bias + activation
- `dw_conv_*`: depthwise convolution + bias + activation
- `pw_conv`: pointwise convolution, with activation controlled by the case
- `mbconv_*`: expand + depthwise + project, plus residual add when enabled
- `fused_mbconv_*`: fused spatial convolution + project, plus residual add when
  enabled
- `mixconv`: `3x3 + 5x5` branch mixing + pointwise projection
- `denoise`: smoothing branch + projection fusion
- `edge`: edge branch + projection fusion
- `global_avg_pool`: pooling body only
- `fc_layer`: dense projection only

The measurement excludes:

- host-side launch overhead
- AXI interconnect glue logic
- DMA and board-level transport logic

## Resource Scope

The LUT records only the kernel-local HLS report for:

- `LUT`
- `FF`
- `BRAM_18K`
- `BRAM_36K`
- `DSP`
- `Latency`
- `II`
- `Estimated Fmax`

It does not claim to equal final board-level implementation results. Full
system-level validation and calibration still need a later integration stage.
