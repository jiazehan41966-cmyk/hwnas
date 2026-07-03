# Fixed-Vector Primitive HLS v2

This document records the v2 operator-template contract for the fixed-vector
Primitive HLS refactor.

## Parameter Contract

- `PACK_CH` is the physical int8 channel count carried by one AXI-Stream token.
  The physical stream width is `PACK_CH * bitwidth`.
- `CH_BLOCK` is the tile-local channel window used by depthwise and sonar
  spatial kernels.
- Fixed-vector profiles must satisfy `CH_BLOCK = k * PACK_CH`, where
  `k in {1, 2, 4}`.
- YAML exposes only combined implementation profiles such as
  `p16_cb16_ii2_dsp1`; `PACK_CH` and `CH_BLOCK` are not swept independently.
- LUT identity keys include `pack_ch`, `ch_block`, `tile_order`, `target_ii`,
  `target_clock_mhz`, and `dsp_pack`.

## Depthwise v2 Schedule

The first depthwise v2 implementation uses
`tile_order=channel_major_replay`.

- The kernel processes one `CH_BLOCK` at a time.
- The tile-local line buffer is `line_buffer[CH_BLOCK][K][PADDED_W]`.
- Input pixels are replayed once per channel block.
- Output order is channel-block-major.

This schedule intentionally trades input replay latency for lower local state.
Any future `spatial_major_stateful` implementation must use a different
`tile_order` value and therefore a different LUT key.

## Measurement Policy

- The primary deployment target remains `5 ns` / `200 MHz`.
- `II in {1, 2, 4}` may be recorded.
- `6.667 ns` / `150 MHz` rows are recorded as relaxed measurements and must not
  be treated as deployable at 200 MHz.
- Power is only valid when parsed from implementation-stage reports. HLS-only
  rows leave `power_w` blank and use `power_source=csynth_not_recorded`.

## Fused MBConv Policy

`fused_mbconv` stays disabled in v2. If a search run needs a fused cost before a
new fused template exists, use `pw_expand + dw + pw_project` as a conservative
upper bound. This estimate does not include BRAM sharing or stream-forwarding
benefits from a true fused implementation.

## Initial Validation Set

- `dw_conv_k3`, `C=96`
- `dw_conv_k5`, `C=48`
- `identity_repack`, `PACK_CH in {16, 32, 64}`

Results for this round live under `hls_lut_builder/results/v2_fixed_vector/`.

## Failed44 HLS Rerun

The previous current implementation had 44 `defer_current_impl` cases. The v2
fixed-vector rerun maps those cases to primitive measurements rather than using
the disabled fused MBConv template.

- Old failed combinations: 44
- Unique v2 primitive HLS cases: 64
- Primitive HLS result: 64/64 `success`
- Aggregated old-combo result: 44/44 `primitive_hls_success`

Observed II by primitive family:

- `dw_conv_k3`: target II=1, observed II=1
- `dw_conv_k5`: target II=2, observed II=2
- `pw_conv`: target II=1, observed II=8

The `pw_conv` rows are HLS-feasible but target-II missed. They must be treated
as `hls_success_target_ii_missed` and excluded from the NAS deployable LUT path
until a pointwise v2.1 template pass produces acceptable measured profiles. This
also excludes decomposed MBConv rows that depend on `pw_expand` or `pw_project`.

The Failed44 rerun results live under
`hls_lut_builder/results/v2_failed44_fixed_vector/`. The generated formal LUT in
that directory is an HLS-only feasibility record and must not be used as the
official deployable NAS LUT. A row may enter the deployable LUT only after Vivado
downstream succeeds and records complete post-route resource, timing, and power
source metadata.
