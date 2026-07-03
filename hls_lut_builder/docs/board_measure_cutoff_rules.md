# Board Measure Cutoff Rules

This document defines when a LUT case is too heavy to continue board measurement under the **current** implementation stack:

- `packed_stream_v1` operator interface
- current HLS templates in `hls_lut_builder/templates`
- current board harness flow in `hls_lut_builder/board_harness`
- Vitis HLS 2023.2 / Vivado 2023.2 on the current host

These rules are **implementation-specific**. They do **not** mean the operator is theoretically impossible on FPGA. They mean the current template/tool flow is no longer yielding board-measurement data at a reasonable engineering cost.

Interpretation:

- This document is an **empirical cutoff policy under the current implementation**.
- It is not a theorem, not a theoretical FPGA capacity bound, and not a
  hardware-independent statement about the operator families.
- The rules are inferred from observed success / failure frontiers in the
  current measured LUT sweep and clean-serial retries.
- If templates, packing rules, storage binding, or tool versions change, these
  cutoffs must be revalidated rather than reused blindly.

## Current empirical basis

As of `2026-05-06`:

- successful board-measured cases: `40`
- current implementation status split:
  - `40` measured
  - `0` retryable
  - `44` defer_current_impl
- dominant failure mode among deferred cases: `csynth_crash_or_oom`

Observed recoveries under clean serial retry:

- `pw_conv_pw_ir24_expand_e3_112_16_48...`
- `dw_conv_k3_dw_ir24_e3_112_c48_s2...`
- `mbconv_e6_k3_mbconv_stage3_56_24_32_s2...`
- `dw_conv_k3_dw_ir32_e3_56_c72_s2...`
- `pw_conv_pw_ir24_proj_e3_56_48_24...`
- `pw_conv_pw_ir24_expand_e6_112_16_96...`
- `pw_conv_pw_ir24_proj_e6_56_96_24...`
- `pw_conv_pw_ir160_expand_e3_14_96_288...`
- `pw_conv_pw_ir160_proj_e3_7_288_160...`

Observed persistent failures are concentrated in very large packed widths and/or deep channel counts.

## Hard-stop rules

If a case matches any rule below, treat it as **defer / not worth continuing board measurement** unless the template itself is redesigned.

### 1. `pw_conv`

Defer if either input or output packed stream width exceeds `4096 bits`.

Equivalent channel thresholds:

- input channels `> 512`
- output channels `> 512`

Examples that fall into this bucket:

- `pw_conv_pw_ir160_expand_e6_14_96_576...`
- `pw_conv_pw_ir160_proj_e6_7_576_160...`
- `pw_conv_pw_head_7_320_1280...`

Rationale:

- successful cases reached up to `3072 bits` (`384 channels`)
- failures cluster at `4608 bits` (`576 channels`) and above

### 2. `dw_conv_k3`

Defer if channel count is `>= 96`.

Examples:

- `dw_conv_k3_dw_ir64_e6_28_c192_s2...`
- `dw_conv_k3_dw_ir160_e3_14_c288_s2...`
- `dw_conv_k3_dw_ir320_e3_7_c480_s1...`
- `dw_conv_k3_dw_ir320_e6_7_c960_s1...`

Rationale:

- recovered successes exist at `C=48` and `C=72`
- failures begin at `C=96` and then dominate all larger cases

### 3. `dw_conv_k5`

Defer if channel count is `>= 48`.

Examples:

- `dw_conv_k5_dw_ir24_e3_112_c48_s2...`
- `dw_conv_k5_dw_ir32_e3_56_c72_s2...`
- `dw_conv_k5_dw_ir160_e3_14_c288_s2...`

Rationale:

- only small pilot-scale `k5` cases are currently viable
- failures already appear starting at `C=48`

### 4. `mbconv_e3_k3` and `mbconv_e3_k5`

Defer all cases from **stage3 and beyond**.

Operationally:

- allow `stage2`
- defer `stage3`, `stage4`, `stage5`, `stage6`, `stage7`

Examples:

- `mbconv_e3_k3_mbconv_stage3_56_24_32_s2...`
- `mbconv_e3_k5_mbconv_stage7_7_160_320_s1...`

Rationale:

- only stage2 has yielded stable board-measured results
- all stage3+ cases remain in the failure set

### 5. `mbconv_e6_k3`

Allow through **stage3**, defer **stage4 and beyond**.

Operationally:

- allow `stage2`, `stage3`
- defer `stage4`, `stage5`, `stage6`, `stage7`

Rationale:

- `stage3` was recovered successfully in clean serial retry
- `stage4+` remains failure-dominated

### 6. `mbconv_e6_k5`

Defer all cases from **stage3 and beyond**.

Operationally:

- allow `stage2`
- defer `stage3`, `stage4`, `stage5`, `stage6`, `stage7`

Rationale:

- only stage2 currently has a successful board-measured result
- stage3+ remains failure-dominated

## Retry rules for non-hard-stop cases

If a case does **not** match a hard-stop rule:

1. retry once under clean serial conditions
2. if it times out once in clean serial retry, mark `defer_current_impl`
3. if it fails twice in clean serial retry, mark `defer_current_impl`

## Export / harness failures

These are retryable once after script or flow fixes:

- `harness_generation_failed`
- `export_crash_or_oom`

If they recur after one clean retry, stop spending board-measurement time on them and mark `defer_current_impl`.

## Practical use

Use these rules to split the remaining LUT work into two sets:

- `continue_board_measure`
  - cases still worth spending HLS / bitstream / board time on
- `defer_current_impl`
  - cases that should be skipped until templates are redesigned

This keeps LUT construction moving instead of letting a small number of pathological cases dominate wall-clock time.

At the current checkpoint, the retryable set has been exhausted. The remaining unresolved LUT work under the current implementation is the `44`-case `defer_current_impl` set.
