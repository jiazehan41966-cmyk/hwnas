# Next-Round Template Refactor Priorities

This document defines the **next round of template refactor work** for the LUT builder under the current implementation stack:

- `packed_stream_v1`
- current templates in `hls_lut_builder/templates`
- current board harness flow in `hls_lut_builder/board_harness`
- Vitis HLS 2023.2 / Vivado 2023.2

This is not a theory document. It is a **pragmatic refactor plan** derived from the current measured/deferred split:

- `84` total cases
- `40` measured
- `44` `defer_current_impl`

The goal is not to redesign everything at once. The goal is to recover the largest number of currently deferred cases with the smallest number of template-level changes.

## Prioritization method

Priority is based on four factors:

1. **Impact**
   Number of currently deferred LUT cases blocked by a template.
2. **Leverage**
   Whether the refactor creates a reusable implementation pattern for other kernels.
3. **Tractability**
   Whether the current code suggests a clear local bottleneck that can be changed without rewriting the whole flow.
4. **Validation cost**
   Whether success/failure can be checked quickly on a small boundary subset before reopening the whole deferred region.

## Current deferred breakdown

By template / operator family:

| Template | Deferred cases | Current defer frontier |
| --- | ---: | --- |
| `dw_conv.cpp.tmpl` | `22` | `dw_conv_k3` at `C >= 96`, `dw_conv_k5` at `C >= 48` |
| `mbconv.cpp.tmpl` | `19` | `mbconv_e3_*` at `stage3+`, `mbconv_e6_k3` at `stage4+`, `mbconv_e6_k5` at `stage3+` |
| `pw_conv.cpp.tmpl` | `3` | packed stream width `> 4096 bits` |

By root cause:

| Root cause | Count |
| --- | ---: |
| `csynth_crash_or_oom` | `39` |
| `export_crash_or_oom` | `3` |
| `csynth_completed_but_report_missing` | `1` |
| `csynth_failed_no_artifact` | `1` |

The dominant issue is still **HLS front-end scalability**, not board execution.

## Priority 0: Add a heavy-case implementation tier

**Scope**

- `dw_conv.cpp.tmpl`
- `mbconv.cpp.tmpl`
- `pw_conv.cpp.tmpl`

**Why this is first**

The current templates implicitly assume that one implementation style can span the whole search space. The data says that assumption is no longer true. Small and medium cases are fine. Heavy cases fail before board measurement.

The next round should explicitly separate:

- **fast path** for already-working small/medium cases
- **conservative heavy-case path** for large-channel / wide-stream cases

**Required changes**

1. Introduce a compile-time decision path for heavy cases.
2. Stop tying physical interface width directly to logical channel count for the heavy path.
3. Cap internal parallelism with explicit upper bounds instead of letting it scale directly with `IN_CH`, `EXP_CH`, or `OUT_CH`.

**Success criterion**

The template chooses a safer implementation style before HLS enters the current crash region.

## Priority 1: Refactor `dw_conv.cpp.tmpl`

**Impact**

- Highest single-template impact: `22` deferred cases

**Why it is first among kernel templates**

- It blocks both `dw_conv_k3` and `dw_conv_k5`
- It is structurally simpler than `mbconv`
- A successful refactor here should provide the storage/tiling pattern needed later inside `mbconv`

**Current code signals**

The current template uses:

- `MAKE_AXIS_TYPE(IN_CH * 8)`
- `line_buffer[IN_CH][K][PADDED_W]`
- channel loops unrolled with `factor=CH_TILE`
- per-channel shifting of the full line buffer inside the main pipeline

This is workable at low and medium channel counts, but it scales poorly when:

- `IN_CH` grows
- `K = 5`
- packed channel width grows with `IN_CH`

**Likely bottlenecks**

1. Full-channel line-buffer organization
2. Channel-scaled interface width
3. Channel-scaled array partitioning and unrolling
4. Memory-port pressure in the line-buffer update path

**Refactor direction**

1. Replace full-channel physical width with a **channel-tiled heavy path**
   - Keep the logical operator the same
   - Move to multiple beats or multiple channel tiles per pixel for heavy cases
2. Rework line-buffer storage to be **tile-local**, not full-channel physical state updated in one path
3. Split `k3` and `k5` heavy behavior
   - `k5` likely needs a more conservative path than `k3`
4. Cap physical parallelism independently of `IN_CH`
   - example: fixed lane groups rather than `CH_TILE = array_partition_factor -> full scale with case size`

**Validation subset**

First reopen the current boundary cases, not the deepest failures:

- `dw_conv_k3_dw_ir24_e6_112_c96_s2...`
- `dw_conv_k3_dw_ir64_e3_28_c96_s2...`
- `dw_conv_k5_dw_ir24_e3_112_c48_s2...`
- `dw_conv_k5_dw_ir32_e3_56_c72_s2...`

If those still fail, there is no reason to try `C=288/480/960`.

## Priority 2: Refactor `mbconv.cpp.tmpl`

**Impact**

- `19` deferred cases

**Why this is second**

- Large impact
- But materially more complex than `dw_conv`
- It combines three stages and inherits both `pw_conv` and `dw_conv` scaling issues

**Current code signals**

The current template uses:

- full-width stage interfaces:
  - `expanded_word_t = ap_uint<EXP_CH * 8>`
  - `input_word_t = ap_uint<IN_CH * 8>`
- large local arrays:
  - `expand_weights[EXP_CH][IN_CH]`
  - `depthwise_weights[EXP_CH][K][K]`
  - `project_weights[OUT_CH][EXP_CH]`
- dataflow with very shallow stream depths (`depth=4`)
- `EXP_TILE` / `OUT_TILE` driven directly by current config knobs

This is already enough to explain why:

- `mbconv_e3_*` breaks from `stage3+`
- `mbconv_e6_k3` breaks from `stage4+`
- `mbconv_e6_k5` breaks from `stage3+`

**Likely bottlenecks**

1. Full `EXP_CH`-wide intermediate representation between stages
2. Simultaneous scaling of:
   - expand weights
   - depthwise storage
   - project weights
3. Shallow dataflow FIFOs for heavy intermediate traffic
4. `EXP_TILE` scaling too aggressively with heavy `e6` cases

**Refactor direction**

1. Add a **tiled intermediate representation** for heavy cases
   - do not require one physical `EXP_CH * 8` word between stages
2. Decouple `EXP_TILE` and `OUT_TILE` from raw case size
   - conservative caps for heavy `e6` and deep-stage cases
3. Consider stage-class-specific implementations
   - `stage2/3` path
   - `stage4+` heavy path
4. Increase or restructure dataflow buffering only after the width issue is reduced
   - deeper FIFOs alone will not solve full-width explosion

**Validation subset**

Validate in this order:

1. `mbconv_e3_k3_mbconv_stage3_56_24_32_s2...`
2. `mbconv_e3_k5_mbconv_stage3_56_24_32_s2...`
3. `mbconv_e6_k3_mbconv_stage4_28_32_64_s2...`
4. `mbconv_e6_k5_mbconv_stage3_56_24_32_s2...`

These are the first currently deferred boundaries. If they recover, the deeper stages become worth reopening.

## Priority 3: Refactor `pw_conv.cpp.tmpl`

**Impact**

- Only `3` deferred cases

**Why this is still important**

- It is the cleanest place to introduce a **segmented wide-channel interface strategy**
- The same idea is likely needed later inside `mbconv`
- The remaining failures are exactly the ultra-wide packed cases

**Current code signals**

The current template uses:

- `MAKE_AXIS_TYPE(IN_CH * 8)`
- `MAKE_AXIS_TYPE(OUT_CH * 8)`
- one logical pixel transferred as one physical packed word

That works until channel widths become extreme. The remaining deferred `pw_conv` cases are:

- `pw_conv_pw_head_7_320_1280...`
- `pw_conv_pw_ir160_expand_e6_14_96_576...`
- `pw_conv_pw_ir160_proj_e6_7_576_160...`

All three sit beyond the current `4096-bit` empirical boundary.

**Likely bottlenecks**

1. Over-wide AXIS words
2. Over-wide packing/unpacking logic
3. Export / implementation stress from giant buses and weight organization

**Refactor direction**

1. Introduce a **segmented physical stream format** for wide `pw_conv`
   - logical channels stay the same
   - physical transport becomes multiple beats per pixel
2. Cap physical bus width at a fixed maximum
   - similar in spirit to how `fc_layer` needed a segmented path instead of a monolithic bus
3. Keep the current single-word fast path for ordinary widths

**Validation subset**

- `pw_conv_pw_ir160_expand_e6_14_96_576...`
- `pw_conv_pw_ir160_proj_e6_7_576_160...`
- `pw_conv_pw_head_7_320_1280...`

If two of these recover, the segmented-wide strategy is validated.

## Priority 4: Flow hygiene after template changes

This is not a kernel refactor, but it should be done immediately after any template change.

**Required follow-up**

1. Re-run only boundary subsets first
2. Refresh:
   - `board_measure_status_current_impl.csv`
   - `board_measure_status_current_impl.json`
   - `board_measure_deferred_cases_current_impl.csv`
3. Do not immediately re-open all `44` deferred cases
4. Promote recovered cases from `defer_current_impl` to measured only after full board measurement succeeds

## Recommended execution order

1. Build the heavy-case implementation split framework
2. Refactor `dw_conv.cpp.tmpl`
3. Validate `dw_conv` boundary cases
4. Port the same heavy-case strategy into `mbconv.cpp.tmpl`
5. Validate `mbconv` boundary cases
6. Refactor `pw_conv.cpp.tmpl` for segmented wide-channel transport
7. Validate the 3 deferred `pw_conv` cases

## What not to do next

1. Do not reopen all `44` deferred cases without template changes
2. Do not treat the current cutoffs as theory
3. Do not optimize harness or board flow first
   - the current dominant blocker is still HLS/template scalability

## Immediate next tasks

If only one template should be touched first, start with:

- [dw_conv.cpp.tmpl](/E:/1/hwnas/hwnas/hls_lut_builder/templates/dw_conv.cpp.tmpl)

If the goal is highest impact with manageable scope, the first work package should be:

1. design heavy-case channel tiling for `dw_conv`
2. keep existing small/medium path unchanged
3. reopen the four `dw_conv` boundary cases

That will give the strongest signal on whether the next round can actually reduce the deferred set.
