# Deployable LUT v2 Merge Policy

Date: 2026-05-15

This policy converts the current failed44 fixed-vector rerun, k5 low-complexity
follow-up, depthwise fallback blockers, and pointwise v2.1 measurements into a
deployable NAS LUT without mixing HLS-only diagnostics into deployable entries.

## Entry Admission

An operator entry is deployable only when all of the following are true:

- HLS synthesis status is `success`.
- Observed II is less than or equal to the profile `target_ii`.
- Vivado downstream status is `success`.
- Post-route LUT, FF, DSP, BRAM/RAMB, Fmax, and power are present.
- `deployable_at_200mhz` is true from post-route timing.

HLS-only rows, downstream-missing rows, target-II misses, and backend-sensitive
profile failures remain in status files but are excluded from
`deployable_lut_v2.json`.

## Profile Precedence

1. Exact deployable implementation match wins.
2. For medium/large `dw_conv_k5`, prefer p8/cb8/II2 measured implementations
   over p16/cb16 rows when the shape is one of the validated boundary profiles:
   C480 7x7 s1, C576 14x14 s2, or C960 7x7 s1.
3. For the remaining failed44 depthwise blockers, prefer measured p8/cb8
   fallback rows over failed p16/cb16 rows when the exact shape matches.
4. p16/cb16 medium/large k5 rows that failed during Vivado `synth_design`
   helper/runtime setup are labelled `backend_sensitive_profile_rtl` and never
   become deployable entries.
5. Pointwise rows must come from pointwise v2.1 or a later measured profile.
   The old `pw_conv_fixed_vector` p16/cb16 rows are excluded because observed
   II=8 is caused by unbanked input unpack and the template lacks explicit
   PE x SIMD input-channel folding.

## k5 Depthwise Mapping

Use the measured low-complexity k5 entries:

- C480, 7x7, s1: `PACK_CH=8`, `CH_BLOCK=8`, `target_ii=2`.
- C576, 14x14, s2: `PACK_CH=8`, `CH_BLOCK=8`, `target_ii=2`.
- C960, 7x7, s1: `PACK_CH=8`, `CH_BLOCK=8`, `target_ii=2`.

Smaller k5 p16/cb16 entries can enter the deployable LUT only if they have
their own post-route success. They are not inferred from HLS success.

## Pointwise v2.1 Mapping

The first v2.1 profile family is:

- `PACK_CH=16`
- `output_parallelism=16` as PE
- `input_parallelism=4` as SIMD
- `array_partition_factor=16` for input-channel banking
- target II diagnostics in `{2, 4, 8}`

The policy is to admit the lowest-latency pointwise v2.1 row that satisfies
the entry-admission rules. If multiple profiles are deployable for the same
shape, keep the lowest latency row as the default and preserve the others in
the status table as alternate deployable profiles.

## MBConv Decomposition

Until fused MBConv is restored, MBConv deployability is computed from:

`pw_expand + dw + pw_project`

A decomposed MBConv combination is deployable only when every primitive
component is deployable under this policy.

For latency, use the sum of primitive latencies. For resource feasibility,
report the conservative sum of primitive LUT/FF/DSP/BRAM until the execution
model proves time-multiplexed reuse. Additionally, include inter-stage storage:

- expand output buffer: `H_expand * W_expand * hidden_channels * bitwidth`
- depthwise output buffer: `H_project * W_project * hidden_channels * bitwidth`

Convert each buffer to BRAM18 with `ceil(bits / 18432)`. A combo missing this
buffer accounting is not a final deployable MBConv LUT row even if all three
primitive rows have post-route success.

## Output Artifacts

The current deployable merge is produced under
`hls_lut_builder/results/v2_deployable_lut/`:

- `deployable_lut_v2.json`: operator-level deployable entries only.
- `combo_deployability_v2.csv`: failed44 old-combo status, primitive mapping,
  selected profile per primitive, sum latency, conservative resource sum,
  inter-stage buffer BRAM, and exclusion reason.
- `merge_summary_v2.json`: source paths, admission rules, status counts, and
  reason buckets.

As of 2026-05-16, the merge contains 5 deployable operator entries and 5
deployable old direct-depthwise combinations. The remaining 39 old combinations
are excluded because at least one selected primitive is still HLS-only with
downstream metrics missing. Pointwise v2.1 has fixed the HLS target-II miss for
all 23 `target_ii=2` failed44 pointwise shapes, but it has not yet produced
deployable pointwise entries.

As of 2026-05-17, the pointwise v2.1 downstream pass and the two-row depthwise
p8/cb8 fallback pass are merged. The deployable LUT contains 44 operator
entries, and all 44 old failed44 combinations are deployable at measured
post-route level.

Status files should retain rejected rows with explicit reason buckets:

- `hls_only_downstream_missing`
- `target_ii_missed`
- `backend_sensitive_profile_rtl`
- `downstream_failed`
- `missing_inter_stage_buffer_accounting`
