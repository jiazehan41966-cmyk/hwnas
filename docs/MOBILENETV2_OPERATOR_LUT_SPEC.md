# MobileNetV2 Operator and LUT Specification

## Scope

> Status note (2026-07-19): this is the historical operator-screening and LUT
> planning specification. The current semantic-safe runtime policy supersedes
> its `keep` labels for deployment-claimable search: `denoise`, `edge`, and
> `mixconv` remain paused until the required PyTorch/fixed-point/HLS parity and
> G5 evidence pass. Current searches admit only the operators allowed by
> `hls_lut_builder/configs/operator_manifest_semantic_safe.yaml`.

This document fixes the operator-selection and LUT-profiling protocol after the
fair macro-backbone comparison selected `MobileNetV2` as the unified
`accuracy_anchor` and `search_anchor`.

The purpose of this document is to separate three different layers cleanly:

1. `search-level operators`: the blocks that NAS is allowed to choose.
2. `support operators`: fixed modules that stay in the network but are not
   searchable.
3. `LUT kernel measurements`: the lower-level HLS kernels that must be
   profiled to support hardware estimation.

This specification supersedes any earlier ShuffleNetV2-centered operator plan.

## Basis

- Macro backbone winner: `MobileNetV2`
  - Source blueprint: [run_backbone_baseline.py](../run_backbone_baseline.py)
- Operator screening result:
  - [operator_retention_table.json](../results/operator_ablation_nksid_av7k325_search_anchor_singleops_ep120_gpu_20260418/results/operator_retention_table.json)
  - [operator_ablation_summary.json](../results/operator_ablation_nksid_av7k325_search_anchor_singleops_ep120_gpu_20260418/results/operator_ablation_summary.json)

## Final Search-Level Operator Set

These are the operators that should define the MobileNetV2-family micro-NAS
search space after operator screening.

| Operator | Status | Role | Decision basis |
| --- | --- | --- | --- |
| `mbconv` | `keep` | Core searchable block | MobileNetV2 main operator; strong standalone result |
| `fused_mbconv` | `keep` | Core searchable block | Competitive in mixed candidates; useful early-stage variant |
| `denoise` | `keep` | Sonar-friendly extension | Best mixed candidate was `mobile_plus_denoise` |
| `edge` | `keep` | Sonar-friendly extension | Competitive and sonar-oriented |
| `skip` | `keep` | Lightweight structural option | Improves compact mixed variants |
| `mixconv` | `conditional` | Optional extension | Competitive but weaker than `denoise`/`edge` |
| `dw_pw_conv` | `drop` | Removed from main MobileNetV2 line | Clearly behind under the fixed MobileNetV2 template |

## Fixed Support Operators

These operators remain in the network definition, but they are not part of the
searchable operator pool.

| Module | Fixed implementation | Notes |
| --- | --- | --- |
| Stem | `conv_bn_relu6` | MobileNetV2 stem |
| Head pooling | `adaptive_avg_pool_1x1` | Fixed global pooling |
| Head projection/classifier | `conv1x1` or FC head path | Determined by the architecture blueprint |

## Recommended Search-Space Update

For the next formal MobileNetV2-family NAS profile, the searchable operator set
should be updated to:

```yaml
op_choices:
  - mbconv
  - fused_mbconv
  - denoise
  - edge
  - skip
  # mixconv is optional and can be enabled in an ablation profile
```

`dw_pw_conv` should be removed from the main MobileNetV2 profile and kept only
for legacy comparisons if needed.

## LUT Measurement Targets

The LUT should not be built directly from the old coarse block names alone.
Instead, each searchable block is mapped onto a small set of lower-level HLS
kernel measurements.

### LUT I/O contract

The formal LUT interface contract is a packed-stream contract:

- input feature stream width = `in_channels * 8` bits
- output feature stream width = `out_channels * 8` bits
- weights / biases are exposed through `BRAM`
- control is exposed through `AXI-Lite`

This contract is meant to align LUT estimation with a fully pipelined hardware
template. The `stem_conv_k3_s2` pilot has been migrated and validated under
this contract first; the remaining kernels should follow the same I/O policy
before large-scale LUT sampling is launched.

### Kernel-level LUT entries

These are the kernel types that should be profiled explicitly.

| LUT kernel | Required | Used by |
| --- | --- | --- |
| `conv1x1` | yes | stem/head projections, MBConv expand/project, fused block projection |
| `conv3x3` | yes | stem and fused spatial conv with `k=3` |
| `conv5x5` | optional | fused spatial conv with `k=5` if enabled |
| `dwconv3x3` | yes | MBConv and denoise/edge variants with `k=3` |
| `dwconv5x5` | yes | MBConv or denoise/edge variants with `k=5` |
| `adaptive_avg_pool_1x1` | yes | fixed head pooling |
| `skip` | yes | zero/near-zero cost structural path |

### Block-to-kernel mapping

| Search block | LUT support kernels |
| --- | --- |
| `mbconv(k=3/5, expand=r)` | `conv1x1` expand + `dwconv{k}` + `conv1x1` project |
| `fused_mbconv(k=3/5, expand=r)` | `conv{k}` fused spatial conv + `conv1x1` project |
| `denoise(k=3/5)` | `dwconv{k}`-style local branch + smoothing branch + `conv1x1` fuse |
| `edge(k=3/5)` | edge-aware spatial branch + `conv1x1` fuse |
| `mixconv` | grouped mixed-depthwise branch set + `conv1x1` fuse |
| `skip` | `skip` |

`denoise`, `edge`, and `mixconv` still need custom HLS handling, but they
should be treated as first-class LUT targets because they survived operator
screening.

## Representative Shape Buckets

The LUT should be organized around the MobileNetV2 stage skeleton, not the old
ShuffleNetV2 stage structure.

The stage blueprint in the macro backbone definition is:

- stem: `32`, stride `2`
- `ir16`: depth `1`, channels `16`, stride `1`
- `ir24`: depth `2`, channels `24`, stride `2`
- `ir32`: depth `3`, channels `32`, stride `2`
- `ir64`: depth `4`, channels `64`, stride `2`
- `ir96`: depth `3`, channels `96`, stride `1`
- `ir160`: depth `3`, channels `160`, stride `2`
- `ir320`: depth `1`, channels `320`, stride `1`
- head: `1280`

### Stage-level representative shapes

| Bucket | Spatial change | Channel change | Typical module |
| --- | --- | --- | --- |
| Stem | `224 -> 112` | `1 -> 32` | `conv_bn_relu6` |
| Stage1 `ir16` | `112 -> 112` | `32 -> 16` | MBConv, `expand=1`, `k=3` |
| Stage2 `ir24` | `112 -> 56` | `16 -> 24` | MBConv/Fused, `k=3/5` |
| Stage3 `ir32` | `56 -> 28` | `24 -> 32` | MBConv/Fused, `k=3/5` |
| Stage4 `ir64` | `28 -> 14` | `32 -> 64` | MBConv/Fused, `k=3/5` |
| Stage5 `ir96` | `14 -> 14` | `64 -> 96` | MBConv/Fused, `k=3/5` |
| Stage6 `ir160` | `14 -> 7` | `96 -> 160` | MBConv/Fused, `k=3/5` |
| Stage7 `ir320` | `7 -> 7` | `160 -> 320` | MBConv/Fused, `k=3/5` |
| Head | `7 -> 1` | `320 -> 1280` | `conv1x1` + `adaptive_avg_pool_1x1` |

### Practical LUT profiling rule

The nine buckets above are the stage skeleton. They are the minimum required
coverage, but not necessarily the only final LUT rows.

For each searchable operator family, profiling should cover:

1. all stage buckets where the operator is allowed to appear,
2. both `k=3` and `k=5` when supported,
3. representative width settings that can actually occur in the final MobileNetV2
   search space.

This means the final LUT should be built as:

```text
operator family x representative stage bucket x valid kernel/expand variant
```

not as a single flat list of block names.

## Immediate Follow-up Actions

1. Update the formal MobileNetV2 NAS config so the main searchable set matches
   the retained operators.
2. Create a LUT profiling manifest using the kernel-level targets listed above.
3. Add an explicit mapping layer from search blocks to LUT kernel entries in the
   hardware estimator.
4. Keep `mixconv` in a separate ablation profile unless a second controlled
   experiment upgrades it from `conditional` to `keep`.
