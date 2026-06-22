# Phase0 v3 Deployable Search Evidence

Last updated: 2026-06-07

This note records the evidence needed to explain why the Phase0 v3 search
procedure can produce board-deployable candidates. It is a companion analysis
document. It does not replace or mutate the Phase0 v3 board evidence in
`docs/PHASE0_V3_BOARD_RESULTS.md`.

## Scope And Terminology

- Search-level candidate: an architecture produced by RL search and ranked by
  search proxy metrics such as eval10 macro_f1 and strict LUT aggregate
  latency/resource estimates.
- Claimable candidate: a full-network candidate that has full-route PASS,
  bitstream present, actual Vivado DSP within the gate threshold, matching
  bitstream SHA256, and COM5 measurement `status_code=0`.
- Retrain metric: final PyTorch retraining accuracy on the NKSID split. It is
  not a board measurement. If retrained weights are to be claimed on board,
  weights must be exported, the bitstream rebuilt, and COM5 re-measured.

## Search-Space Shrinkage Evidence

The Phase0 v3 search space was reduced for hardware feasibility, not for
arbitrary simplification. Under the current fixed stage depths/channels, the
stage-level block-choice product changed from 8 to 4, indicating a tighter
operator-policy space. This number does not represent the full architecture
search space, because the full specification also includes channels, strides,
depth settings, skip choices, expansion policy, and hardware constraints.

| Item | Phase0 v2 | Phase0 v3 low-DSP |
|---|---:|---:|
| Stage choice sizes | `[1, 4, 1, 2]` | `[1, 2, 1, 2]` |
| Restricted stage block-option product | 8 | 4 |
| Stage1 choices | k3/e3, k5/e3, k3/e6, k5/e6 | k3/e3, k3/e6 |
| Kernel choices | `[1, 3, 5]` | `[1, 3]` |
| Expand choices | `[1, 3, 6]` | `[1, 3, 6]` |
| Search episodes | 300 | 300 |
| Search eval epochs | 10 | 10 |
| Selection metric | macro_f1 | macro_f1 |
| Pareto topk | 16 | 24 |
| Scalar reward weights | accuracy 1.0; hardware weights 0 | accuracy 1.0, latency 0.02, resource 0.04, DSP 0.08, BRAM 0.04, LUT 0.03 |
| strict_board_lut exact mode | enabled | enabled |
| Shape-only LUT match | disabled | disabled |
| LUT interpolation | disabled | disabled |

The removed structures are mainly the k5 branches in early MBConv choices.
They increase compute, buffering, and routing pressure. The route-gate evidence
supports this design choice: the v2 candidates selected for route-gating all
failed full route with DSP at 840, global congestion level 6, and negative WNS.
By contrast, the v3b low-DSP route-aware run used a low-DSP override catalog and
all four route-gated candidates passed full route.

| Item | Phase0 v2 | Phase0 v3b low-DSP prune6 |
|---|---:|---:|
| Restricted stage block-option product | 8 | 4 |
| RL total evaluated records | 300 | 300 |
| Infeasible search-level records | 16 | 0 |
| Pareto feasible candidate pool | 284 | 300 |
| strict LUT exact-match hits in generated search records | 2100/2100 | 2100/2100 |
| strict LUT hit rate for actual strict-LUT queries | 100% | 100% |
| true_misses | 0 | 0 |
| deferred_hits | 0 | 0 |
| Route-gate candidates | 3 | 4 |
| Quick/default route PASS | 0 | 0 |
| Full-route PASS | 0 | 4 |
| COM5 status_code=0 claimable measurements | 0 | 4 |
| Claimable ratio among route-gated candidates | 0/3 = 0% | 4/4 = 100% |
| Claimable ratio among total evaluated records | 0/300 = 0% | 4/300 = 1.33% |

Important denominator notes:

- The 100% LUT coverage above means exact-match coverage for the strict-LUT
  queries actually produced in these Phase0 v2/v3b runs. It should not be
  written as coverage of all possible CNN shapes outside the declared search
  space.
- The `pareto_selection.json` `candidate_count` is the feasible candidate pool.
  The search code passes `searcher.feasible_candidates` into Pareto selection;
  therefore v2 has 300 evaluated records but only 284 Pareto candidates because
  16 search-level records were infeasible.
- The v3b `4/4` claimable ratio applies only to selected route-gated top
  candidates in this run. It should be written as: among the selected
  route-gated candidates, all four v3b candidates passed full-route and COM5
  validation. It must not be generalized to all search candidates.

## strict_board_lut Measurement Conditions

The strict LUT policy is intentionally conservative. An entry is usable only
when UART `status_code=0` and WNS is non-negative. Timing-fail rows are kept as
references but are not used as strict NAS hardware costs.

| Condition | Current evidence | Paper/report wording |
|---|---|---|
| Board target | `ALINX_AV7K325`, `xc7k325t-ffg900-2` | Fixed AV7K325 target |
| Clock | board config `period_ns=5.0`, `freq_hz=200000000`; COM5 measurement JSON reports `clock_freq_hz=200000000` | 200 MHz, 5 ns |
| Vivado version for route/board reports | Vivado 2023.2, build 4029153, in full-route timing/utilization reports and logs | Vivado 2023.2 |
| Quantization bitwidth | strict LUT `op_spec.bitwidth=8`; search config `quantization_bits=8` | 8-bit quantized operator/harness setting |
| Shape and op key | `op_spec` records op, input/output channels, input resolution, kernel, stride, groups, expand ratio, parallelism, unroll factor | Exact shape/op lookup |
| Interpolation | `lut_enable_interpolation=false` | No interpolation |
| Shape-only match | `lut_allow_shape_only_match=false` | No approximate shape matching |
| Resource report stage | strict LUT rows come from board harness timing/utilization records for measured operator/combo cases; full-network claimable resources come from Vivado placed/full-route reports | Do not call these HLS estimates |
| Power | strict LUT metadata sets `power_w=null`; no estimated power is promoted as measured power | Do not report power as measured unless separately measured |
| HLS/Vitis global version | source paths/logs exist, but the strict LUT metadata does not store one single global HLS/Vitis version field | Add explicit provenance before claiming a single global HLS/Vitis version |

This strict policy improves reproducibility because the NAS estimator cannot
silently interpolate or substitute a similar shape. The tradeoff is reduced
search freedom: potentially good structures are excluded until their exact
operator/combo shapes have stable board-harness coverage.

## DSP <= 700 Gate

The AV7K325 target has 840 DSP blocks. The Phase0 v3 full-route gate requires
actual Vivado DSP <= 700, which is 83.3% of the board DSP count and leaves 140
DSP blocks, or 16.7%, as routing/control/system margin.

The threshold is an engineering feasibility gate derived from observed route
failures rather than a formal timing theorem:

- v2 route-gated candidates: full route failed, DSP reached 840, congestion
  level was 6, and WNS was negative.
- v3b quick/default route: all four candidates still failed at DSP 840 and
  congestion level 6.
- v3b full low-DSP route: all four candidates passed with actual DSP 524 or
  612, WNS from 0.022 ns to 0.223 ns, and no missing low-DSP coverage.

Although all four claimable candidates achieved non-negative WNS, some timing
margins were narrow, especially `rl_arch_185` with WNS 0.022 ns. The deployment
claim is therefore tied to the recorded Vivado run, bitstream hash, and COM5
measurement artifact, not to an assumption that every future route seed will
close timing.

Recommended manuscript wording:

> Based on previous failed implementations, candidates with high early-stage
> expansion and high DSP utilization were more likely to fail timing closure or
> routing. Therefore, a conservative actual-DSP upper bound was used as a
> route-aware feasibility gate.

## Reward And Pareto Metric Definitions

The scalar RL reward uses the current run's observed maxima for normalization:

```text
R = w_acc * norm(task_score)
    - w_latency * norm(latency_ms)
    - w_energy * norm(energy_mj)
    - w_dsp * norm(DSP)
    - w_bram * norm(BRAM)
    - w_lut * norm(LUT)
```

For Phase0 v3 low-DSP, the configured weights are:

```text
w_acc=1.0, w_latency=0.02, w_energy=0.0,
resource fallback=0.04, w_dsp=0.08, w_bram=0.04, w_lut=0.03
```

Energy-related terms were disabled in the scalar reward for this run
(`w_energy=0.0`). Board-level power was not promoted as a measured metric in
Phase0 v3b; any power-related objective should be treated only as a proxy unless
separate board-level power measurement is added.

The Pareto layer maximizes the selected task metric, here macro_f1, and
minimizes active hardware or route-risk objectives. With the v3 config, the
relevant objectives include latency, DSP, BRAM, LUT, power proxy,
physical_risk, and early_expand_pressure when enabled by objective weights,
constraints, or physical Pareto flags.

The physical metrics are route-risk proxies, not a real physical-routing model.
Their current implementation is:

```text
early_expand_pressure =
    sum over MBConv/FusedMBConv blocks of
    (input_resolution^2 * Cin * max(0, expand_ratio - 1)) / 1e6

interconnect_pressure =
    sum over blocks of
    (output_resolution^2 * (Cin + Cout + hidden_channels)) / 1e6

memory_pressure = peak_buffer_bytes / 1024^2

fanout_pressure =
    max over blocks of (input_resolution * hidden_channels) / 1024

physical_risk =
    early_expand_pressure
    + 0.25 * interconnect_pressure
    + 0.10 * memory_pressure
    + 0.10 * fanout_pressure
```

In Phase0 v3, early-expand hard limits were relaxed to e6 only so the measured
low-DSP stage1 e6 case could survive pre-pruning. Final claimability still
depends on full-route PASS, actual Vivado DSP <= 700, matching bitstream, and
COM5 status_code=0.

## Search-Level, Retrain-Level, And Claimable-Level Reporting

Do not mix these three result layers:

| Layer | Metric source | What can be claimed |
|---|---|---|
| Search-level | RL eval10 candidate JSON and strict LUT aggregate estimates | NAS candidate ranking and hardware-risk screening |
| Retrain-level | 150-epoch PyTorch retrain on NKSID split | Final software accuracy for retrained architectures |
| Claimable board-level | Full-route gate, bitstream SHA, COM5 measurement JSON | Board-deployable candidate status and real board e2e latency |

Search eval10 macro_f1 and retrain150 macro_f1 should not be directly compared
as the same measurement. The eval10 value is a short search proxy for candidate
ranking; retrain150 is the final software accuracy protocol; board-level
artifacts establish deployability and real latency.

The current claimable board candidates are:

| Arch | Role | Search eval10 macro_f1 | Retrain150 macro_f1 | Retrain150 top1 | Real e2e ms | WNS/TNS ns | LUT | DSP | BRAM | Power | Bitstream | COM5 |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|---:|
| `rl_arch_185` | additional claimable model | 0.619166 | not run | not run | 49.025740 | 0.022 / not reported | 15380 | 524 | 101 | not measured | match | 0 |
| `rl_arch_186` | accuracy-first claimable model | 0.629512 | 0.854482 | 0.915385 | 49.062010 | 0.094 / not reported | 18598 | 612 | 127.5 | not measured | match | 0 |
| `rl_arch_242` | latency-efficient claimable model | 0.627406 | 0.839023 | 0.913462 | 24.872910 | 0.113 / not reported | 18553 | 612 | 127.5 | not measured | match | 0 |
| `rl_arch_276` | resource-min claimable model | 0.605372 | not run | not run | 24.836150 | 0.223 / not reported | 15354 | 524 | 101 | not measured | match | 0 |

After independent 150-epoch retraining, `rl_arch_186` remains the accuracy-first
candidate because it has higher retrain macro_f1 and top1 than `rl_arch_242`.
`rl_arch_242` remains the recommended latency-efficient deployable candidate
because its real board latency is about half that of `rl_arch_186`. However, it
no longer satisfies the original search-stage "within 0.003 macro_f1" closeness
rule after retrain; use latency-efficient as the primary label.

`rl_arch_185` and `rl_arch_276` are board-claimable but were not independently
retrained in the 150-epoch protocol. `rl_arch_276` can be called resource-min
only within the already claimable candidate set; without retrain150 it should
not be presented as a final accuracy-resource Pareto optimum.

## Source Artifacts

- `configs/search/nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_v2_av7k325.yaml`
- `configs/search/nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_v3_lowdsp_av7k325.yaml`
- `hls_lut_builder/results/nas_board_lut_strict_current84_arch84/summary.json`
- `hls_lut_builder/results/nas_board_lut_strict_current84_arch84/nas_board_lut_status.json`
- `results/remote_phase0_v2_rl300_search/results/summary.json`
- `results/remote_phase0_v2_rl300_search/results/lut_stats.json`
- `results/remote_phase0_v3b_lowdsp_prune6_rl300_search/results/nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_v3b_lowdsp_prune6/results/summary.json`
- `results/remote_phase0_v3b_lowdsp_prune6_rl300_search/results/nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_v3b_lowdsp_prune6/results/lut_stats.json`
- `hls_lut_builder/board_harness/results/pareto_route_gate_phase0_v2_pending/route_gate_summary.json`
- `hls_lut_builder/board_harness/results/pareto_route_gate_phase0_v3b_lowdsp_prune6/route_gate_summary.json`
- `hls_lut_builder/board_harness/results/pareto_route_gate_phase0_v3b_lowdsp_prune6/reports/phase0_v3_board_candidate_comparison.md`
- `results/retrain_phase0_v3_claimable_20260606/phase0_v3_claimable_retrain_comparison.json`
- `run_search.py`
- `src/hwnas_fpga/search/rl_searcher.py`
- `src/hwnas_fpga/search/pareto.py`
- `src/hwnas_fpga/hardware/cost.py`
