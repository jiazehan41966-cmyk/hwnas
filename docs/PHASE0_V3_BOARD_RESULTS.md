# Phase0 v3 Board Results

Last updated: 2026-06-06

This document is the handoff record for the Phase0 v3 low-DSP route-aware
search and board validation round. It is intentionally result-focused: NAS LUT
latency remains an estimator value, while the latencies below are real COM5
board measurements from full-network bitstreams.

The 2026-07-03 first-principles audit preserves these historical route/COM5
measurements but recalibrates classification metrics as legacy fold-0
validation evidence, not untouched-test generalization estimates.

## Scope And Guardrails

- Search family: Phase0 v3 low-DSP route-aware, current84/arch84 strict LUT.
- Remote training is complete for this round; do not reopen the server for this
  result handoff.
- Vivado route gate is complete locally; do not rebuild bitstreams unless a new
  experiment explicitly asks for it.
- Board measurements used COM5 on AV7K325 at 200 MHz.
- A candidate is claimable only when all are true:
  - full-route gate is `PASS`
  - bitstream exists
  - actual Vivado DSP is `<=700`
  - measurement bitstream SHA256 matches the route-clean bitstream
  - COM5 measurement has `status_code=0`

## Key Artifacts

- Search results:
  `results/remote_phase0_v3b_lowdsp_prune6_rl300_search/results/nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_v3b_lowdsp_prune6/results/pareto_selection.json`
- Route gate root:
  `hls_lut_builder/board_harness/results/pareto_route_gate_phase0_v3b_lowdsp_prune6/`
- Final comparison reports:
  - `hls_lut_builder/board_harness/results/pareto_route_gate_phase0_v3b_lowdsp_prune6/reports/phase0_v3_board_candidate_comparison.md`
  - `hls_lut_builder/board_harness/results/pareto_route_gate_phase0_v3b_lowdsp_prune6/reports/phase0_v3_board_candidate_comparison.csv`
  - `hls_lut_builder/board_harness/results/pareto_route_gate_phase0_v3b_lowdsp_prune6/reports/phase0_v3_board_candidate_comparison.json`
- First claimable `rl_arch_186` evidence bundle:
  `hls_lut_builder/board_harness/results/pareto_route_gate_phase0_v3b_lowdsp_prune6/evidence_bundles/phase0_v3_claimable_rl_arch_186_first_measure_20260606T083121/`

## Search And Route-Gate Evidence

- v3b remote RL300 search completed with `unique_encoding_count=4`.
- Strict LUT audit passed with `miss=0`, `true_miss=0`, and `deferred_hit=0`.
- Quick/default route gate failed for all 4 candidates, with Vivado DSP hitting
  `840` and congestion level `6`; this confirms that quick/default
  implementation is not claimable for this round.
- Full low-DSP route gate passed for all 4 candidates:
  - `rl_arch_185`: WNS `0.022 ns`, DSP `524`, LUT `15380`, BRAM `101`
  - `rl_arch_186`: WNS `0.094 ns`, DSP `612`, LUT `18598`, BRAM `127.5`
  - `rl_arch_242`: WNS `0.113 ns`, DSP `612`, LUT `18553`, BRAM `127.5`
  - `rl_arch_276`: WNS `0.223 ns`, DSP `524`, LUT `15354`, BRAM `101`

## Board-Claimable Results

All 4 full-route PASS candidates have claimable COM5 measurements.

| arch_id | macro_f1 | top1 | real_e2e_ms | cycles | WNS ns | LUT | BRAM | DSP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `rl_arch_185` | 0.619166 | 0.790385 | 49.025740 | 9805148 | 0.022 | 15380 | 101 | 524 |
| `rl_arch_186` | 0.629512 | 0.792308 | 49.062010 | 9812402 | 0.094 | 18598 | 127.5 | 612 |
| `rl_arch_242` | 0.627406 | 0.786538 | 24.872910 | 4974582 | 0.113 | 18553 | 127.5 | 612 |
| `rl_arch_276` | 0.605372 | 0.767308 | 24.836150 | 4967230 | 0.223 | 15354 | 101 | 524 |

Selection rules for this round:

- `accuracy-first`: `rl_arch_186`
- `latency-balanced`: `rl_arch_242`, because its macro_f1 is within `0.003` of
  `rl_arch_186` and its measured latency is about half.
- `resource-min`: `rl_arch_276`, because it has the lowest actual DSP and the
  lowest measured latency among claimable candidates.

`rl_arch_186` stability check: `runs=5`, all 5 runs returned `status_code=0`,
cycles fixed at `9812402`, checksum fixed at `33551839`, and latency standard
deviation was `0.0 ms`.

## Reproducibility Commands

Refresh the route-gate summary after measurements:

```bash
python hls_lut_builder/board_harness/scripts/pareto_route_gate.py refresh \
  --pareto-selection results/remote_phase0_v3b_lowdsp_prune6_rl300_search/results/nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_v3b_lowdsp_prune6/results/pareto_selection.json \
  --output-root hls_lut_builder/board_harness/results/pareto_route_gate_phase0_v3b_lowdsp_prune6 \
  --topk 24 \
  --full-topk 8 \
  --full-max-actual-dsp 700 \
  --lowdsp-override-catalog hls_lut_builder/board_harness/configs/phase0_v3_lowdsp_override_catalog.yaml
```

Regenerate the compact comparison report from the refreshed JSON if needed. Use
the existing comparison outputs above as the authoritative record for the
2026-06-06 Phase0 v3 board validation.
