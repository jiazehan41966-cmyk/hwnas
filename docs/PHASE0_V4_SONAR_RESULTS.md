# Phase0 v4 Sonar Results

Evidence snapshot: 2026-06-22; repository documentation sync: 2026-07-03.

## Scope

This handoff consolidates the current Phase0 v4 sonar evidence without merging
incompatible evidence classes. Search proxy, retrain150 validation, Vivado
route, COM5 deterministic harness measurement, image quality, and measured
power/energy must be reported separately.

## Evidence Status

| Evidence layer | Current status | Claim boundary |
|---|---|---|
| RL300 search proxy | completed for the selected v4 run | `macro_f1`, `top1`, latency, LUT, DSP, and BRAM are search-time proxy fields |
| PyTorch retrain150 | completed for `rl_arch_135/154/169/193/60` | validation-set classification evidence; not board accuracy |
| Vivado full route | 6 PASS, 1 FAIL among 7 Pareto rows | route-clean requires nonnegative WNS and actual DSP `<=700` |
| COM5 | 6 rows have stable five-run measurements | deterministic harness-input latency/output sanity only |
| Board validation accuracy | not run | do not convert COM5 sanity into NKSID board accuracy |
| PSNR/SSIM | completed on NKSID val fold 0 with 520 samples | `input_as_reference`; operator-effect/structure-preservation analysis only |
| Four-way sonar ablation | incomplete | no formal ablation conclusion while `comparison_ready=false` |
| Measured power/energy | not measured | Vivado/search power remains estimate or proxy |

## Classification Evidence

The search-proxy headline remains `rl_arch_135` with macro_f1 `0.642115` and
top1 `0.807692`. The corresponding retrain150 validation results are:

| Candidate | Stage-3 op | retrain150 macro_f1 | top1 | weighted_f1 |
|---|---|---:|---:|---:|
| `rl_arch_135` | `mbconv k3 e3` | 0.860728 | 0.921154 | 0.924047 |
| `rl_arch_154` | `edge k3 e1` | 0.797473 | 0.875000 | 0.879548 |
| `rl_arch_169` | `denoise k3 e1` | 0.828635 | 0.896154 | 0.902516 |
| `rl_arch_193` | `mbconv k3 e3` | 0.827046 | 0.900000 | 0.904171 |
| `rl_arch_60` | `skip k1 e1` | 0.746253 | 0.851923 | 0.856819 |

Source:
`results/retrain_phase0_v4_sonar_stage3_k3_topk_20260621/phase0_v4_topk_retrain150_comparison.csv`.

## Route And COM5 Evidence

The current closure ledger records 7 Pareto candidates: 6 route-clean,
five-run COM5 board-claimable rows and 1 route-fail row.

| Candidate | Stage-3 op | WNS ns | actual DSP | COM5 latency ms | Status |
|---|---|---:|---:|---:|---|
| `rl_arch_60` | `skip k1 e1` | 0.223 | 524 | 24.836150 | board-claimable |
| `rl_arch_175` | `skip k1 e1` | 0.022 | 524 | 49.025740 | board-claimable |
| `rl_arch_193` | `mbconv k3 e3` | 0.113 | 612 | 24.872910 | board-claimable |
| `rl_arch_135` | `mbconv k3 e3` | 0.094 | 612 | 49.062010 | board-claimable |
| `rl_arch_154` | `edge k3 e1` | 0.121 | 528 | 39.010840 | board-claimable |
| `rl_arch_169` | `denoise k3 e1` | 0.150 | 528 | 50.550715 | board-claimable |
| `rl_arch_116` | `denoise k3 e1` | -3.915 | 840 | — | route-fail; COM5 blocked |

Board claimability is limited to the recorded bitstream SHA256 and COM5 run
set. COM5 is not the power path and is not full validation-set inference.

Sources:

- `results/phase0_v4_sonar_stage3_k3_board_experiment/hardware_closure_plan.csv`
- `results/phase0_v4_sonar_stage3_k3_board_experiment/evidence_boundary_table.csv`
- `results/phase0_v4_sonar_stage3_k3_board_experiment/phase0_v4_vs_v3_board_comparison.csv`

## Image-Quality Evidence

The NKSID dataset is a classification dataset and does not provide paired clean
and degraded reference images. The recorded dataset-mode run therefore uses
`reference_policy=input_as_reference`.

| Transform | PSNR | SSIM | Interpretation |
|---|---:|---:|---|
| `identity` | inf | 1.000000 | implementation sanity check |
| `denoise` | 33.792236 | 0.913349 | operator-effect/structure preservation |
| `edge` | 15.961825 | 0.277619 | edge-map transformation, not restoration |
| `edge_enhanced` | 28.003025 | 0.936747 | operator-effect/structure preservation |

These values must not be described as clean-reference restoration quality and
must not be combined with `macro_f1` or `top1`.

Reproduce dataset mode:

```powershell
python scripts/measure_sonar_image_quality.py `
  --data-dir data/NKSID `
  --split val `
  --fold 0 `
  --image-size 224 `
  --transforms identity,denoise,edge,edge_enhanced `
  --output-dir results/sonar_image_quality_psnr_ssim_20260622
```

Paired mode is available when true reference/candidate directories exist:

```powershell
python scripts/measure_sonar_image_quality.py `
  --reference-dir <reference_dir> `
  --candidate-dir <candidate_dir> `
  --output-dir results/sonar_image_quality_paired
```

## Ablation Completion Boundary

The four planned variants are `no_sonar`, `denoise_only`, `edge_only`, and
`denoise_edge`. Only `no_sonar` has started, with `3/300` candidates evaluated;
the other three are not run. Every row is currently
`comparison_ready=false`.

Resume anchor:
`results/phase0_v4_sonar_ablation_rl300_20260621/phase0_v4_sonar_ablation_no_sonar_rl300_eval10_seed42/checkpoints/search_state.json`.

The closure packager reads this checkpoint and emits `partial_started` instead
of mislabeling the run as completed:

```powershell
python scripts/phase0_v4_three_lane_closure.py
```

The default command is packaging-only. `--run-retrain` and `--run-hardware`
are explicit long-running actions and are not part of documentation refresh.
