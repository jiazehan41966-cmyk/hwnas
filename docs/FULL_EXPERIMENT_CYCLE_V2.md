# Full Experiment Cycle V2

Status: preregistered; Stage 0/G1 is complete at 45/45, while G2, G4, Gate 0,
manual Stage-3 approval, and later hardware/power gates still block a new
claimable search cycle.

## Replacement boundary

The `20260712_nksid_av7k325_full` cycle was stopped after plan review. Its
training patches, CUDA smoke, and G5 E3 outputs are retained as non-formal
debugging or side evidence. They must not be merged into V2 formal results.

## Corrected gate order

1. Freeze source, environment, dataset inputs, and the statistical protocol.
2. Pass the complete project test suite and a CUDA runtime smoke/benchmark.
3. Complete G0 and all 45 G1 baseline cells.
4. Complete G2 independent hardware probes and HLS shortlist coverage.
5. Complete the pre-search G4 anchor on `rl_arch_193/fold1/seed42`.
6. Run Gate 0 in Phase A/B/C order on approved remote GPU resources.
7. After explicit manual approval, launch the matched RL versus multi-objective
   aging-evolution search. Both requested methods must be listed in the approval
   artifact; implementation alone does not approve or unfreeze Stage 3.
   `configs/experiment/stage3_replan_approval.template.json` remains
   `approved:false` and is only a decision template, not gate evidence.
8. Freeze accuracy-first, sonar-robust, and deployment-balanced roles using inner validation and
   hardware evidence only.
9. Consume outer validation exactly once for final reporting; outer results
   cannot replace either candidate.
10. Repeat INT8, parity, HLS, full-route, COM5, and power validation for final
    candidates, then package three deployment roles.

## Compute and storage controls

- Only one formal local GPU job may run at a time.
- G1 uses `num_workers=0` on Windows to avoid interpreter-spawn drift.
- The 1,200-work-unit Gate 0 cannot be launched locally without a successful
  representative GPU benchmark and an explicit compute allocation.
- Every stage writes a completion manifest before the next stage starts.
- The RL/aging comparison uses paired seeds 42/43/44, 200 evaluated candidates
  per method/seed, 3 evaluation epochs per candidate, exclusive GPU use, and
  counterbalanced RL->aging / aging->RL execution order.
- Search cost is reported primarily as parent-observed full-process
  `job_gpu_reserved_hours`; search-call `gpu_reserved_hours` and CUDA-event
  time remain separate diagnostic breakdowns.
- Large checkpoints and HLS/Vivado projects require a storage estimate and
  retention decision before launch; deletion remains a separate approved act.

## Candidate and power rules

- `accuracy-first` maximizes `f_clean` on the full search Pareto front.
- `sonar-robust` maximizes `f_robust` on the full search Pareto front.
- `deployment-balanced` minimizes equal-axis normalized distance to the ideal
  point and is reported as a generalized knee approximation.
- Outer metrics are report-only.
- Power targets the three same-protocol roles. The G4 anchor `rl_arch_193`
  remains a separate reference; role collisions use the next deterministic
  inner-ranked route-clean candidate.

## Evidence boundaries

- Search proxy, retrain, INT8, HLS, route, COM5, image quality, calibration,
  and external power remain separate evidence layers.
- COM5 fixed-input stability is not NKSID board accuracy.
- `input_as_reference` PSNR/SSIM is not clean-reference restoration quality.
- Missing route, board, or meter evidence remains `PENDING` or `NOT_MEASURED`.
