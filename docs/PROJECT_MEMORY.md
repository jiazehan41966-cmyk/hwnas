# Project Memory

Last updated: 2026-07-03

Purpose: this file is the persistent local memory index for the HW-NAS FPGA Sonar repository. New tasks in this workspace should read this file first when they need repository context, audit history, archive paths, or workflow boundaries.

## Archive Index

Primary repository overview:

- `E:\1\hwnas\hwnas\docs\repository_complete_introduction.md`

Static audit archives:

- `E:\1\hwnas\hwnas\docs\audit_structure_discovery.md`
- `E:\1\hwnas\hwnas\docs\audit_entry_config.md`
- `E:\1\hwnas\hwnas\docs\audit_candidate_hardware.md`
- `E:\1\hwnas\hwnas\docs\audit_search_training_metrics.md`
- `E:\1\hwnas\hwnas\docs\audit_deploy_result_consumption.md`

Supporting result/archive policy documents:

- `E:\1\hwnas\hwnas\docs\RUN_STORAGE_AUDIT.md`
- `E:\1\hwnas\hwnas\docs\RESULT_RETENTION_DECISIONS.md`
- `E:\1\hwnas\hwnas\docs\PHASE0_V3_BOARD_RESULTS.md`
- `E:\1\hwnas\hwnas\docs\PHASE0_V3_DEPLOYABLE_SEARCH_EVIDENCE.md`
- `E:\1\hwnas\hwnas\docs\PHASE0_V3_RETRAINED_BOARD_REINJECTION.md`
- `E:\1\hwnas\hwnas\docs\PHASE0_V4_SONAR_OP_PRECHECK.md`
- `E:\1\hwnas\hwnas\docs\PHASE0_V4_SONAR_RESULTS.md`

Concrete local archive outputs:

- `E:\1\hwnas\hwnas\results_archive\2026-04-review\`
- `E:\1\hwnas\hwnas\results_archive\2026-04-review\ARCHIVE_MANIFEST.txt`
- `E:\1\hwnas\hwnas\results\phase0_v4_sonar_op_precheck\`
- `E:\1\hwnas\hwnas\hls_lut_builder\results\phase0_v4_sonar_stage3_k3_pilot\`
- `E:\1\hwnas\hwnas\results\retrain_phase0_v4_sonar_stage3_k3_topk_20260621\`
- `E:\1\hwnas\hwnas\results\phase0_v4_sonar_stage3_k3_board_experiment\`
- `E:\1\hwnas\hwnas\results\phase0_v4_sonar_ablation_rl300_20260621\`
- `E:\1\hwnas\hwnas\results\sonar_image_quality_psnr_ssim_20260622\`

Current Phase0 v3 board-validation outputs:

- `E:\1\hwnas\hwnas\hls_lut_builder\board_harness\results\pareto_route_gate_phase0_v3b_lowdsp_prune6\`
- `E:\1\hwnas\hwnas\hls_lut_builder\board_harness\results\pareto_route_gate_phase0_v3b_lowdsp_prune6\reports\phase0_v3_board_candidate_comparison.md`
- `E:\1\hwnas\hwnas\hls_lut_builder\board_harness\results\pareto_route_gate_phase0_v3b_lowdsp_prune6\evidence_bundles\phase0_v3_claimable_rl_arch_186_first_measure_20260606T083121\`

Current Phase0 v3 retrained-weight board reinjection outputs:

- `E:\1\hwnas\hwnas\hls_lut_builder\board_harness\results\retrain_phase0_v3_board_reinject_20260608\`
- `E:\1\hwnas\hwnas\hls_lut_builder\board_harness\results\retrain_phase0_v3_board_reinject_20260608\evidence_bundles\phase0_v3_retrained_board_reinject_20260608\`
- `E:\1\hwnas\hwnas\hls_lut_builder\board_harness\results\retrain_phase0_v3_board_reinject_20260608\reports\retrained_board_reinject_comparison.md`
- `E:\1\hwnas\hwnas\hls_lut_builder\board_harness\results\retrain_phase0_v3_board_reinject_20260608\reports\retrained_board_reinject_commands.md`
- `E:\1\hwnas\hwnas\hls_lut_builder\board_harness\results\retrain_phase0_v3_board_reinject_20260608\reports\retrained_board_reinject_acceptance_audit.md`
- `E:\1\hwnas\hwnas\hls_lut_builder\board_harness\results\retrain_phase0_v3_board_reinject_20260608\power_measurements\`

## Current Archive Coverage

Covered:

- Repository structure discovery and real functional layering.
- Entry points, config flow, run artifact creation, and legacy wrappers.
- Candidate representation, model construction, and hardware cost mapping.
- Search/training metric propagation across random, RL, and Proxyless paths.
- Deployment, inference, ONNX/INT8/HLS stub artifacts, and result consumption scripts.
- Local result archival policy plus the completed `results_archive/2026-04-review/` move recorded by the retention review.
- Phase0 v3 deployable-search evidence boundaries, including denominator rules and separation of search proxy, retrain, and board-claimable layers.
- Phase0 v3 retrained-weight board reinjection for `rl_arch_186` and `rl_arch_242`, including retrain150 metrics, new routed bitstream SHA256 values, and COM5 deterministic harness-input measurements.
- Phase0 v4 sonar-op precheck evidence roots for the local-only stage3 k3 pilot and launch-readiness package.
- Phase0 v4 retrain150, full-route, five-run COM5, partial-ablation, and
  PSNR/SSIM evidence boundaries, summarized in `docs/PHASE0_V4_SONAR_RESULTS.md`.

Known gap:

- No standalone archived HLS/LUT production-chain audit file was found as of 2026-06-22. Treat HLS/LUT conclusions as partial unless a future file such as `docs/audit_hls_lut_production.md` is added.

Current board-claimable Phase0 v3 result:

- As of 2026-06-06, Phase0 v3 low-DSP route-aware validation has 4 claimable full-network AV7K325 COM5 candidates.
- `accuracy-first`: `rl_arch_186`, macro_f1 `0.629512`, top1 `0.792308`, real board e2e latency `49.062010 ms`, WNS `0.094 ns`, actual DSP `612`.
- `latency-balanced`: `rl_arch_242`, macro_f1 `0.627406`, real board e2e latency `24.872910 ms`, WNS `0.113 ns`, actual DSP `612`.
- `resource-min`: `rl_arch_276`, macro_f1 `0.605372`, real board e2e latency `24.836150 ms`, WNS `0.223 ns`, actual DSP `524`.
- Do not report NAS LUT `latency_ms` as real board e2e latency. Use `docs/PHASE0_V3_BOARD_RESULTS.md` and the comparison report paths above for board-claimable numbers.

Current Phase0 v3 retrained-weight board reinjection result:

- As of 2026-06-08, `rl_arch_186` and `rl_arch_242` have independent retrained-weight board reinjection evidence under `retrain_phase0_v3_board_reinject_20260608`.
- `accuracy-first`: `rl_arch_186`, retrain150 macro_f1 `0.854482`, top1 `0.915385`, weighted_f1 `0.921304`, new bitstream SHA256 `82bcf060b4d2d92597bf4a2ce35d882814fba6fb2d179e8b7cc1749a68e26562`, WNS `0.094 ns`, TNS `0.000 ns`, DSP `612`, COM5 `status_code=0`, cycles `9812402`, real board e2e latency `49.062010 ms`, board checksum `67108608`, board argmax `3`.
- `latency-efficient`: `rl_arch_242`, retrain150 macro_f1 `0.839023`, top1 `0.913462`, weighted_f1 `0.914380`, new bitstream SHA256 `298489a05ec25b9a252ab390868ae228304c89233df02a02b6ffcffad9361eba`, WNS `0.113 ns`, TNS `0.000 ns`, DSP `612`, COM5 `status_code=0`, cycles `4974582`, real board e2e latency `24.872910 ms`, board checksum `33619712`, board argmax `2`.
- These COM5 results are deterministic harness-input board latency/output sanity checks, not full NKSID validation-set board accuracy. The final validation-set accuracy source remains PyTorch retrain150.
- After retrain150, describe `rl_arch_242` as `latency-efficient` or `latency-favored`, not `latency-balanced` or equal-accuracy, unless future evidence supports that stricter claim.
- A compact evidence bundle was frozen at `retrain_phase0_v3_board_reinject_20260608\evidence_bundles\phase0_v3_retrained_board_reinject_20260608`; it copies reports/manifests/measurements/logs/Vivado reports and records bitstream paths plus SHA256 hashes without copying `.bit` files.
- The acceptance audit `retrained_board_reinject_acceptance_audit.json/.md` reports `overall_pass=true`, with 2 root gates and 20 per-candidate gates passing for both `rl_arch_186` and `rl_arch_242`.
- Board-level measured power/energy protocol is now CSV-import based under `retrain_phase0_v3_board_reinject_20260608\power_measurements`; until real external power-meter CSV files are imported and `retrained_board_power_energy_acceptance_audit.json` passes, keep power/energy as `not measured`, and treat Vivado power only as an estimate/proxy.

Current Phase0 v4 sonar result:
- The 2026-06-22 evidence snapshot contains 7 Pareto rows, 6 route-clean and
  five-run COM5 board-claimable rows, and 1 route-fail row.
- Classification search proxy, retrain150 validation, route/COM5, PSNR/SSIM,
  and measured power remain separate evidence layers.
- The four-way sonar ablation is incomplete: only `no_sonar` has started
  (`3/300` evaluated); all rows have `comparison_ready=false`.
- Use `docs/PHASE0_V4_SONAR_RESULTS.md` for exact metrics, artifact paths, and
  non-claim boundaries.

## Read Rules For Future Tasks

For a new repository task:

1. Read this file first.
2. For a high-level repository explanation, read `docs/repository_complete_introduction.md`.
3. For entry/config/runtime questions, read `docs/audit_entry_config.md`.
4. For architecture candidate, model, or hardware-cost questions, read `docs/audit_candidate_hardware.md`.
5. For search, training, metrics, feasibility, or reproducibility questions, read `docs/audit_search_training_metrics.md`.
6. For deployment, inference, result tables, figures, checkpoint schema, or artifact consumers, read `docs/audit_deploy_result_consumption.md`.
7. For storage cleanup or archive policy, read `docs/RUN_STORAGE_AUDIT.md` and `docs/RESULT_RETENTION_DECISIONS.md`.
8. For Phase0 v3 low-DSP route-aware board results, read `docs/PHASE0_V3_BOARD_RESULTS.md`.
9. For Phase0 v3 retrained-weight board reinjection results, read `docs/PHASE0_V3_RETRAINED_BOARD_REINJECTION.md`.
10. For Phase0 v4 sonar search/retrain/route/COM5/image-quality status, read `docs/PHASE0_V4_SONAR_RESULTS.md`.

Do not use memories, assumptions, datasets, metrics, or conclusions from other projects unless the user explicitly asks to connect them.

## Metric Priorities

When working on this project, prioritize:

- `macro_f1`
- `top1`
- `latency_ms`
- `LUT`
- `DSP`
- `BRAM`
- `power_w`
- `feasibility`
- reproducibility through config, seed, and artifact paths

Do not treat `accuracy` as equivalent to `top1` unless the current code path proves that equivalence.

## Periodic Archive Maintenance

Default cadence: weekly, Monday 09:00 Asia/Shanghai.

Maintenance checklist:

1. List `docs/audit_*.md`, `docs/repository_complete_introduction.md`, and storage policy docs.
2. Add any newly created archive file paths to this file.
3. Update `Last updated`.
4. If a new audit changes the repository overview, update `docs/repository_complete_introduction.md`.
5. Keep HLS/Vivado/long training out of this maintenance task unless the user explicitly requests dynamic validation.
6. Record missing expected archive files under `Known gap`.
