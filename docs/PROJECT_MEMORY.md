# Project Memory

Last updated: 2026-05-30

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

## Current Archive Coverage

Covered:

- Repository structure discovery and real functional layering.
- Entry points, config flow, run artifact creation, and legacy wrappers.
- Candidate representation, model construction, and hardware cost mapping.
- Search/training metric propagation across random, RL, and Proxyless paths.
- Deployment, inference, ONNX/INT8/HLS stub artifacts, and result consumption scripts.

Known gap:

- No standalone archived HLS/LUT production-chain audit file was found as of 2026-05-30. Treat HLS/LUT conclusions as partial unless a future file such as `docs/audit_hls_lut_production.md` is added.

## Read Rules For Future Tasks

For a new repository task:

1. Read this file first.
2. For a high-level repository explanation, read `docs/repository_complete_introduction.md`.
3. For entry/config/runtime questions, read `docs/audit_entry_config.md`.
4. For architecture candidate, model, or hardware-cost questions, read `docs/audit_candidate_hardware.md`.
5. For search, training, metrics, feasibility, or reproducibility questions, read `docs/audit_search_training_metrics.md`.
6. For deployment, inference, result tables, figures, checkpoint schema, or artifact consumers, read `docs/audit_deploy_result_consumption.md`.
7. For storage cleanup or archive policy, read `docs/RUN_STORAGE_AUDIT.md` and `docs/RESULT_RETENTION_DECISIONS.md`.

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
