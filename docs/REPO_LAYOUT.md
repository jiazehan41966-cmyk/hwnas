# Repository Layout

This document is the canonical map of the repository after the cleanup pass on
branch `codex/repo-cleanup-review`.

## Active Top-Level Entry Points

- `run_backbone_baseline.py`
  Macro-architecture screening and anchor export.
- `run_search.py`
  Unified search entrypoint for random, RL, and Proxyless-style runs.
- `run_search_space_probe.py`
  Search-space feasibility probing without training.
- `run_retrain.py`
  Final retraining for a selected searched architecture.
- `run_export.py`
  Export, quantization, and deployment-facing artifacts.
- `run_operator_ablation.py`
  Operator ablation experiments tied to macro templates / anchor roles.
- `run_rl_search.py`
  Thin compatibility wrapper that forwards to `run_search.py --search-method rl`.
- `scripts/measure_sonar_image_quality.py`
  PSNR/SSIM/MSE reporting for dataset-transform analysis or paired references.
- `scripts/phase0_v4_three_lane_closure.py`
  Packaging-only by default; reconciles search, retrain, route, COM5, and
  ablation evidence without launching long-running work unless explicitly asked.

## Core Source Tree

- `src/hwnas_fpga/data`
  Dataset loading, transforms, and fold handling.
- `src/hwnas_fpga/models`
  Backbone baselines, searchable block builder, and Proxyless supernet code.
- `src/hwnas_fpga/search_space`
  Search-space configuration, sampling, pruning, and probe utilities.
- `src/hwnas_fpga/hardware`
  FPGA board profiles, analytical estimators, LUT support, and report parsing.
- `src/hwnas_fpga/metrics`
  Reusable image-quality metrics kept separate from classification metrics.
- `src/hwnas_fpga/search`
  Search algorithms and Pareto utilities.
- `src/hwnas_fpga/training`
  Training and retraining helpers.
- `src/hwnas_fpga/deploy`
  Export, inference, quantization, and HLS-facing utilities.
- `src/hwnas_fpga/runtime.py`
  Shared runtime assembly helpers used by the top-level scripts.
- `src/hwnas_fpga/experiment.py`
  Experiment artifact tracker.

## Configuration Layout

- `configs/backbone_baseline*.yaml`
  Backbone screening protocols.
- `configs/search/*.yaml`
  Active search configs that back the current macro-screening, A1, random,
  RL, and ProxylessNAS workflows.
- `configs/search/legacy/*.yaml`
  Historical one-off, budget-comparison, alignment, and smoke configs kept for
  provenance but not part of the current recommended workflow.
- `configs/hardware/*.yaml`
  Hardware manifests and LUT build inputs.

## Results and Artifact Policy

- `results/`
  Keep by default. Do not delete casually. Review each run for one of:
  `formal`, `baseline`, `smoke`, `debug`, `restart`, `viz`, or `stale`.
- `artifacts/`
  Structured exports and LUT tables that support reproducible experiments.
- `reference/`
  Retained on purpose. These repositories are design references, not part of
  the active runtime, but should remain available for provenance.

## Legacy Area

- `scripts/legacy/`
  Historical scripts that were moved out of the repository root to reduce
  clutter while preserving provenance.
- `docs/legacy/`
  Superseded planning notes that are useful for audit trails but no longer
  represent the recommended execution order.
- `configs/search/legacy/`
  Historical search configs that should not be selected by default.

## Immediate Cleanup Rules

- Keep `reference/` intact.
- Do not delete `results/` until each directory has been reviewed.
- Prefer thin compatibility wrappers over abrupt removal of old entrypoints.
- Prefer one canonical maintained path for each workflow:
  search -> retrain -> export.
