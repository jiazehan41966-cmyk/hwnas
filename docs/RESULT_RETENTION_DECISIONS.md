# Result Retention Decisions

Date: 2026-04-17

This document reviews the `results/` directories that were tagged
`archive_candidate` in [RUN_STORAGE_AUDIT.md](RUN_STORAGE_AUDIT.md).

Scope of this pass:

- `reference/` remains untouched.
- No `results/` directories were deleted in this review.
- The goal here is to decide which runs should stay local as representative
  examples and which ones are safe to archive out of the main workspace.

Archival action completed:

- On 2026-04-17, the 28 directories listed in the `Archive` section were moved
  from `results/` to `results_archive/2026-04-review/`.
- A local manifest was written to
  `results_archive/2026-04-review/ARCHIVE_MANIFEST.txt`.

Review criteria:

- Is the directory referenced by active scripts, docs, or tests?
- Does it hold a unique artifact that is not reproduced elsewhere?
- Does it serve as a useful representative smoke artifact for a maintained
  workflow?
- Is it clearly a restart, debug, probe, or visualization output that can be
  regenerated?

Review result:

- Keep local: 4 directories, about 4.17 MB total
- Archive: 28 directories, about 74.96 MB total

Important note:

- The only current code references found for these archive candidates are in
  [RUN_STORAGE_AUDIT.md](RUN_STORAGE_AUDIT.md) and two legacy plotting scripts
  under [scripts/legacy](../scripts/legacy).
- No active entrypoint, maintained script, or test currently depends on these
  result directories by name.

## Keep Local

These are the representative smoke outputs worth keeping in the main workspace.

| Directory | Why keep it |
| --- | --- |
| `results/backbone_nksid_av7k325_smoke` | Canonical smoke artifact for the maintained macro-backbone screening flow driven by [run_backbone_baseline.py](../run_backbone_baseline.py). |
| `results/smoke_random_mobile_anchor_macrofi` | Canonical smoke artifact for the current random search path on the `mobile_anchor` space; useful as the smallest end-to-end sample that records `macro_f1` and FPGA cost together. |
| `results/smoke_proxyless_dummy_20260327_v2` | Canonical smoke artifact for the Proxyless path. This is the better representative of the two dummy Proxyless smoke runs, so `v2` stays and the older one can go. |
| `results/operator-ablation-from-selected-pool-smoke` | Best representative smoke artifact for the maintained operator-ablation flow because it exercises the selected-backbone-pool path instead of a narrower one-off config path. |

## Archive

These runs are safe to move out of the main workspace because they are either
superseded, restart/probe/debug outputs, or derived visualizations.

### Superseded Restart / Probe Runs

| Directory | Reason |
| --- | --- |
| `results/backbone_nksid_restart_20260328_150005` | Restart-era macro comparison output. Superseded by the later formal structured backbone artifacts in [artifacts/backbone_baseline_nksid_av7k325_formal_structured_20260329_011516](../artifacts/backbone_baseline_nksid_av7k325_formal_structured_20260329_011516). |
| `results/runtime_probe_cpu_nksid_ep1_e5` | One-off runtime probe with 1 evaluated candidate; diagnostic only. |
| `results/runtime_probe_cpu_nksid_ep1_e1` | One-off runtime probe with 1 evaluated candidate; diagnostic only. |
| `results/runtime_probe_cpu_nksid_ep3_e1` | One-off runtime probe with 3 evaluated candidates; diagnostic only. |
| `results/runtime_probe_cfg_ep3_e1` | Config probe variant; diagnostic only. |
| `results/runtime_probe_cfg_ep1_e5` | Config probe variant; diagnostic only. |
| `results/runtime_probe_cfg_ep1_e1` | Config probe variant; diagnostic only. |
| `results/operator_ablation_runtime_probe_20260417` | Incomplete runtime probe for operator ablation; no durable result payload worth keeping local. |

### Smoke / Debug Runs With Better Replacements

| Directory | Reason |
| --- | --- |
| `results/fbnet-like-smoke` | Early backbone smoke run; redundant once `results/backbone_nksid_av7k325_smoke` is kept as the macro smoke sample. |
| `results/operator-ablation-config-smoke` | Narrow config-only smoke; redundant once the selected-pool smoke is retained. |
| `results/operator-ablation-auto-pool-smoke` | Intermediate operator-ablation smoke; redundant once the selected-pool smoke is retained. |
| `results/smoke_nksid_rl_cpu_runnable` | Tiny RL smoke for CPU-runnable debugging; superseded by current formal RL outputs and test coverage. |
| `results/smoke_proxyless_dummy_20260327` | Older of the two dummy Proxyless smoke runs; superseded by `smoke_proxyless_dummy_20260327_v2`. |
| `results/lightweight-sonar-av7k325-smoke` | Profile smoke only; redundant with current A1 probe and formal search outputs. |
| `results/mobile-anchor-av7k325-smoke` | Profile smoke only; redundant with current A1 probe and formal search outputs. |
| `results/small-av7k325-post-pool-smoke` | Profile smoke only; not part of the current main experiment path. |
| `results/family-profile-smoke` | Profile smoke only; not part of the current main experiment path. |
| `results/small-profile-smoke` | Profile smoke only; not part of the current main experiment path. |
| `results/official_proxyless_native_debug3` | Debug log stub only; no structured results. |
| `results/official_proxyless_run_debug` | Debug log stub only; no structured results. |
| `results/official_proxyless_run_debug2` | Empty debug directory; no retained value. |

### Derived Visualization Outputs

| Directory | Reason |
| --- | --- |
| `results/search_strategy_budget200_av7k325_small64_viz` | Derived figures only; reproducible from underlying runs and legacy plotting scripts. |
| `results/quick_mixed_compare_viz` | Derived figures only; reproducible. |
| `results/search_strategy_budget200_av7k325_full_viz` | Derived figures only; reproducible. |
| `results/search_strategy_selection_viz` | Derived figures only; referenced only by a legacy plotting script. |
| `results/search_strategy_budget200_av7k325_rl_vs_proxyless_official_viz` | Derived figures only; reproducible. |
| `results/_viz_check` | Generic visualization check output; derivable from maintained visualizer. |
| `results/rl_vs_proxyless_heavy_budget_viz` | Derived figure bundle; referenced only by a legacy plotting script and reproducible from source runs. |

## Current Local Set

Kept in `results/`:

- Keep the 4 directories listed in the `Keep Local` section in `results/` as
  representative smoke outputs for the maintained workflows.

This review does not recommend deleting any non-archive-candidate directories.
