# Legacy Scripts

This directory keeps historical entrypoints that are no longer part of the
active experiment flow, but may still be useful for provenance or one-off
inspection.

Archived scripts:

- `run_full_nas_legacy.py`
  The original monolithic end-to-end pipeline. Replaced by the modular
  `run_search.py -> run_retrain.py -> run_export.py` flow.
- `retrain_best_legacy.py`
  Older retraining entrypoint with its own argument contract. Replaced by
  `run_retrain.py`.
- `visualize_results_hardcoded.py`
  Older plotting script with hard-coded result paths. Replaced by the generic
  `scripts/visualize_results.py`.
- `visualize_search_strategy_experiment.py`
  Historical comparison visualizer tied to older named runs.
- `plot_rl_vs_proxyless_heavy_budget.py`
  Historical figure generator for one specific heavy-budget comparison.

Retention policy:

- Keep these files for traceability.
- Do not use them for new experiments.
- If an archived script becomes needed again, restore its behavior through the
  maintained entrypoints instead of expanding legacy code further.
