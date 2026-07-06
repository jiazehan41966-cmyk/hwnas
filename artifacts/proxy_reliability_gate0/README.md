# Proxy Reliability Gate 0 artifact contract

This tracked directory contains schemas and protocol documentation only.
Large manifests, per-work observations, checkpoints, and analysis figures stay
under the ignored `results/proxy_reliability_gate0/` tree.

Current status on 2026-07-04:

- implementation and seven targeted tests: complete;
- real NKSID NASWOT smoke: complete;
- formal 48 × 5 × 5 × 6 work matrix: not run;
- Gate 0 decision: `not_ready`.

Execution update on 2026-07-05:

- v1 remains frozen at 48 candidates / 7,200 budget-level units and was not
  formally started;
- v2 uses one 150-epoch prefix trajectory per architecture/fold/seed,
  yielding 1,200 work units;
- v2 phases are 240 signal-discovery, 192 fold-robustness, and 768 full
  confirmation trajectories;
- formal v2 observations remain `0/1200`; CPU benchmarks are non-evidence.
- three CPU 1-epoch benchmarks completed in 47.51 / 58.68 / 80.07 seconds
  for minimum / median / maximum analytic-latency candidates; all are marked
  `formal_eligible=false` and performed no outer evaluation.

Templates:

- `classification_observations.template.csv`: long classification input;
- `hardware_observations.template.csv`: separate hardware truth input.

Frozen compact evidence:

- `manifest_summary.json`: manifest fingerprint, protocol/config hashes,
  sampling strata, and architecture/hardware range coverage. The full
  7,200-row work matrix remains under `results/`.
- `manifest_summary_v2.json`: optimized prefix manifest fingerprint, staged
  counts, batch alignment, and epoch reduction.

See `docs/PROXY_RELIABILITY_AUDIT.md` for the frozen design and commands.
