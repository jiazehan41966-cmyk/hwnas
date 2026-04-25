Historical search configs live here.

They are kept for provenance and backtracking, not because they represent the
current recommended experiment path.

Typical contents:

- smoke / debug configs
- old 120-budget baselines
- CPU-runnable or MIT-aligned comparisons
- one-off strategy comparison sweeps
- superseded generic `nksid_fpga_search*.yaml` entry points retained only for
  provenance after the MobileNetV2 mainline switch on 2026-04-18

Current active configs stay one level up in `configs/search/`.
