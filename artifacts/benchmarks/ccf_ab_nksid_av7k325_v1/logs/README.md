# Verification log boundary

`pytest_full_stdout.log` records the latest authoritative foreground verification
summary. The command `python -m pytest -q` completed with `473 passed, 1 warning,
28 subtests passed` in 119.76 seconds on 2026-07-15.

The campaign-level machine-readable smoke audit is
`../manifests/integration_smoke.json`; the repository Gate ledger is
`../../../measurement_first_rebuild/status.json`. Test success validates code
and contracts only; it does not change scientific Gate or claimability status.
