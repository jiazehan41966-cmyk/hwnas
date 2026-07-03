# Phase0 v3 Retrained-Weight Board Reinjection

Last updated: 2026-06-08

This document is the handoff record for reinjecting `retrain150`
checkpoints into isolated Phase0 v3 full-network board harness projects. It is
separate from `docs/PHASE0_V3_BOARD_RESULTS.md`, which remains the authoritative
record for the original 2026-06-06 Phase0 v3 board-claimable evidence.

The 2026-07-03 first-principles audit preserves the retrain150 and COM5
artifacts but treats retrain150 scores as legacy fold-0 validation evidence,
not untouched-test generalization estimates.

## Scope And Guardrails

- Target candidates: `rl_arch_186` and `rl_arch_242`.
- Dataset/software accuracy source: PyTorch `retrain150` on NKSID with
  `input_channels=1`, `image_size=224`, `classes=8`, `seed=42`.
- Board run source: retrained checkpoint weights exported into new isolated
  harness projects under the reinjection result root.
- COM5 board measurements are deterministic harness-input latency/output sanity
  checks, not full NKSID validation-set board accuracy.
- Search `eval10`, PyTorch `retrain150`, and board COM5 are separate result
  layers and must not be compared as the same metric.
- Power is not promoted as a measured board-level metric in this run.
- A CSV-import protocol for measured board-input power/energy has been added,
  but no real power CSV has been imported yet. Until that import is completed,
  power/energy remains `not measured`.
- `rl_arch_242` should be described as `latency-efficient` or
  `latency-favored`, not `equal-accuracy balanced`, because retrain150 no longer
  supports the earlier eval10 closeness claim.

## Key Artifacts

- Reinjection result root:
  `hls_lut_builder/board_harness/results/retrain_phase0_v3_board_reinject_20260608/`
- Compact evidence bundle:
  `hls_lut_builder/board_harness/results/retrain_phase0_v3_board_reinject_20260608/evidence_bundles/phase0_v3_retrained_board_reinject_20260608/`
- Comparison reports:
  - `hls_lut_builder/board_harness/results/retrain_phase0_v3_board_reinject_20260608/reports/retrained_board_reinject_comparison.md`
  - `hls_lut_builder/board_harness/results/retrain_phase0_v3_board_reinject_20260608/reports/retrained_board_reinject_comparison.csv`
  - `hls_lut_builder/board_harness/results/retrain_phase0_v3_board_reinject_20260608/reports/retrained_board_reinject_comparison.json`
- Command/runbook record:
  `hls_lut_builder/board_harness/results/retrain_phase0_v3_board_reinject_20260608/reports/retrained_board_reinject_commands.md`
- Acceptance audit:
  - `hls_lut_builder/board_harness/results/retrain_phase0_v3_board_reinject_20260608/reports/retrained_board_reinject_acceptance_audit.md`
  - `hls_lut_builder/board_harness/results/retrain_phase0_v3_board_reinject_20260608/reports/retrained_board_reinject_acceptance_audit.json`
- Static `.mem` validation summary:
  `hls_lut_builder/board_harness/results/retrain_phase0_v3_board_reinject_20260608/reports/static_mem_validation.json`
- Power/energy CSV import protocol:
  - `hls_lut_builder/board_harness/scripts/import_retrained_board_power_energy.py`
  - `hls_lut_builder/board_harness/results/retrain_phase0_v3_board_reinject_20260608/power_measurements/`
  - `hls_lut_builder/board_harness/results/retrain_phase0_v3_board_reinject_20260608/power_measurements/power_measurement_manifest.template.json`
  - pending measured outputs:
    `hls_lut_builder/board_harness/results/retrain_phase0_v3_board_reinject_20260608/power_measurements/reports/retrained_board_power_energy_summary.*`
    and
    `hls_lut_builder/board_harness/results/retrain_phase0_v3_board_reinject_20260608/power_measurements/reports/retrained_board_power_energy_acceptance_audit.*`
- Evidence bundle manifest:
  `hls_lut_builder/board_harness/results/retrain_phase0_v3_board_reinject_20260608/evidence_bundles/phase0_v3_retrained_board_reinject_20260608/evidence_bundle_manifest.json`
- Evidence bundle SHA256 list:
  `hls_lut_builder/board_harness/results/retrain_phase0_v3_board_reinject_20260608/evidence_bundles/phase0_v3_retrained_board_reinject_20260608/SHA256SUMS.txt`
- Export script:
  `hls_lut_builder/board_harness/scripts/export_retrained_weights_to_e2e_mem.py`
- Report script:
  `hls_lut_builder/board_harness/scripts/retrain_board_reinject_report.py`
- Unit test:
  `tests/test_retrained_weight_export.py`

## Implementation Summary

The reinjection path adds `parameter_mode=trained_checkpoint_mem` to the
full-network board validation wrapper. For each candidate, the flow loads the
`final_best_model.pt` checkpoint, checks that the candidate architecture matches
the checkpoint source, folds BatchNorm into Conv weights, quantizes Conv/Linear
weights to int8, exports int32 biases, and writes harness-compatible `.mem`
files according to the existing bank layout. It then generates:

- `trained_weight_manifest.json`
- `software_board_parity.json`
- new routed bitstream
- SHA256-locked COM5 measurement outputs
- summary reports in Markdown, JSON, and CSV

The software parity JSON is retained only as deterministic harness-input sanity.
It is not used as a full numerical equivalence claim against the routed FPGA
implementation.

The compact evidence bundle copies reports, manifests, measurement outputs,
logs, and routed timing/utilization reports. It does not copy `.bit` files; the
bundle manifest records each bitstream's absolute path and SHA256 hash.
When measured power CSV files are present, the bundle script also copies the
power manifest, raw CSV files, power summary, and power acceptance audit into
`power_measurements/`.

## Final Comparison

| Arch | Positioning | retrain macro_f1 | retrain top1 | retrain weighted_f1 | WNS ns | TNS ns | LUT | DSP | BRAM | Bitstream SHA256 | COM5 status_code | cycles | real_e2e_ms | Board checksum | Board argmax | COM5 runs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `rl_arch_186` | accuracy-first | 0.854482 | 0.915385 | 0.921304 | 0.094 | 0.000 | 18598 | 612 | 127.5 | `82bcf060b4d2d92597bf4a2ce35d882814fba6fb2d179e8b7cc1749a68e26562` | 0 | 9812402 | 49.06201 | 67108608 | 3 | 5 |
| `rl_arch_242` | latency-efficient | 0.839023 | 0.913462 | 0.914380 | 0.113 | 0.000 | 18553 | 612 | 127.5 | `298489a05ec25b9a252ab390868ae228304c89233df02a02b6ffcffad9361eba` | 0 | 4974582 | 24.87291 | 33619712 | 2 | 5 |

## Power/Energy Measurement Status

| Arch | rail_scope | measured power status | dynamic_energy_mj | notes |
|---|---|---|---:|---|
| `rl_arch_186` | `board_input_total` | not measured |  | CSV import protocol ready; no real power CSV imported yet |
| `rl_arch_242` | `board_input_total` | not measured |  | CSV import protocol ready; no real power CSV imported yet |

The only acceptable measured-power source for this section is an external
power-meter CSV imported through
`import_retrained_board_power_energy.py`. Vivado power reports remain
implementation estimates/proxies and must not be reported as measured board
power or measured energy.

Both candidates passed the retrained-weight board reinjection check for this
deterministic harness input:

- full-route completed successfully
- WNS is non-negative
- TNS is `0.000 ns`
- actual DSP is `612`, within the `<=700` route-aware gate
- bitstream SHA256 was locked during COM5 measurement
- all 5 COM5 runs returned `status_code=0`
- cycles and checksum were stable across the 5 COM5 runs

The acceptance audit also passed overall. It checks 2 root gates and 20
per-candidate gates for each of `rl_arch_186` and `rl_arch_242`, including
architecture presence, `parameter_mode=trained_checkpoint_mem`, checkpoint hash
presence, export count, full-route status, timing, DSP limit, actual bitstream
SHA256, COM5 status, 5-run stability, cycles/checksum stability, and positive
latency.

## Interpretation

`rl_arch_186` remains the accuracy-first candidate because it has the highest
PyTorch retrain150 `macro_f1`, `top1`, and `weighted_f1` among the two
independently reinjected candidates.

`rl_arch_242` remains the latency-efficient candidate because its routed
retrained-weight bitstream keeps the same board-level latency class as the
original low-DSP claimable result: `24.87291 ms`, about half of
`rl_arch_186`'s `49.06201 ms`. It should not be described as
equal-accuracy balanced unless a future retrain or board-accuracy protocol
supports that claim.

## Validation

- `python -m py_compile` passed for the new/extended board-reinjection scripts.
- `python -m pytest tests/test_retrained_weight_export.py -q` passed with
  `3 passed`.
- `python -m pytest tests/test_retrained_board_power_energy.py -q` covers the
  CSV-import power/energy protocol.
- `retrained_board_reinject_acceptance_audit.json` reports `overall_pass=true`.
- Static `.mem` validation passed for both candidates:
  - 12 export entries each
  - 122 `.mem` files each
  - 0 missing files
  - 0 SHA256 mismatches
  - 0 depth mismatches

## Non-Claims

This run does not claim:

- full validation-set board accuracy
- measured board power
- that software proxy checksum/argmax exactly matches FPGA output for all
  inputs
- route-seed-agnostic timing robustness beyond the recorded Vivado runs
- any update to the original 2026-06-06 Phase0 v3 board evidence
