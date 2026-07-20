# G2 A8 to T6 admission audit: 2026-07-17

## Decision

The eight corrected A8 rows are valid G2 calibration and full-network route/COM5 evidence, but their T6 formal sample-count increment is `0`. They must not be copied into the T6 complete-network HLS/route ledger as if each row had a full-network HLS synthesis.

## Source cohort

- Corrected ledger: `results/experiment_cycle_20260712_v2/calibration_v2_a8_corrected/full_network_evidence.jsonl`
- Ledger SHA256: `11ecb6c3872eaef2f88066e1a6c6c527044978e3102fb471f9a3d9d7b4891d52`
- Evidence audit status: `PASS_READY_FOR_CALIBRATION_REBUILD_CORRECTED`
- Rows: 8, comprising four fit rows and four independent probes.
- Unique architecture/fingerprint count: 8/8.
- Architecture families: one, `mainline_mbconv_skip`.
- Source-freeze verification: `PASS`, 508/508 files for the A8 cohort.

## Evidence classification

| T6 field or invariant | A8 evidence | Admission |
|---|---|---|
| Canonical architecture hash | present | usable |
| Candidate-specific full-network route | present, timing-clean, routed bitstream | usable as route-side auxiliary evidence |
| Route LUT/DSP/BRAM/FF | present | usable |
| Route WNS | present | usable |
| Route TNS | not serialized in the corrected ledger | missing for T6 row |
| COM5 cycles/latency/status/checksum | present with five-run stability audits | usable as board-side auxiliary evidence |
| Full-network HLS source hash | absent | fail |
| One full-network HLS top csynth report | absent | fail |
| HLS cycles/LUT/DSP/BRAM | present only as a sum of six per-operator csynth entries | composite estimate, not full-network csynth |
| HLS FF | absent from the aggregate | fail |
| Network-level HLS II | absent | fail |
| HLS failure-category and elapsed-time contract | absent | fail |
| `claimability_status=CLAIMABLE` in T6 schema | absent | fail |
| At least five architecture families | one family | fail |
| Formal sample threshold | 0 admitted of required 100 | fail |

## Why the HLS evidence is not a full-network sample

For each candidate, `candidate_hls_report.json` maps the stem, stage blocks, global average pool and classifier to separate cached operator kernels. Its reported cycles and resources are arithmetic aggregates over those component csynth XML files; a structural skip contributes zero. There is no single candidate-specific HLS top function, candidate-specific generated full-network HLS source, or one csynth report that captures inter-layer FIFOs, scheduling, sharing and network-level initiation interval.

The downstream Vivado harness is candidate-specific and routes as a full network, so the route and COM5 layer remains legitimate. That does not retroactively turn the component-sum HLS estimate into a complete-network HLS measurement.

## Permitted reuse

- Keep the eight rows in the corrected G2 calibration dataset and retain its independent quality conclusions.
- Use route resources, WNS and COM5 latency only under their existing A8 source-freeze and latency-only claim boundary.
- Use component HLS aggregates as a separately labelled analytic/composite predictor input when studying why LUT/BRAM/latency calibration failed.
- Do not include these rows in grouped T6 cross-validation, HARP training, the T6 sample denominator or the formal `<30`, `30-99`, `>=100` threshold.

## Required collector for an admissible T6 row

Each future row must bind one semantic-safe complete-network candidate to:

1. Candidate and canonical architecture SHA256.
2. Generated full-network HLS source tree SHA256 and generation-config/command SHA256.
3. One candidate-specific HLS top csynth report SHA256 with cycles, latency, II, LUT, DSP, BRAM and FF.
4. Candidate-specific route report SHA256 with LUT, DSP, BRAM, FF, WNS, TNS, achieved clock, routing status and failure category.
5. Vivado/Vitis HLS versions, target `xc7k325t-ffg900-2`, clock policy, start/end timestamps and elapsed time.
6. Semantic-equivalence result and exact failure-stage classification; failed HLS/route attempts remain ledger rows.
7. Architecture-family label suitable for grouped five-fold validation.
8. Explicit `claimability_status`; only `CLAIMABLE` rows with hashes recomputed from disk enter the formal count.

## Gate state

T6 and F4 remain `PENDING`. The corrected A8 cohort improves understanding of the current proxy but does not reduce the formal complete-network requirement from 100 and does not close G2.
