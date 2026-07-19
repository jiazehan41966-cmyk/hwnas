# T6 grouped-proxy prediction contract patch: 2026-07-17

## Scope

This patch closes the missing prediction-side evidence interface for T6 without creating any real HLS/route result. The current G1 run remains source-frozen and uninterrupted. All executable additions use the `.txt` runtime-evidence convention and live outside the frozen source snapshot.

## Frozen interface

- Four methods: `analytic_lut`, `linear_regression`, `gradient_boosting`, `harp_gnn`.
- Thirteen targets: HLS cycles/II/LUT/DSP/BRAM/FF and route WNS/TNS/achieved clock/LUT/DSP/BRAM/FF.
- Split: five-fold leave-one-architecture-family-out, with exact train/test sample lists and each family held out once.
- Prediction key: one and only one `(sample_id, method_id, target_name)` row for every complete truth sample.
- Provenance: predictor artifact/config, fold manifest, feasibility config, truth CSV, project commit/code state and source-freeze SHA are all file- or identity-bound.
- HARP ownership is explicit (`paper_id=harp_2023`); project baselines use `paper_id=project_internal`. Author-paper metrics are not accepted as local predictions.

## Feasibility rule

The policy keeps board capacity separate from claimability. AV7K325 physical capacity is LUT 203800, FF 407600, BRAM 445 and DSP 840. Formal route feasibility uses LUT≤203800, FF≤407600, BRAM≤445, DSP≤700, WNS≥0 and achieved clock≥200 MHz. The auditor recomputes both measured and predicted feasibility; submitted boolean labels are never trusted directly.

## Test evidence

The synthetic-only contract test created 100 complete truth rows across five families and 5,200 held-out prediction rows (`100 × 4 × 13`). The positive case produced 52 method-target metric rows and four method-level feasibility summaries. Four injected failures were rejected:

1. duplicate prediction key;
2. missing prediction key;
3. measured value inconsistent with truth;
4. predicted feasibility label inconsistent with recomputed limits.

The retained summary is `results/benchmarks/ccf_ab_nksid_av7k325_v1/contract_tests/hls_proxy_v1/contract_test_summary.json`, SHA256 `b314cbd876715999cf66c862cf433e4b273a57487931fd40f101af2d32f17661`.

## Runtime artifact hashes

| Artifact | SHA256 |
|---|---|
| `hls_proxy_prediction_template.csv` | `b9974f6307dcc3af5dfb0aee965a503865a9795f28669e0df1b019e3c1916bac` |
| `hls_proxy_fold_manifest_template.json.txt` | `6cbbeb2c8fa5a09d1e15b2643c24efc837ad20b2ff846bd64dcc4814b9dfbc16` |
| `hls_proxy_feasibility_template.json.txt` | `ad56219a153183a9e1b9b2c0f6edf3c8d5050125aedec6a60d285a79f364e0b5` |
| `audit_hls_proxy_predictions.txt` | `768b8f171f0447e2ea1e25714d069873ef8deda522661aea65b1977087900e56` |
| `test_hls_proxy_contract.txt` | `6cde088dbd4702ef73136ad6eb5a47aeec8997ede04dd18f8eba01ff451d6603` |

## Evidence boundary

This contract test increments the real T6 denominator by exactly zero. T6 and F4 remain `PENDING` until at least 100 semantic-safe complete-network project rows and their real held-out predictions pass the same auditor. Passing the proxy contract does not establish route-feasible deployment, COM5 latency, board accuracy or external-instrument power.
