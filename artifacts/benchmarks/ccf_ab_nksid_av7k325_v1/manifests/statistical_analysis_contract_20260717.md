# Statistical analysis contract audit: 2026-07-17

## Scope and claim boundary

This card audits the prespecified statistical machinery for T2, T3, T4 and T9. It does not contain a cross-method result. Formal analysis remains blocked until every compared method has the same 15 claimable fold-seed units and the independent file/provenance audits pass.

## Frozen comparison unit and questions

- Unit of analysis: one paired `(outer_fold, seed)` unit; five folds x seeds 42, 43 and 44 give `n=15` per method.
- Primary closed-set question: difference in macro-F1 between each pair of the four closed-set methods.
- Open-set questions: paired differences for known-class macro-F1, NMA, OSFM, OSCRmac, unknown AUROC and FPR95.
- Robustness question: paired differences in the prespecified normalized F1-SNR AUC for additive-noise and speckle families. Clean, blur and contrast conditions remain separately labelled and are not silently folded into SNR.
- Direction is serialized as `left_minus_right`; lower-is-better metrics must be interpreted with their declared direction rather than by sign alone.

## Prespecified inference

- Descriptive reporting: all 15 values, `mean +/- sample standard deviation`, sample count and the fold-seed identifiers.
- Uncertainty: 10,000-iteration paired bootstrap, resampled independently within each outer-fold stratum and then pooled. The seed is derived deterministically from the experiment-family and comparison identifiers.
- Hypothesis test: two-sided paired sign-flip permutation test of the mean difference. With `n=15`, all `2^15 = 32,768` sign assignments are enumerated, so the formal test is exact rather than a 10,000-draw approximation.
- Effect size: paired Cohen's `dz = mean(pairwise difference) / SD(pairwise difference)` when the difference SD is nonzero.
- Multiplicity: Holm step-down family-wise correction within the declared experiment family. Raw and adjusted p-values must both be retained; nonsignificant comparisons are not removed.
- Analysis rows remain `PENDING_G1_LEDGER` until the measurement-first gate ledger permits their release.

## Binding and fail-closed rules

- Builders key every method by the complete `(fold, seed)` pair set and reject missing or unpaired units before inference.
- Closed-set, open-set and corruption builders independently validate the method, run fingerprint, source-freeze binding and current prediction/checkpoint hashes before reading metrics.
- Summary-only values, author-reported paper numbers, smokes and unmatched seeds cannot enter the formal paired analysis.
- Statistical significance does not substitute for practical magnitude: every formal contrast must retain mean difference, 95% interval and Cohen's dz alongside the p-values.

## Verification executed

- Command: `D:\software\python\python.exe -m pytest -q -p no:cacheprovider tests/test_benchmark_statistics.py tests/test_benchmark_metrics.py`
- Environment controls: `PYTHONDONTWRITEBYTECODE=1`; project `src` supplied through `PYTHONPATH`; the active CUDA training environment was not modified.
- Result: `10 passed in 0.88s`.
- Covered contracts include deterministic stratified paired bootstrap, exact 15-unit permutation, monotone Holm adjustment, rejection of unpaired arrays, known exact-HV fronts, dominated/out-of-reference HV points, objective direction, Pareto coverage/NDCG, calibration summary and open-set unknown separation.

## Source bindings

- `src/hwnas_fpga/benchmarks/statistics.py`: `f8c9b2487ac0fac4eddccba85637414a55b60c3a43fde9b72621d2d49c607630`
- `tests/test_benchmark_statistics.py`: `4df99dca5eeea9d30c23eb9cd44caf9ddc09c3d1a1ddb44fdadbc9ebcd06cc14`
- `tests/test_benchmark_metrics.py`: `708831021badc9591e8949d98988504c3ef383a76e8be41732403e6bc2cb3f6f`
- `build_closed_classification_artifacts.txt`: `d761ba16cb430b8daf2c1b9de7fe38cee58c8b775529be041b2834f797cd24fc`
- `build_open_set_artifacts.txt`: `0f202516ca10e1476ff3d1a411c7a22784795f4538dcaa0d0cfc0948ace08a48`
- `build_sonar_robustness_artifacts.txt`: `d4e6c88646a2fecc7efd7ee1a0ba6c934251f8084a09faaa976c308b720e468d`

## Remaining analysis gate

The machinery is contract-tested, but T2/T3/T4/T9 statistics are not yet available. They require all applicable 15-unit method cohorts, independent audits and regenerated source-data tables. No p-value or effect-size claim is authorized by this card alone.
