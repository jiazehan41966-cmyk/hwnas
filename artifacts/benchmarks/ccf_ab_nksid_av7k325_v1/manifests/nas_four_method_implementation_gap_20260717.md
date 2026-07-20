# Four-method NAS implementation gap and amendment: 2026-07-17

## Decision

T5 must not start from the current two-method launcher. The formal target remains Random, RL, Aging Evolution and an HW-PR-NAS paper-spec adapter at 300 actual evaluator calls for each seed 42-51. A new source freeze is required after the missing four-method implementation is added and before any formal run starts.

The fourth method must be labelled `hw_pr_nas_paper_spec_low_data_adapter`, not `HW-PR-NAS author reproduction`. The pinned official repository does not contain an executable author release, and the project's 300-call budget is below the paper's stated data expectation for a new search space.

## Current executable coverage

| Requirement | Current evidence | Status |
|---|---|---|
| Random search | `run_search.py --search-method random` is implemented | available as an individual method |
| RL search | `run_search.py --search-method rl` is implemented | available as an individual method |
| Aging Evolution | `run_search.py --search-method aging_evolution` is implemented | available as an individual method |
| HW-PR adapter | Local Pareto-rank/ListMLE fit exists only in `src/hwnas_fpga/benchmarks/hwpr.py` | smoke only; no acquisition/search loop |
| Matched launcher | `scripts/run_aging_vs_rl_benchmark.py` launches only RL and Aging, with default budget 200 | insufficient for T5 |
| Comparison packager | `scripts/compare_search_methods.py` aggregates arbitrary methods, but inferential comparison is hard-coded to Aging minus RL | insufficient for four-method T9/T5 inference |
| Formal output | `results/benchmarks/ccf_ab_nksid_av7k325_v1/formal/search_comparison/comparison.json` | absent |

## Author-source audit

- Official repository: `https://github.com/IHIaadj/HW-PR-NAS`, pinned commit `296c6576fbae2b277e56c704ff3b6e648ec4c2be`.
- The remote exposes only `main`, `add-license-1` and tag `v0.1.0`; no hidden release branch contains the missing predictor modules.
- README-listed `base_surrogate.py`, accuracy/latency/energy predictor files are absent.
- `search_algo.py` calls an undefined `valid_loss()` and implements a single-objective tournament loop rather than a callable Pareto-rank surrogate search.
- `test.py` passes an argparse namespace to an incompatible `evolution_search` signature.
- The present local helper implements the paper's Pareto-rank target and listwise score ordering with a local tabular encoding and three-layer MLP. It does not reproduce the paper's concatenated architecture-features, GCN and LSTM encoder.

## Paper-supported components

The primary paper describes one unified predictor that scores architectures by Pareto rank rather than independently regressing each objective. It specifies architecture features plus GCN and LSTM encodings, a three-layer fully connected predictor, Pareto fronts obtained by nondominated sorting, and a listwise ranking loss. It reports batch size 18 and states that a new search space generally needs ground-truth evaluation of at least 500 architectures for predictor training.

Primary sources:

- `https://doi.org/10.1145/3579853`
- `https://research.ibm.com/publications/multi-objective-hardware-aware-neural-architecture-search-with-pareto-rank-preserving-surrogate-models`
- Author-hosted manuscript: `https://www.uphf.fr/LAMIH-intra/site/publications/210242.pdf`

## Equal-budget project amendment

The following amendment is required to make the B-class comparison executable without hiding extra evaluations:

1. Count every call that produces `f_clean`, `f_robust` or latency for a new architecture against the same 300-call budget. Rejected proposals, duplicate proposals and failed calls remain separately counted.
2. Pilot HW-PR with 18 initial evaluated architectures followed by 32 surrogate-guided calls, so the 50-call pilot actually tests acquisition behavior. Formal HW-PR uses 50 initial evaluated architectures followed by 250 guided calls.
3. Compute the three project objectives exactly as the other methods do: `1-f_clean`, `1-f_robust`, and `latency/latency_limit`, while LUT/DSP/BRAM remain hard feasibility constraints.
4. Fit/update Pareto-rank targets only from already evaluated architectures. Unevaluated candidates may be scored by the surrogate but never contribute ground-truth metrics or HV.
5. Use deterministic, hash-recorded initial designs and proposal streams. The paper's Latin-hypercube idea must be adapted explicitly for the project's discrete stage/block variables; the transformation and its seed must be archived.
6. Serialize surrogate training rows, predicted ranks/scores, Kendall tau, NDCG@k, top-k recall, update cadence, proposal rejection counts and wall/GPU accounting.
7. Label the implementation boundary on every T5 row and F2/F3 metadata: `paper_spec_reimplementation`, `author_runtime_ready=false`, `low_data_adaptation=true`, and `author_code_numerical_result=false`.
8. Do not compare the adapter's local values to author-reported FBNet/NAS-Bench numbers. Only the four methods rerun under the project protocol may be numerically ranked.

## Required code work after the active G1 source-freeze cohort finishes

1. Add a resumable `hw_pr_nas` searcher with explicit initial-design, fit, score, acquire, evaluate and update events.
2. Generalize the launcher to all four methods, counterbalance method order by seed, force budget 300 and seed set 42-51, and reject protocol/config mismatches.
3. Generalize paired inference from Aging-vs-RL to all six method pairs for exact final HV and other prespecified seed-level metrics, with Holm correction by experiment family and retained Cohen's dz/intervals.
4. Add tests for evaluator-call accounting, no leakage from unevaluated candidates, resume equivalence, deterministic acquisition, all-pairs Holm adjustment and fail-closed source/protocol mismatches.
5. Run 10-call smokes, then the 50-call x three-seed pilot. Freeze the pilot decision and only then launch 300-call x ten-seed formal runs after G3 permits claimable search.
6. Create and verify a new source snapshot before the first formal T5 run; never mix the current G1 fingerprint with the future four-method implementation fingerprint.

## Present gate

T5, F2 and F3 remain `PENDING`; G3 remains `FROZEN`. This card authorizes implementation planning only and does not authorize a formal NAS performance claim.
