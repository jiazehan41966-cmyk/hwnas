# NAS four-method protocol amendment: HW-PR evidence boundary

## Reason for amendment

The pinned HW-PR-NAS checkout at commit `296c6576fbae2b277e56c704ff3b6e648ec4c2be` is not an executable author artifact. Its README names surrogate modules that are absent, `search_algo.py` contains an undefined `valid_loss()` call, and `test.py` calls an incompatible search entrypoint. The repository therefore cannot be represented as a successfully reproduced author runtime.

The four-method comparison remains in scope, but the fourth method is frozen as `hwpr_paper_spec_local_adapter`, a B-class method migration. Every table, figure and manifest must display `NOT_AUTHOR_RUNTIME` and `paper_encoder_equivalent=false`. Author-repository numerical results are never imported into T5.

## Common formal protocol

- Methods: `random`, `rl`, `aging_evolution`, `hwpr_paper_spec_local_adapter`.
- Pilot: 50 actual evaluator calls per method for seeds 42, 43 and 44.
- Formal: 300 actual evaluator calls per method for seeds 42 through 51.
- Each evaluator call trains/evaluates one candidate under the same frozen NKSID inner split, epoch budget, data transforms, search space and analytic hardware estimator.
- Rejected proposals, duplicate proposals, surrogate-only scoring and failed evaluator calls are logged separately and do not silently consume or enlarge the successful-evaluation budget.
- Method order follows a four-method Latin-square rotation by seed so wall-clock drift does not always favor one method. GPU execution is sequential and exclusive.
- Existing search-time `f_robust` remains the fixed four-condition protocol. The later image-domain SNR experiment is an extension and does not redefine the search objective.

## Exact Pareto/HV contract

- Exact normalized minimization vector: `(1-f_clean, 1-f_robust, latency_ms/50.0)`.
- Reference point: `(1, 1, 1)`.
- LUT, DSP and BRAM are the formal hard feasibility constraints. Energy and power estimates remain diagnostic columns and are not a fourth Pareto objective or measured-power evidence.
- A candidate with normalized latency greater than one contributes no dominated volume beyond the fixed reference point; latency is not silently clipped into an improvement.
- T5/F2/F3 use the exact tested hypervolume implementation and retain every per-call anytime point.
- Secondary metrics: bidirectional Pareto coverage, unique feasible non-dominated count, feasible ratio, top-k recall, NDCG@k, GPU-hours, wall-clock, peak CUDA memory, duplicate/rejection/failure counts and each seed's final distribution.

## Local HW-PR paper-spec adapter

- Architecture representation: the declared local stage-tabular encoding; it is not the missing author feature+GCN+LSTM encoder.
- Target: exact Pareto rank computed from evaluated candidates under the common three-objective contract.
- Surrogate: three-layer MLP trained with the paper-described ListMLE Pareto-rank loss.
- Warm start: 20 random evaluated candidates.
- Update cadence: refit after every 10 additional successful evaluator calls.
- Proposal pool: 64 unique unevaluated architectures sampled from the same frozen search space at each decision.
- Selection: highest surrogate score with 0.10 predeclared random exploration probability; proposal, score, model seed and training loss are archived.
- The adapter may be compared numerically under the unified project protocol, but the supported claim is only about this local paper-spec migration, not the missing author implementation.

## Implementation gate

No `.py`, `.yaml`, `.json` or `.toml` file may be added or modified while the current pretrained G1 run depends on code state `bdf1a9aab4f50b6de0eddcf7a9493bd4e3b70ee46c596b041e02c73a6ae82471`. The adapter implementation begins only after that run is complete and independently audited, or inside a separately frozen source snapshot with a new campaign fingerprint. Until then, T5 remains unavailable and the HW-PR formal status is `PENDING_LOCAL_ADAPTER_IMPLEMENTATION`.
