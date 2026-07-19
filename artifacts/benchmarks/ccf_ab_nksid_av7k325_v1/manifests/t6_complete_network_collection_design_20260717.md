# T6 complete-network HLS/route collection design: 2026-07-17

## Objective

Collect at least 100 unique, semantic-safe complete-network HLS/route rows for grouped five-fold proxy evaluation. The execution target is 125 prespecified candidates (25 per macro family) so that synthesis/route failures remain visible without making the successful-row threshold depend on selective replacement.

This is a collection design, not collected evidence. Formal T6/F4 remain `PENDING` until the on-disk auditor recomputes all hashes and accepts at least 100 claimable complete rows.

## Why historical rows are insufficient

- The 29 board-harness candidate files contain only 25 unique architecture IDs, nine unique detailed encodings and four top-level operator sequences.
- The historical RL200 evaluator log contains 200 rows but only four unique network encodings. Repeated training/evaluation of the same encoding cannot be counted as distinct HLS networks.
- The corrected A8 cohort has eight unique routed networks but uses component-summed operator HLS estimates and therefore contributes zero complete-network HLS rows.
- `denoise` and `edge` remain outside the semantic-safe T6 pool until PyTorch-to-fixed-point/HLS numeric parity and matching weight-export evidence exist.

## Frozen candidate domain

- Base family profile: the seven-stage `mobile_anchor` space used by the matched RL/Aging configurations.
- Allowed operators for this T6 collection: `mbconv` and legal identity `skip` only.
- Kernel sizes: 3 and 5 for `mbconv`.
- Expansion ratios: 1 and 2 for `mbconv`.
- Width choices: the profile's 0.75x and 1.0x per-stage channel choices.
- Depth choices: the profile's per-stage shallow/deep options.
- Skip is legal only when stride is 1 and input/output channels match; the project search-space validator must accept every frozen candidate before HLS generation.
- Target: `xc7k325t-ffg900-2`, Vitis HLS/Vivado 2023.2, one frozen clock policy and one full-network generator version.

## Five macro families

Family labels are fixed from width and depth profiles before resource or performance results are known. Kernel, expansion and legal skip choices vary within each family to provide micro-architecture diversity.

| Family ID | Width profile | Optional-stage depth profile | Target rows |
|---|---|---|---:|
| `ma_wlow_dshallow` | 0.75x at all seven stages | minimum legal depth at every stage | 25 |
| `ma_whigh_dshallow` | 1.0x at all seven stages | minimum legal depth at every stage | 25 |
| `ma_walternating_dmixed` | alternating 0.75x/1.0x by stage | alternating shallow/deep where depth is variable | 25 |
| `ma_wlow_ddeep` | 0.75x at all seven stages | maximum legal depth at every variable stage | 25 |
| `ma_whigh_ddeep` | 1.0x at all seven stages | maximum legal depth at every variable stage | 25 |

The high/deep family is intentionally retained even if it has more infeasible or route-failed candidates; removing it after observing failures would bias false-feasible and false-infeasible estimates.

## Deterministic within-family sampling

1. Enumerate or rejection-sample legal unique encodings using a frozen seed per family.
2. Deduplicate by canonical architecture SHA256, not by `arch_id`.
3. Balance kernel 3/5, expansion 1/2 and legal skip-count bins as far as the macro profile permits.
4. Select 25 candidates per family without using accuracy, proxy resource, HLS or route outcomes.
5. Freeze the ordered manifest and its SHA256 before the first synthesis. Replacement is allowed only for a pre-HLS semantic/schema failure and must retain an explicit rejected row and reason.

## Staged execution

- Stage A: one candidate per family. Require source generation, semantic equivalence, complete-network csynth and route provenance to pass for all five before scaling.
- Stage B: five candidates per family (25 targets). This is descriptive only if fewer than 30 complete claimable rows exist.
- Stage C: extend to 15 candidates per family (75 targets). Results are exploratory and cannot support formal grouped inference.
- Stage D: extend to 25 candidates per family (125 targets). Run the auditor; formal grouped five-fold analysis starts only if at least 100 rows are complete and claimable.

No stage silently drops failed HLS or route attempts. Each failure retains its architecture/source/config hashes, elapsed time, last valid report and normalized failure category.

## One-row evidence contract

Each row must contain the fields frozen by the hardware auditor, including paper/method IDs, candidate and candidate-pool hashes, explicit `network_scope=COMPLETE_NETWORK`, the full-network HLS top-function name and source hash, a bound semantic-equivalence report, source-freeze manifest SHA, project commit/code-state SHA, command/config hash, csynth-report hash, HLS cycles/II/LUT/DSP/BRAM/FF, route-report hash, WNS/TNS/achieved clock/resources/status, failure stage, tool versions, elapsed time and `claimability_status`.

Stage-aware report rules are fixed. A csynth PASS row must bind its report and all HLS metrics; a route PASS row must additionally bind its route report and route metrics. A legitimate csynth failure may leave csynth and route reports/metrics empty, while a legitimate route failure must retain the successful csynth report and HLS metrics but may leave the unavailable route report/metrics empty. Every failed attempt must provide `failure_stage` and `failure_category`. Valid failed rows remain in the all-target failure denominator but do not create schema errors or enter the complete regression-target count.

Component-level HLS aggregates may be retained as separate predictor features, but the measured target fields come only from the candidate-specific full-network top and route.

## Grouped analysis

- Split: grouped five-fold by the five fixed macro family IDs; no random row split.
- Methods: current analytic/LUT estimator, linear regression, gradient boosting and HARP-GNN.
- Metrics: MAE, RMSE, sMAPE, Spearman rho, Kendall tau, top-k recall, false-feasible rate, false-infeasible rate and calibration curves.
- HARP input must be generated from the candidate-specific HLS C/C++ through the supported LLVM/program-graph route; NAS architecture JSON is not a valid HARP graph input.
- Report both all-target failure rates and successful-measurement regression metrics. A method is not rewarded by excluding difficult failed candidates.

The grouped evidence is a long prediction table with exactly one row for every admitted complete-network sample × method × target combination. The 13 fixed targets are HLS cycles/II/LUT/DSP/BRAM/FF plus route WNS/TNS/achieved clock/LUT/DSP/BRAM/FF. At 100 complete rows this requires exactly 5,200 held-out prediction rows and yields 52 method-target metric rows. Each prediction binds its predictor artifact and config, fold manifest, feasibility config, truth CSV, source freeze and project code state.

The fold manifest must make each architecture family the held-out test family exactly once; its train sample list is the exact complement and no sample may occur in two test folds. Formal feasibility is recomputed from held-out predictions and truth using the frozen AV7K325 policy: route LUT≤203800, FF≤407600, BRAM≤445, DSP≤700, WNS≥0 and achieved clock≥200 MHz. The board's physical DSP capacity of 840 is retained separately and must not replace the stricter deployment gate. False-feasible is FP divided by measured-infeasible count; false-infeasible is FN divided by measured-feasible count. Top-k recall uses the prespecified top 10% with metric direction declared per target.

## Execution boundary

Do not launch this collection while the current G1 source-freeze cohort is running. First implement and smoke the complete-network HLS generator, freeze the five-family candidate manifest and create a new source snapshot. The AV7K325 board is not required for HLS/route collection; it becomes mandatory for the later three-candidate COM5 latency and power phases.

The grouped prediction contract has now passed a synthetic-only test with 100 truth rows, five families, 5,200 prediction rows and 52 metric rows. Duplicate, missing, measured-value mismatch and feasibility-label mismatch cases are rejected. This test increments the real T6 denominator by zero; no actual complete-network project prediction exists yet.
