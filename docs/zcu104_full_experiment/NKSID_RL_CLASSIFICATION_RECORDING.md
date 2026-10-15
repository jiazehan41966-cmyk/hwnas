# NKSID RL-HW-NAS Classification Recording Protocol

Date: 2026-09-08; implementation reconciliation: 2026-09-09

Scope: NKSID image classification on ZCU104. This protocol excludes Cylinder2,
Roboflow, Figshare, detection labels, and cross-method NAS comparisons.

## Fixed Scope

- Task: NKSID 8-class sonar image classification.
- Search method: RL-HW-NAS only. Random, Aging Evolution, Proxyless, SPOS, and
  historical fold-0 candidates may appear only as archived context, not as
  current comparison systems.
- Data protocol: `nksid_outer5fold_inner_contiguous_v1`.
- Final report units: 5 outer folds x seeds `42,43,44`.
- Search and candidate selection: outer validation is never used. Use training
  split plus inner validation only.
- Hardware platform: AMD ZCU104, `xczu7ev-ffvc1156-2-e`.
- Search hardware cost source: completed ZCU104 operator cost LUT:
  `artifacts/zcu104_full_campaign_v1/operator_cost_lut/mobilenetA_full_dmdc_cfem_v2_recovered/hardware_cost_lut.formal.json`.
  Its paired coverage audit is:
  `artifacts/zcu104_full_campaign_v1/operator_cost_lut/mobilenetA_full_dmdc_cfem_v2_recovered/coverage_audit.formal.json`.
  Do not fall back to AV7K325, analytic estimates, dummy tables, or unrelated historical rows.
  This table itself contains HLS latency, routed operator resources, and Vivado
  vectorless power estimates. The estimator sums HLS latency/resources, takes
  maximum operator vectorless power as `power_w`, and sums each operator's
  latency times vectorless power as `energy_mj`. These are search proxies,
  not measured board latency, power, or energy.

## Data And Task Record

Current dataset audit snapshot:

- JSON: `results/nksid_classification_setup_20260908/nksid_protocol_audit.json`
- Markdown: `results/nksid_classification_setup_20260908/nksid_protocol_audit.md`

Fields to copy into every run summary:

| Field | Current value or source |
|---|---|
| `dataset_version` | local NKSID files bound by `train_abs.txt`, `kfold_train.txt`, `kfold_val.txt` SHA256 |
| `split_id` | `nksid_outer5fold_inner_contiguous_v1` |
| `folds` | `0,1,2,3,4` for final reporting |
| `seeds` | `42,43,44` |
| `input_size` | `1x224x224`, grayscale |
| `class_names` | `big_propeller`, `cylinder`, `fishing_net`, `floats`, `iron_pipeline`, `small_propeller`, `soft_pipeline`, `tire` |
| `sample_count` | `2617` |
| `class_distribution` | `203,288,20,951,112,94,115,834` by class index |
| `preprocessing` | frozen geometry `stretch_224`, normalization mean/std `0.5/0.5`, training augmentation profile from config |
| `claim_boundary` | image-index split only; group-safe acquisition generalization is not claimable without mission/group metadata |

## RL Search Record

Use the existing `run_search.py` path and existing RL reward settings. The
20-episode RL-only config is (the legacy filename contains an eight-stage MobileNetA space):

```powershell
python run_search.py --config configs/search/zcu104_fourstage_cfem20_v2_rl_entropy_pareto.yaml
```

The current runtime connects `ZCU104MobileNetAOperatorCostEstimator` to the
88-row table and verifies the table, coverage, and preflight binding. The old
RAW20/CFEM20 9-row adapter is retired. Successful wiring does not establish
search, full-network route, or board evidence.

A separate 200-episode configuration is
[the feasible-budget variant](../../configs/search/zcu104_mobilenetA_v2_rl_entropy_pareto_feasible_budget.yaml):
its latency limit is 120 ms and operator-aggregate BRAM budget is 700, versus
25 ms and 218 in the 20-episode configuration. These are distinct protocols,
and 700 is not a board capacity claim. See [config versions](../SEARCH_CONFIG_CANONICAL.md).

The eight-stage architecture and legal operator choices are defined once in
[the current method specification](../method_design.md). Record the resolved
encoding and its hash for every candidate. The classification target slot is
32 -> 32 at 28x28, stride 1; Skip is legal only at same-channel stride-1
placements. The 640-input detection branch has different shapes, labels,
head, and metrics and is outside this classification protocol.

Required search artifacts:

| Artifact | Required fields |
|---|---|
| `config.yaml` | frozen search space, reward config, hardware LUT paths, constraints |
| `results/dataset_summary.json` | dataset, fold, seed, input size, inner split status |
| `results/search_space_summary.json` | exact searchable space identity and candidate count |
| `results/candidates.jsonl` | candidate order, `arch_id`, `architecture_sha256`, metrics, reward extras |
| `checkpoints/search_state.json` | total evaluated, feasible, infeasible, best-so-far candidate, best reward |
| `results/search_efficiency.json` | wall time, GPU reserved hours, CUDA event seconds, candidate counts |
| `results/pareto_selection.json` | front size, objectives, directions, exact normalized HV metadata |
| `results/pareto_representatives.json` | accuracy-first, knee-point, latency-energy-first roles |
| `results/lut_stats.json` | formal LUT hit/miss and fallback checks |
| `results/rl_search_metrics.json` | RL-only main-table search efficiency summary |
| `results/rl_search_curve.csv` | evaluator-order reward and Macro-F1 convergence curve |

Before launching an RL search, run the formal LUT preflight locally:

```powershell
python scripts/audit_zcu104_mobilenetA_formal_lut_preflight.py
```

The expected output record is:

```text
artifacts/zcu104_full_campaign_v1/operator_cost_lut/mobilenetA_full_dmdc_cfem_v2_recovered/preflight.formal.json
```

This preflight confirms the operator-cost LUT integrity and coverage only. It
does not establish full-network HLS, full route, bitstream, board classification,
latency/FPS, or measured board power.

Main search metrics:

- `Search Time = search_efficiency.wall_clock_seconds`.
- `GPU-hours = search_efficiency.gpu_reserved_hours`.
- `CPU wall-time = search_efficiency.wall_clock_seconds` when CUDA is not used.
- `Evaluated Candidates = summary.total_evaluated`.
- `Valid Candidates = summary.feasible`.
- `Search-admissible Rate = feasible / total_evaluated`; use `NOT_APPLICABLE` when total is zero. This is estimator/config admissibility, not full-network route success.
- `Best-so-far Macro-F1` is the cumulative maximum of `candidate.metrics.macro_f1`
  over evaluation order.
- `Best-so-far Reward` is the cumulative maximum of `extra.reward` or
  `extra.reward_with_bonus`, reported with the chosen reward definition.

Search efficiency metrics:

- `Sample Efficiency`: candidates needed to first reach 95% of the final
  best feasible Macro-F1.
- `Convergence Speed`: the same candidate index divided by total evaluated
  candidates; optionally also report wall-clock time to that point.
- `Reward Curve`: evaluation index vs `extra.reward` and best-so-far reward.
- `Macro-F1 Curve`: evaluation index vs `candidate.metrics.macro_f1` and
  best-so-far Macro-F1.
- `Valid Candidates / GPU-hour = feasible / gpu_reserved_hours`; if CUDA is
  not used, mark this `NOT_APPLICABLE_CPU_RUN`.

These efficiency metrics describe the RL process itself. They do not support a
claim that RL is better than Random, Aging, Proxyless, or SPOS.

After a search finishes, generate the summary with:

```powershell
python scripts/summarize_rl_search_metrics.py --run-dir <search_run_dir>
```

## Pareto Front Quality

Primary Pareto objectives for this run:

```text
maximize macro_f1
minimize latency_ms
minimize energy_mj
minimize dsp
minimize bram
minimize lut
```

Required records:

- Pareto front candidate list.
- Non-dominated solution count.
- Exact normalized Hypervolume with reference point `(1,1,1,1,1,1)`.
- Normalization bounds: Macro-F1 in `[0,1]`; hardware objectives use the frozen
  ZCU104 constraints from the search config.
- Diversity / Spread is optional and must stay out of the main table until a
  formula is frozen.

## Candidate Freeze And Retraining

Freeze final candidates only from the RL Pareto front after the search artifacts
above pass. Recommended roles:

- `accuracy_first`
- `knee_point`
- `latency_energy_first`

Each frozen candidate must bind:

- `arch_id`
- `architecture_sha256`
- source `pareto_selection.json` SHA256
- search config SHA256
- selection role and rule

Final classification uses `run_eval_protocol.py`, not the search proxy metric.
Record:

- Macro-F1
- Top-1 Accuracy
- Tail-class F1 for `fishing_net` and `small_propeller`
- Per-class F1
- Confusion Matrix

Macro-Precision and Macro-Recall are appendix or error-analysis fields, not
main-table fields.

## Robustness

Run robustness only on the classification test/outer-validation samples already
bound by the protocol. Required fields:

- Clean Macro-F1
- Noisy Macro-F1
- Delta Macro-F1
- noise type
- noise strength
- corruption seed and artifact SHA256

`Noise Robustness Score` is not used. PSNR and SSIM are auxiliary fields only
for synthetic noise experiments with a clean reference image.

## Complexity And INT8

Required records:

- Params
- MACs
- FP32 Macro-F1 / Top-1
- INT8 Macro-F1 / Top-1
- Delta Macro-F1
- Delta Top-1
- PTQ or QAT type
- INT8 parity status

Candidates that fail the frozen INT8 accuracy/parity gate do not enter formal
board power comparison.

## ZCU104 Deployment

A deployment row is complete only after HLS, Full Route, bitstream generation,
runtime binding, and board verification share the same identity chain.

Required records:

- LUT, FF, DSP, BRAM and utilization
- Actual Clock
- WNS, TNS, Timing Met
- bitstream SHA256
- System E2E Latency
- Kernel Latency
- FPS / Throughput
- `N_completed`
- `T_active`
- p50 / p95 latency
- Board Macro-F1
- Board Top-1
- SW-FPGA Argmax Agreement
- Mismatch Count
- CRC Error Count
- Runtime Error Count

Formula:

```text
FPS = N_completed / T_active
```

Formal FPS comes from a continuous inference window. `1000 / latency_ms` is only
a cross-check.

## Power And Energy

Formal source: ZCU104 onboard INA226/hwmon VCC12 board-input power capture.

Required records:

- Idle Power
- Active Power
- Dynamic Power
- Energy/Image
- FPS/W
- sampling protocol
- sample duration
- raw CSV path and SHA256

Formulas:

```text
Dynamic Power = Active Power - Idle Power
Energy/Image = 1000 * integral(P_active(t) - P_idle) dt / N_completed
FPS/W = FPS / Active Power
```

Do not write search cost, Vivado `report_power`, or `power_w` estimates as
measured board power.

## Full-Network Cost-Model Calibration

After complete-network ZCU104 measurements exist, calibrate the search-stage
hardware estimate using one relative-error definition plus rank/error checks:

- MAE for latency, LUT, DSP, BRAM
- relative absolute error
- Spearman rho
- False-feasible Rate

Keep NMAE, sMAPE, and MAPE from being reported side by side unless a single
primary relative-error definition is frozen first.

## Main Table Fields

Main table:

`Macro-F1`, `Top-1`, `Tail-class F1`, `Params`, `MACs`, `Latency`, `FPS`,
`LUT`, `DSP`, `BRAM`, `Power`, `Energy/Image`, `Search Time`, `GPU-hours`,
`Evaluated Candidates`, `Search-admissible Rate`, `Hypervolume`.

Appendix or optional:

`Precision`, `Recall`, `Diversity / Spread`, `PSNR`, `SSIM`, `CPU-hours`.
