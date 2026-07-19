# Hardware publication-artifact builder gap: 2026-07-17

## Current finding

The repository registers the formal titles for T6-T8 and F4/F9-F12 and now has
fail-closed HLS/route, board-latency and external-power collection contracts. It
does not yet contain a single formal builder that converts accepted hardware
evidence into all required table/figure formats. Existing Phase0 scripts and
smoke plots cannot fill this role, and manual assembly is prohibited.

This is an implementation gap, not a data result. All named artifacts remain
`PENDING`.

## Required authoritative inputs

The future builder must accept paths explicitly and recompute their SHA256 values:

1. HLS/route truth CSV accepted by `audit_hardware_collection.txt`, with at least
   100 claimable complete-network rows across five frozen architecture families;
2. grouped-CV prediction CSV containing one held-out prediction per
   `sample_id/method/target`, plus predictor model/config/code hashes and the
   frozen fold assignment;
3. the grouped proxy-analysis JSON containing MAE, RMSE, sMAPE, Spearman rho,
   Kendall tau, top-k recall, false-feasible/false-infeasible rates and confidence
   intervals for analytic/LUT, linear, gradient boosting and HARP-GNN;
4. frozen three-candidate selection manifest and the common route/HLS truth rows;
5. board-latency CSV accepted by the final board auditor, plus the separate
   full-validation prediction/accuracy audit;
6. three per-candidate power manifests, canonical base summaries and a passing
   schema-v2 UTC-aligned campaign audit;
7. source-freeze, environment/toolchain and measurement-first ledger inputs.

The grouped prediction CSV does not exist yet. Its minimum long-form schema must
be:

`campaign_run_id, sample_id, architecture_family, grouped_fold, method_id,
target_name, predicted_value, measured_value, predicted_feasible,
measured_feasible, predictor_checkpoint_sha256, predictor_config_sha256,
input_truth_sha256, project_code_commit, project_code_state_sha256,
source_freeze_manifest_sha256, claimability_status`.

Each successful truth row must appear exactly once per method/target in its held-
out family fold. Training-fold predictions cannot be mixed into formal T6/F4.

## Required output mapping

| Artifact | Required source rows | Mandatory content |
|---|---|---|
| T6 | grouped proxy metrics | target/method, n, MAE, RMSE, sMAPE, rho, tau, top-k recall, false-feasible/infeasible, CI, status |
| F4 | held-out prediction pairs | predicted-versus-measured panels, identity line, calibration/CI, family/fold markers |
| T7 | fixed candidate HLS/route/board summaries | candidate role/ID, cycles/II/resources, WNS/TNS/clock, latency p50/p95/p99, FPS, error rate, macro-F1/top1 |
| F9 | T7 resource source | LUT/DSP/BRAM/FF absolute values and utilization percentages for the three roles |
| F10 | raw accepted board rows | per-role latency distribution and ECDF with n and failed-inference accounting |
| T8 | passing power summaries | idle/active/dynamic W, total/dynamic mJ per inference, FPS/W, temperature, blocks/samples/inferences |
| F11 | raw bound meter CSVs | idle/active time-series panels with candidate/block IDs and receipt-aligned active intervals |
| F12 | joined accuracy/latency/power candidates | macro-F1, board latency and dynamic-energy Pareto plot; only three-candidate power PASS rows |

T7/T8 must not use Vivado estimated power, GPU diagnostics, author ZCU102 values or
search-time proxies in measured columns. T6 must not use operator-summed A8 rows
as complete-network truth.

## Archive outputs for every build

- T6-T8: CSV, Markdown and LaTeX generated from the same row objects.
- F4/F9-F12: 300 dpi PNG, vector PDF, `*_source.csv` and `*_meta.json`.
- Figure metadata: bilingual title, caption, supported claim, limitations,
  generator SHA, every input SHA and every output SHA.
- One build manifest: command, Python environment, source freeze, tool versions,
  input/output hashes, creation time and claimability state.

The builder must write to a temporary staging directory, validate every required
output and only then atomically promote a complete artifact set. Missing inputs,
non-passing audits, duplicate held-out predictions, insufficient rows/families,
candidate-role mismatch or power UTC failure must leave existing formal artifacts
untouched and return non-zero.

## Required implementation after active G1

1. Add a canonical `scripts/build_hardware_benchmark_artifacts.py` under a new
   source freeze; do not retain the final implementation only as `.txt`.
2. Add tests for held-out prediction uniqueness, table-format identity, synthetic
   plotting, file/meta hashes, atomic promotion and every fail-closed gate.
3. Run a synthetic-data smoke strictly below `results/.../smoke/`; mark every
   value non-scientific and visually QA all generated figures.
4. Run the real builder only after T6/T7/T8 source audits pass, then rebuild
   readiness and measurement-first ledgers.

The physical AV7K325 board and external meter are not needed to implement this
builder, but real T7/T8/F9-F12 output remains impossible until their measurements
exist.
