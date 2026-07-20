# HLS / AV7K325 / power execution gap card

## Current observed state (read-only, refreshed 2026-07-17)

- The corrected A8 isolated ledger contains 8 unique route/COM5 evidence rows: 4 fit rows plus 4 independent probes, all in the single `mainline_mbconv_skip` family. Its HLS fields are component-summed operator estimates rather than a bound candidate-specific complete-network HLS top, so its formal T6 increment is exactly zero. The T6 threshold remains at least 100 claimable complete-network HLS/route rows across at least 5 architecture families.
- G2 hardware measurement is `PENDING`; G3 search remains `FROZEN`; G4 INT8/board is `PENDING`; external power is `NOT_MEASURED`.
- The pinned HARP checkout and dedicated CUDA environment pass the LLVM program-graph contract smoke. This proves adapter/environment shape only, not prediction accuracy on project HLS/route samples.
- In `results/experiment_cycle_20260712_v2/calibration_v2_a8_corrected`, the corrected evidence audit passes 8/8 identity/provenance completeness, and all four frozen independent probes pass source-frozen route plus COM5 stability 5/5. The external-probe-count gate is therefore closed in that isolated campaign. Its `g2_pass` remains false because the independent interval-screening quality gates are not all PASS. The canonical measurement-first ledger is intentionally not rewritten while the active G1 run depends on its frozen code state.
- The corrected independent-probe diagnostics explain the remaining G2 failure: DSP passes interval screening (`P90 APE=12.46%`, Spearman rho=1.0), but LUT (`P90 APE=44.30%`), BRAM (`85.33%`) and latency (`55.98%`, rho=0.4) fail the frozen quality limits. Only DSP may hard-screen; LUT, BRAM and latency must pass through to HLS/route measurement. No scalar or interval estimate for those three may be relabelled as reliable measured hardware performance.
- `vivado` and `vitis_hls` are not on the current PATH, but the read-only discovery pass found a matched 2023.2 installation: `F:\vivado\Vivado\2023.2\bin\vivado.bat` reports `vivado v2023.2 (64-bit)`, and `F:\vivado\Vitis_HLS\2023.2\bin\vitis_hls.bat` reports `Vitis HLS v2023.2 (64-bit)`. The preflight now resolves these versioned fallbacks without changing the system PATH.
- Windows reports `Silicon Labs CP210x USB to UART Bridge (COM5)` as present and healthy. COM5 proves only the serial bridge path; it does not prove bitstream validity, board inference, HLS/route closure or external power measurement.
- Windows also reports a healthy FTDI `USB Serial Converter` (`VID_0403/PID_6014`). A read-only Vivado 2023.2 `hw_server` enumeration on 2026-07-17 resolved this device as target `localhost:3121/xilinx_tcf/Digilent/210512180081` and opened one hardware device, `xc7k325t_0`. This proves cable/target visibility only; no bitstream was programmed and no inference was executed.
- A Vivado 2023.2 batch query resolves both `xc7k325t-ffg900-2` and `xc7k325t-ffg676-2`. The formal AV7K325 board target remains `xc7k325t-ffg900-2`; `ffg676-2` remains limited to historical/operator LUT sampling and cannot support full-board closure.
- A Vitis HLS 2023.2 trivial-kernel csynth on `xc7k325t-ffg900-2` passes with II=1 and produces a bound csynth report. A separate Vivado 2023.2 trivial-RTL synth/place/route pass obtains both Synthesis and Implementation licenses, routes 36/36 routable nets with zero route errors, and reports WNS=4.091 ns and TNS=0.000 ns. Nine input/script/report/checkpoint SHA256 bindings pass.
- Both designs are explicitly `SMOKE_ONLY_NOT_FOR_SCIENTIFIC_CLAIMS`: they are not complete networks, do not increment the formal HLS/route sample count, do not populate T6/T7, and do not prove bitstream, COM5 inference, latency or power.

A structural T6 design-of-experiments pool is now frozen separately: 100 unique complete-network encodings, five architecture families with 20 candidates each, and five fixed leave-one-family-out folds with 80 training and 20 held-out candidates. This closes candidate selection only. No selected candidate has complete-network semantic-equivalence, HLS or route truth yet, so formal T6 evidence remains `0/100`.

## Collection gate and sample levels

- Fewer than 30 semantic-safe complete-network samples: descriptive evidence only.
- 30-99 samples: exploratory proxy analysis only.
- At least 100 samples: grouped 5-fold comparison may be produced, grouped by architecture family and only after synthesis/route provenance passes.
- Every row must bind paper/method, candidate and candidate-pool SHA, explicit complete-network scope/top function, generated HLS source, a semantic-equivalence report, source freeze/project commit/code state, tool version, target part/clock, command/config, stage reports and failure category.
- HLS cycles, II, LUT, DSP, BRAM and FF are separate targets from route WNS/TNS, achieved clock and route status. Failed synthesis and failed route remain rows; they must not be silently discarded.
- Stage-aware auditing requires reports and metrics only for stages that passed. Legitimate csynth/route failures remain in the all-target failure denominator without fabricating unavailable downstream reports.

## Frozen proxy comparison

- Methods: current analytic/LUT estimator, linear regression, gradient boosting and HARP-GNN.
- Metrics: MAE, RMSE, sMAPE, Spearman rho, Kendall tau, top-k recall, false-feasible rate, false-infeasible rate and prediction-versus-measurement calibration.
- Cross-validation: grouped 5-fold by architecture family. No random row split is allowed.
- T6/F4 remain unavailable until the sample-level gate is met. HARP paper numbers from a different dataset/toolchain never populate project T6.

## Three deployment candidates

Candidate roles are selected only after the formal NAS/retrain evidence exists:

1. `accuracy_first`: highest macro-F1 among route-feasible candidates.
2. `knee_point`: knee of the normalized accuracy/latency/resource frontier.
3. `resource_min`: lowest LUT candidate whose macro-F1 is within one percentage point of the best route-feasible candidate.

The same AV7K325 toolchain, clock policy, bitstream flow, COM5 harness and external instrument protocol must be used for all three. ZCU102/ESDA numbers are C-class workflow references and are never ranked against AV7K325 values.

## Board and power acceptance

- Board latency: bind exactly one distinct candidate to each frozen role, retain at least 1,000 inference timestamps per role over the same sample/target map, and report p50/p95/p99, FPS and error rate after route-clean bitstream verification.
- Power: one external instrument and one bound wiring/range/sample-rate protocol for every candidate; at least three idle blocks and three active blocks, with at least 1,000 inferences per active block. Active CSV UTC epoch timestamps must align to the RUN_REPEAT receipt interval.
- Raw meter time series are mandatory. Primary metric is dynamic energy in mJ/inference; idle/active/dynamic W, FPS/W and board temperature are secondary.
- Search-time power/energy estimates remain diagnostic and can never be relabelled as measured energy.
- F9-F12 and T7/T8 remain unavailable until their raw source data and instrument/tool manifests exist.

## Next external-state requirement

The Vivado/Vitis HLS executable-discovery, target-part database, HLS/route license-and-report path, and JTAG cable/target-enumeration blockers are closed. Meaningful formal hardware execution still requires semantic-safe complete-network candidates, a frozen candidate/bitstream, and a connected external power instrument with a verified acquisition command. Until those checks pass, software preparation may continue, but no hardware/power Gate is promoted.

## Prepared execution controls

- `hardware_preflight_20260716.ps1` performs a read-only four-part preflight for Vivado, HLS, COM/UART and the external-instrument acquisition executable. It can auto-discover versioned Xilinx/AMD installations and optionally record version lines with `-ProbeVersions`. The refreshed checks are true for Vivado, HLS and COM5; strict readiness still exits 2 because no external-instrument acquisition command is registered.
- `query_av7k325_parts_20260717.txt` is the non-synthesis Vivado batch query for the formal and historical parts. `query_av7k325_hw_target_20260717.txt` opens and closes the hardware target without programming it; the associated `hw_server` is launched with an idle timeout.
- `results/benchmarks/ccf_ab_nksid_av7k325_v1/smoke/hardware_toolchain_20260717/smoke_manifest.json` binds the HLS and route smoke inputs, scripts and reports. Its formal sample-count increment is zero by construction.
- The canonical external-power calculator retains its focused tests. The supplemental schema-v2 `.txt` auditor additionally binds three exact candidate roles, protocol/calibration files, all raw CSV/receipt hashes, candidate/route/bitstream/payload/parity provenance and active UTC alignment. Its synthetic three-candidate pass path yields the expected 120 mJ/inference, while time misalignment, duplicate roles and instrument mismatch are rejected. This v2 logic must move into tested canonical source under a new freeze before final T8 acceptance. No real meter manifest exists, so power remains `NOT_MEASURED`.
- `hardware_collection_runbook_20260716.md` freezes the execution order, candidate roles, archive roots and acceptance sequence.
- HLS/route, board-latency, power-time-series, shared power-protocol and per-candidate power-manifest templates are available beside this card.
- `audit_hardware_collection.txt` distinguishes descriptive (<30), exploratory (30–99) and formal-count (≥100) complete-network evidence, requires at least five architecture families, verifies all stage-appropriate file bindings and retains legitimate failed attempts. Its final combined regression passes 100 complete rows plus one failure across five families. The same auditor's board contract passes three roles x 1,000 paired rows and rejects candidate drift, label drift and insufficient rows. These synthetic tests prove contracts only and contribute zero measured samples.
- `hls_proxy_prediction_template.csv`, the grouped fold manifest and AV7K325 feasibility templates now close the prediction-side schema gap. `audit_hls_proxy_predictions.txt` requires exactly one held-out prediction per complete sample × four methods × 13 targets, re-verifies all bound files and recomputes regression, ranking and feasibility errors. Its synthetic contract passes 100 truth rows/5,200 predictions/52 method-target metric rows and rejects duplicate, missing, measured-value mismatch and feasibility-label mismatch cases. The test is explicitly synthetic and contributes zero formal T6 rows.
- No formal builder for T6-T8 or F4/F9-F12 exists yet. The raw input schemas and fail-closed audits are prepared, but publication artifacts must later be generated by a single hash-bound builder under a new source freeze; manual table/figure assembly is prohibited.

## Structural DOE freeze

`../manifests/t6_structural_candidate_pool_v1.json.txt` is the frozen pre-truth input for hardware collection. Its legacy analytic-feasibility field is used only to balance sampling strata and must not be reported as HLS or route feasibility. Each row remains `COMPLETE_NETWORK_PENDING_EXPORT` until source-linked export and semantic equivalence pass; only independently audited complete-network HLS/route rows may increase the T6 count.

The immediate five-family export subset is frozen in `../manifests/t6_five_family_pilot_manifest_v1.json.txt` and independently passes its input audit. It binds one unique source candidate per family and explicitly records that the physical board is not required for export, semantic equivalence, csynth or route. This input freeze contributes zero real T6 rows; the board becomes required only for bitstream/COM5 dynamic validation after route-feasible candidates exist.

The existing isolated full-network planner was run in planning-only mode on all five pilots. Every run returns `not_generated_mapping_incomplete`, contains at least six missing component rows, rejects reuse of the fixed arch-84 bitstream, and has no candidate-HLS mapping. The hash-bound audit is `../manifests/t6_five_family_mapping_gap_audit_v1.json.txt`, status `PASS_EXPECTED_GAP_CONFIRMED`. Therefore the next implementation task is not board operation: it is source-linked candidate-specific HLS mapping plus semantic equivalence under a new source freeze. The current stitched RTL planner is also not a complete-network HLS C/C++ generator suitable by itself as HARP input.
