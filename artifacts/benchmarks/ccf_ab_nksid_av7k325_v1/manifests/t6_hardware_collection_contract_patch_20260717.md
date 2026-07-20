# T6 hardware-collection contract repair: 2026-07-17

## Problem found

The pre-collection schema could not mechanically prove that a row represented a
complete-network HLS top, and semantic-equivalence status was not bound to a
report hash. It also required csynth and route reports for every row even when a
stage legitimately failed before those reports existed. Because grouped readiness
required zero audit errors, one honest failed attempt could prevent formal
readiness forever, contradicting the requirement to retain failures for failure-
rate and false-feasibility analysis.

## Repair applied

The CSV and auditor now require or track:

- `paper_id` and `method_id`;
- candidate-pool manifest SHA;
- explicit `network_scope` and full-network HLS top-function name;
- bound semantic-equivalence report path/SHA;
- source-freeze manifest SHA, project Git commit and project code-state SHA;
- existing candidate/source/toolchain/config/command/checkpoint bindings;
- stage-aware csynth/route reports, metrics and failure categories.

Only rows whose `network_scope` is `COMPLETE_NETWORK` or `FULL_NETWORK` can enter
`complete_hls_route_rows`. Operator-only rows remain valid inventory records but
cannot increase the T6 denominator.

Stage-aware behavior is now:

- csynth PASS requires a bound csynth report and complete HLS metrics;
- route PASS additionally requires a bound route report and route metrics;
- csynth failure may leave csynth and route reports/metrics empty;
- route failure retains the successful csynth report/HLS metrics but may leave
  route report/metrics empty;
- every failure requires `failure_stage` and `failure_category`;
- a route PASS with non-passing csynth is rejected as inconsistent.

## Contract tests

The updated `.txt` auditor compiles and its CLI help loads successfully.

An isolated temporary-directory test exercised five cases:

| Case | Expected | Observed |
|---|---|---|
| complete-network csynth+route PASS | one complete row, no errors | PASS |
| operator-only csynth+route PASS | zero complete rows, no schema error | PASS |
| legitimate csynth failure | one failure, no missing-report error | PASS |
| legitimate route failure after csynth PASS | one failure, no missing-route-report error | PASS |
| failure without stage/category | rejected | PASS |

A second threshold test generated 100 complete rows equally spanning five
architecture families plus one legitimate csynth failure. The independent audit
returned:

`total_rows=101, semantic_safe_rows=101, full_network_rows=101,
claimable_rows=101, complete_hls_route_rows=100,
architecture_family_count=5, failure_rows=1,
evidence_level=FORMAL_COUNT_REACHED, grouped_5fold_ready=true, errors=[]`.

Auditing the header-only template returns zero schema errors but
`overall_ready=false`, which is the required fail-closed state before collection.

After the same shared auditor was extended with the T7 board-latency contract, the
HLS threshold regression was rerun against the final combined file. It again
returned 101 total rows, 100 complete rows, one retained failure, five families,
`grouped_5fold_ready=true` and no errors.

## Frozen artifact hashes

| Artifact | SHA256 |
|---|---|
| `runtime/audit_hardware_collection.txt` | `1a48570a4d2dd016e5e7cb14ae77019bb6e398793cbabbacc198eb522137ccf4` |
| `runtime/hls_route_sample_template.csv` | `51eee0037a8178adc79685a101515ddf8f8d97a6265523d49992d5cf0ef92a18` |
| `runtime/hardware_collection_runbook_20260716.md` | `3c72d57aad423f1e96c8e01290cb2b82057c48aca0505169f5910c43559ba45b` |
| `manifests/t6_complete_network_collection_design_20260717.md` | `cd76632dcbb0151f513dd5bb3d171b4a111fb379c6ce406cfbb647f1db5c32a2` |

## Claim and execution boundary

This repairs the future evidence contract; it does not create any measured HLS or
route row. T6 and F4 remain `PENDING`, and the formal sample increment remains
zero. The complete-network generator and 125-candidate manifest still require a
new source freeze after the active G1 chain. No AV7K325 board is needed for that
HLS/route collection stage.
