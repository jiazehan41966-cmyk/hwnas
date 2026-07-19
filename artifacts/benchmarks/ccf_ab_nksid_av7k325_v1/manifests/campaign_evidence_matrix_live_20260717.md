# Live campaign evidence matrix: 2026-07-17

## Authority and snapshot warning

This matrix reconciles the current filesystem after the formal-readiness snapshot
dated 2026-07-16 07:27. It is a progress ledger, not a replacement for the
canonical machine-readable readiness audit. The canonical JSON must not be
regenerated while the active G1 source-frozen process is running because doing so
would alter the frozen code-state cohort.

The older `formal_readiness.json` is therefore intentionally stale. In particular,
it does not contain the accepted pretrained 15-unit run, treats the original
scratch directory more strongly than the current independent-audit policy allows,
and counts historical HLS/route rows that do not meet the new full-network T6
admission contract.

## Requirement-by-requirement status

| Requirement | Current verified evidence | Status | Remaining proof |
|---|---|---|---|
| R1 paper/source audit | Five main papers pinned; commits/remotes match; enriched T1 v2 joins source/smoke/environment/local status and independently passes 6x26 cross-format audit | SOURCE AUDIT PASS / CANONICAL INTEGRATION PENDING | After G1 migrate v2 join into the tested canonical builder, promote `t1_v2.*` to `t1.*`, rebuild readiness, and register HW-PR-NAS/HARP/ESDA gitlinks |
| R2 dedicated environments | Six paper-specific environments and lock files; all probes PASS | PASS | Re-probe only if an environment or adapter changes |
| R3 closed set | ImageNet-pretrained MobileNetV2 independently accepted 15/15; NAS champion independently accepted 14/15; all 14 paired NAS macro-F1 values are below pretrained with interim mean difference -0.2396; execution was stopped at the atomic boundary before fold4/seed44 | PAUSED PENDING USER DECISION; 29/60 units accepted | User decides whether to finish the final NAS unit, whether to continue SURE/scratch-v2, and whether any candidate/protocol change is authorized; no automatic downstream launch |
| R4 open/long-tail | The 2,617-sample, 15-unit 5-known/3-unknown input manifest independently passes all hashes, memberships and no-unknown-training-leakage checks; the upgraded result auditor passes a bound one-unit non-scientific fixture and rejects an unbound CE+MSP smoke | INPUT/AUDIT CONTRACT FROZEN / RESULTS PENDING | Add the frozen manifest binding to the canonical entrypoint under a new source freeze, then run CE+MSP, DMCL and PLUD each 5 folds x 3 seeds plus independent open-set result audit |
| R5 formal NAS comparison | Exact-HV/statistical contracts tested; four-method runner still incomplete | PENDING / G3 FROZEN | Random, RL, Aging and labelled HW-PR paper-spec adapter; 300 evaluator calls x 10 paired seeds |
| R6 HLS/route proxy | Toolchain and fail-closed collection/prediction contracts pass synthetic tests; a 100-unique-candidate structural DOE and one deterministic pilot per family are independently frozen; planning-only execution reproduces incomplete component mapping for all five pilots and rejects reference-bitstream reuse; historical A8 rows remain rejected | INPUT PASS / GENERATOR GAP CONFIRMED / TRUTH PENDING; 0 real rows | After the software source-freeze boundary, implement source-linked candidate-specific HLS mapping and semantic equivalence for the five frozen pilots; if all five pass, queue the remaining 95, retain every HLS/route failure, then produce 5,200 real held-out predictions |
| R7 board and power | JTAG target and COM5 observed; 3x1000 board-latency and three-candidate UTC-aligned power contracts pass synthetic tests; no fixed-candidate real campaign or external power trace | PENDING / NOT_MEASURED | Three fixed route-feasible candidates, real board latency, and external-instrument idle/active blocks |
| R8 measurement-first gates | G0 PASS; G1/G2/G4 PENDING; G3 FROZEN; G5 PAUSED; power NOT_MEASURED | PENDING | Close each gate with its own evidence layer; no proxy-to-board promotion |
| R9 archival bundle | T1 and F1 files exist; T2-T9 and F2-F12 absent by design | 2/21 file bundles available | Generate every remaining artifact from audited formal sources and run rebuild verification |

## Closed-set live unit accounting

| Method | Required units | Independently accepted now | Boundary |
|---|---:|---:|---|
| ImageNet-pretrained MobileNetV2 | 15 | 15 | method-level summary allowed; no cross-method claim |
| Frozen NAS champion | 15 | 7 | folds 0 and 1 complete; fold 2 seed 42 accepted; atomic-unit evidence only |
| Scratch MobileNetV2 v2 | 15 | 0 | original scientific run is not substituted for the required clean re-audit cohort |
| Same-backbone SURE | 15 | 0 | source/environment smoke is not a formal NKSID result |
| **Total** | **60** | **22** | T2/F6 remain unavailable until 60/60 and paired audit |

## Archive integrity checks and repairs after G1

1. UTF-8 decoding and visual inspection confirm that the Chinese titles in
   `artifact_status.json`, F1 metadata, F1 source CSV and the rendered F1 image are
   correct. Windows PowerShell 5.1 must read these files with explicit
   `-Encoding UTF8`; its ANSI default can create display-only mojibake and must not
   be treated as file corruption.
2. T1 omits source-smoke and dedicated-environment evidence and can display
   `formal_eligible=False` with an empty blocker. The repair contract is frozen in
   `t1_reproducibility_coverage_gap_20260717.md`.
3. `.gitmodules` declares HW-PR-NAS, HARP and ESDA, but their top-level gitlinks
   are not registered yet; the exact live state and safe post-G1 repair are
   recorded in `r1_live_repository_audit_20260717.md`.
4. The canonical readiness snapshot must be rebuilt after the software chain,
   not edited by hand, so its hashes bind the final accepted runs.
5. Smoke figures under `results/.../smoke/` remain non-scientific and must never be
   copied into the formal `figures/` directory.
6. T6 eligibility starts at zero under the complete-network admission contract;
   repeated encodings or component-summed operator HLS rows cannot inflate the
   sample count.

## Next automatic boundaries

1. Continue the current NAS champion process without interruption and audit each
   atomic unit after write completion.
2. After NAS 15/15, require the full independent audit before the wrapper starts
   scratch-v2.
3. After all four closed-set methods pass, build T2/F6, then execute the frozen
   corruption protocol for T4/T9/F7/F8.
4. Only after the software chain and new source freeze may the T1/readiness
   generator defects be repaired and all affected artifacts regenerated.

No board operation is required for the current R3 execution. Physical AV7K325 and
the external power instrument enter only after the three final route-feasible
candidates are frozen for R7.
