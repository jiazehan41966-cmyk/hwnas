# Open-set sample-binding result auditor patch: 2026-07-17

## Closed evidence gap

The previous result auditor validated the 5-known/3-unknown class split, output counts, prediction/checkpoint hashes and metrics, but it did not prove that each prediction row belonged to the predeclared sample-level manifest. A formally complete 15-unit run could therefore have passed without binding the newly frozen 2,617-sample evidence object.

## Added fail-closed checks

The staged runtime auditor now requires `immutable_config.open_set_sample_manifest` to bind:

- the sample manifest path and SHA256;
- the independent input-audit path and SHA256;
- `FROZEN_INPUT_NOT_RESULT` identity, 2,617 samples and all 15 fold-seed units;
- the exact class-level protocol SHA and source-freeze manifest SHA;
- a `PASS` input audit with 2,617 rehashed samples, 15 accepted units, no errors and zero result increment.

For each completed result unit it independently verifies:

- prediction `sample_id` order equals the manifest's `outer_val_indices` order;
- each persisted target equals the frozen sample target;
- class holdout protocol and base split SHA equal the frozen unit;
- run record and provenance contain the same complete binding;
- every prediction row contains the exact sample-manifest SHA.

## Contract evidence

- Auditor: `runtime/audit_open_run.txt`, SHA256 `a179bb8880ddf82ed44a8b777ce5b5dece6bb957c51b5a3014bf9675ddf2f773`.
- Test generator: `runtime/test_open_run_sample_manifest_contract.txt`, SHA256 `2fdcef445fae1d98c4776e4b034d2fe06e2a11df792f6b65648acb777fe7e68e`.
- Bound one-unit fixture result: SHA256 `319ff249354dd8001834eb19227d0a211057ea70648a5db99c11e2b4feb3bbde`, status `PASS`.
- Bound fixture audit: SHA256 `a12d4c87b9e16b459227d469423bb206ab758012be39fe1ad532a8f3dbbcdc89`, one accepted unit, zero errors.
- Unbound CE+MSP rejection: SHA256 `19a418a97df52197600170f4cd7b7761f2379fda7e92596efabcbaa0d5fb7f46`, expected exit code 2 and `FAIL`.

All fixtures are explicitly `NON_SCIENTIFIC_SMOKE_ONLY_FORMAL_RESULT_INCREMENT_0`. R4, T3, F5 and all method rankings remain unavailable.

## Remaining canonical integration

After the active G1 source-frozen chain reaches a safe boundary, `run_eval_protocol.py` must add an explicit sample-manifest input, validate it before training, include the complete binding in `immutable_config`, records, checkpoint metadata and prediction rows, then create a new source freeze. Until that happens, no CE+MSP, DMCL or PLUD run can pass the upgraded auditor.
