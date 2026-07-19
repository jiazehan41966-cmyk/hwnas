# Open-set sample freeze patch: 2026-07-17

## Purpose

Freeze the exact NKSID samples and memberships used by the predeclared 5-known/3-unknown experiment before CE+MSP, DMCL or PLUD formal training begins. This closes the gap between the class-level protocol and the required per-sample provenance without creating a performance result.

## Frozen evidence

- Dataset: NKSID, 2,617 images, eight classes.
- Units: five outer folds x seeds 42, 43 and 44 = 15 paired units.
- Inner-validation fraction: 0.15.
- Manifest: `open_long_tail_sample_manifest_v1.json.txt`.
- Manifest SHA256: `59878d48786129c983e976b1cf8f4fc03bda79bd9e05ec5671ab42dedc1f7a3e`.
- Builder SHA256: `097c6e27e34253d66c111a6d4b43e3ae8b3cf88e961cd38833ea896c0996b087`.
- Independent audit: `open_long_tail_sample_manifest_v1_audit.json.txt`.
- Audit SHA256: `63043398d989e10da319ab1a70bafa8204651e279bbf9cce28269ef1eb5f759f`.
- Auditor SHA256: `c14a21d702099593138d361cc69001c500f729187a9ef78b11d22837f88305c9`.

## Independent checks passed

- All 2,617 image paths, labels, sizes and SHA256 values were independently reverified.
- All 15 train/inner/outer memberships were reconstructed from the canonical split implementation.
- Unknown classes are absent from known-only training and threshold-calibration subsets.
- Outer known and unknown subsets are disjoint and exactly cover each outer fold.
- A fold's outer membership is invariant across seeds; the five outer folds are disjoint and cover all 2,617 samples.
- All class-level and evaluation protocols, dataset split files, source-freeze manifest and retained source archive remain hash-bound.

## Evidence boundary and next integration step

The manifest has `claimability_status=FROZEN_INPUT_NOT_RESULT` and adds zero formal result units. R4, T3 and F5 remain `PENDING`. After the active G1 source-frozen chain finishes, the canonical `run_eval_protocol.py` path must require and persist this exact manifest SHA for every CE+MSP, DMCL and PLUD formal record; the independent result auditor must reject any missing or mismatched binding.

## Result-auditor contract

`audit_open_run.txt` now fails closed unless the immutable run configuration binds both this manifest and its independent audit. For every fold-seed unit it additionally requires exact outer-sample order, frozen target labels, class protocol, base split SHA, record/provenance binding and row-level manifest SHA.

- Auditor SHA256: `a179bb8880ddf82ed44a8b777ce5b5dece6bb957c51b5a3014bf9675ddf2f773`.
- Bound non-scientific one-unit fixture: `PASS`, audit SHA256 `a12d4c87b9e16b459227d469423bb206ab758012be39fe1ad532a8f3dbbcdc89`.
- Unbound CE+MSP smoke: correctly rejected with `open-set sample manifest/audit binding is invalid`; rejection artifact SHA256 `19a418a97df52197600170f4cd7b7761f2379fda7e92596efabcbaa0d5fb7f46`.
- Both fixtures have `formal_result_increment=0`; they prove the audit contract only.
