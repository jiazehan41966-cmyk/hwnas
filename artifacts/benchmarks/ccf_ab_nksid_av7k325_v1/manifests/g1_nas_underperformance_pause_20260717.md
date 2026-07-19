# G1 NAS Underperformance Pause

## Trigger

After fold 4 / seed 43 completed atomically, the NAS champion had 14 independently accepted fold-seed units. Against the same 14 ImageNet-pretrained MobileNetV2 units, its interim macro-F1 mean was `0.6904529534` versus `0.9300808161`; the paired mean difference was `-0.2396278627`. Every one of the 14 paired differences was negative, ranging from `-0.3250370668` to `-0.1621433112`. A 10,000-replicate paired bootstrap with seed `20260717` gave an interim 95% interval of `[-0.2612163648, -0.2169519822]`.

This is material underperformance, but the NAS method is still incomplete. The numbers are decision support only and are not released as T2/F6 evidence.

## Safe interruption

- Fold 4 / seed 43 finished with 537 predictions and passed the 14-unit independent partial audit.
- Partial-audit SHA256: `0e8da5427b19fb8ddf2aefb871ee4965839c21f17e294887b2668b50dc6e507c`.
- The two Python processes were stopped only after that atomic unit appeared.
- No fold 4 / seed 44 checkpoint, record or prediction file exists.
- Scheduled task `Codex_HWNAS_G1_20260716` is `Ready` with last result `2`; it is not running and has not entered SURE or scratch-v2.

## User decisions required

1. Whether to spend one final fold-seed run to close NAS at 15/15 and obtain a formally complete paired method result.
2. Whether SURE and scratch-v2 should continue after the NAS result is known.
3. Whether the weak NAS candidate should be diagnosed under the frozen protocol or replaced only through an explicit protocol amendment and new source freeze.

No automatic continuation is allowed until these decisions are recorded separately. Completing the last unit does not imply accepting the candidate as competitive, and changing the candidate cannot be mixed into the current frozen run.

## Fail-closed orchestration verification

The persistent wrapper now checks a SHA-bound user-decision artifact before both `resume_nas_to_15` and `continue_downstream_closed_set_chain`. Wrapper SHA256 is `3d0fc1addcf7ab47353331606d917ff7507b7e222a84b41532ddfc3fcc1d1094`.

A real scheduled-task invocation without an approval artifact was tested. It reverified the source freeze, logged `PAUSED_PENDING_USER_DECISION resume_nas_to_15`, exited with result `2`, retained exactly 14 predictions, and started no NAS training process. The test record is `g1_nas_decision_gate_test_20260717.json.txt`. The neighboring user-decision file is a non-executable template only; it cannot authorize work until an explicit user decision is recorded in the exact approval path.
