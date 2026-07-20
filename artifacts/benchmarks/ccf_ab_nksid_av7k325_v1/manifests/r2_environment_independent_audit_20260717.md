# R2 dedicated-environment independent audit: 2026-07-17

## Result

R2 passes an independent filesystem and manifest-consistency audit for all six
registered main/supplementary paper runtimes. No CUDA probe was re-executed during
the active G1 training process; this audit verifies the already archived probes,
paths, hashes and counts without contending for the GPU.

## Checks applied to every environment card

- paper-specific JSON card exists and is listed by `environment/index.json`;
- `pinned_commit == observed_commit`;
- isolation state equals `READY_DEDICATED_ENVIRONMENT`;
- archived dedicated-environment probe status equals `PASS`;
- probe-observed commit equals the card's pinned commit;
- dedicated interpreter exists at the recorded path;
- dependency lock file exists;
- independently recomputed lock SHA256 equals the recorded SHA256;
- non-empty lock-line count equals the recorded package count;
- verification fingerprint is a 64-character SHA256 value.

All conditions pass for six of six cards. There are six unique dedicated virtual
environment paths, so no paper ID is silently redirected to another paper's
runtime directory.

## Audited runtimes

| Paper ID | Archived probe | Packages | Lock SHA256 | Verification fingerprint |
|---|---:|---:|---|---|
| `hw_pr_nas_2023` | PASS | 16 | `e136caaa1776bef95b235119d73d883c69caebbf34a2ddda1befe3b17249be2a` | `bc048cbf8435b5ba294ef664663e28c9034705b40b5152576dc85b1b51615030` |
| `sure_2024` | PASS | 16 | `e136caaa1776bef95b235119d73d883c69caebbf34a2ddda1befe3b17249be2a` | `4b24934310c5b58734725a500c242256d53193364b8ce325c10dfbc80f695e99` |
| `harp_2023` | PASS | 16 | `e136caaa1776bef95b235119d73d883c69caebbf34a2ddda1befe3b17249be2a` | `020a7a423c7055820c9f4955c4d00384e810ee5b946f2d6a4f60f7f9757665dc` |
| `esda_2024` | PASS | 16 | `e136caaa1776bef95b235119d73d883c69caebbf34a2ddda1befe3b17249be2a` | `84e116545379c9827f7f27a781457dd515212d728371ffae84217885b5779813` |
| `dmcl_sonar_oltr_2025` | PASS | 16 | `e136caaa1776bef95b235119d73d883c69caebbf34a2ddda1befe3b17249be2a` | `a1fb5acc73d7a4c2ef16700dd3887a1ebecddfc738bcd6cdd8bb0e9778d571cf` |
| `plud_sonar_oltr_2024` | PASS | 16 | `e136caaa1776bef95b235119d73d883c69caebbf34a2ddda1befe3b17249be2a` | `b92827b87b0e59adfcb8864855e2a88769fedc33042bf1b0c3f7d69089b3a74b` |

The identical lock SHA is expected: the six adapters use the same frozen minimal
CUDA dependency set, while their isolated paths and method-specific probe
fingerprints remain distinct.

## Claim boundary

R2 proves that the registered adapter runtimes are present, isolated, pinned and
dependency-locked. It does not prove author-environment equivalence, reproduce an
author table, clear an unverified redistribution license, or make any local
benchmark result formally eligible. HW-PR-NAS remains a paper-spec adapter because
the pinned official repository is incomplete; SURE/DMCL/PLUD remain isolated due
to unverified redistribution licenses.
