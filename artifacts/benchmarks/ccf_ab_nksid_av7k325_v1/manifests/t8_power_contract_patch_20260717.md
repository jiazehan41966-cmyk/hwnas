# T8 external-power collection contract repair: 2026-07-17

## Problem found

The canonical power calculator already enforced three idle/active blocks, 60-second
durations, at least 1,000 inferences, external board-input measurement and receipt
counts. It did not prove that raw active CSV timestamps aligned to the
`RUN_REPEAT` UTC interval, did not bind raw CSV/receipt hashes in the manifest,
did not bind the protocol fingerprint to a protocol file, and did not require the
three manifests to represent the three frozen candidate roles. A nominally
passing campaign could therefore bias dynamic energy by integrating pre/post-
active samples or combine unrelated candidates.

## Repair applied

The schema-v2 candidate manifest adds:

- candidate role and full selection/candidate/checkpoint provenance;
- source freeze, project commit/code state, data/split and route bindings;
- bitstream, payload and parity paths/SHA;
- a shared protocol-manifest path/SHA whose SHA is the protocol fingerprint;
- UTC Unix epoch timestamp basis;
- instrument serial, ranges and bound calibration certificate;
- SHA256 for every idle/active CSV and every RUN_REPEAT receipt.

The new `audit_power_campaign_v2.txt` supplements the canonical power calculator
and validates:

- exactly one manifest for each `accuracy_first`, `knee_point` and `resource_min`;
- three distinct candidate IDs under one selection manifest, source freeze,
  code/data/split and measurement protocol;
- identical instrument identity and calibration across candidates;
- protocol-file content matches instrument, rail, source and timestamp basis;
- every referenced artifact, raw CSV and receipt matches its SHA;
- observed CSV sample rate is within 5% of the instrument rate;
- active CSV first/last UTC epoch timestamps align to receipt
  `active_started_utc/active_finished_utc` within a sampling-aware tolerance;
- receipt UTC and monotonic durations agree, each active interval is at least
  60 seconds, repeat count is at least 1,000, and programming/UART upload are
  outside the interval;
- receipt bitstream/payload/parity SHA values match the candidate manifest.

The `.txt` suffix preserves the active G1 code fingerprint. Before final T8
acceptance, this logic must be migrated into tested canonical source under a new
source freeze.

## Contract tests

The v2 auditor compiles and its CLI loads. A temporary three-candidate campaign
used three idle plus three active 60-second blocks per candidate at 1 Hz. Idle was
5 W, active was 7 W, and each active block contained 1,000 inferences. The base
calculator and v2 audit both passed, producing the expected dynamic energy of
`120 mJ/inference` for all three candidates.

Independent negative cases were rejected:

1. shifting one receipt UTC interval by 10 seconds while leaving its power CSV
   unchanged produced `CSV does not align to receipt UTC interval`;
2. relabelling `resource_min` as a second `accuracy_first` failed the exact-role
   gate;
3. changing one instrument serial number failed both the bound-protocol instrument
   check and same-instrument campaign gate.

## Frozen artifact hashes

| Artifact | SHA256 |
|---|---|
| `runtime/audit_power_campaign_v2.txt` | `5a798f0524eb78d92c5769f2f606a9d2faeadb86533b919c6ce55c88ea448556` |
| `runtime/power_measurement_manifest_template.json.txt` | `f1d3f56c1344c3c826d62b7844e74f2555af19c6d7f805fb1e3154a689210774` |
| `runtime/power_measurement_protocol_template.json.txt` | `da4dc9f898b5ffa449786c7c95255077e7dc19c7ef8cf738966603aa7843571c` |
| `runtime/power_timeseries_template.csv` | `e490184df593f6959e58692811d4c8506f16c8275f02bc1901ca9bf637746433` |
| `runtime/hardware_collection_runbook_20260716.md` | `3c72d57aad423f1e96c8e01290cb2b82057c48aca0505169f5910c43559ba45b` |

## Claim boundary

This is a tested evidence contract, not measured power. No external instrument
command or raw meter trace is currently available; power remains `NOT_MEASURED`,
and T8/F11/F12 remain `PENDING`. GPU power and Vivado estimated power remain
diagnostics/proxies and cannot populate this schema.
