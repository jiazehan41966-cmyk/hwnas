# T7 board-latency collection contract repair: 2026-07-17

## Problem found

The original board-latency CSV recorded only SHA strings for bitstream, payload
and parity evidence. The auditor checked their syntax but did not bind the files,
did not prove one fixed candidate per deployment role, did not require paired
validation samples across candidates, and imposed no minimum inference count.
Such a CSV could not support T7/F9/F10 provenance or a paired three-candidate
comparison.

## Repair applied

Every board row now binds:

- paper/method, candidate ID/SHA and the common candidate-selection manifest;
- candidate manifest and checkpoint;
- source-freeze manifest, project commit and code-state SHA;
- data, split and validation-manifest SHA;
- route report, bitstream, per-inference payload and parity summary paths/SHA;
- toolchain fingerprint, board target/serial, COM port and baud;
- host timestamps, latency, board cycles/clock, CRC/numeric status, temperature,
  target, prediction, correctness and claimability.

The auditor hashes every referenced file with a cache, enforces exactly one
distinct candidate for each of `accuracy_first`, `knee_point` and `resource_min`,
requires role-constant provenance, checks identical `sample_id/target` mappings
across roles, validates latency against host timestamps, and requires at least
1,000 inference rows per role. Transport or numeric failures remain valid rows for
the reported error rate; provenance/schema failures invalidate the collection.

## Contract tests

The updated auditor compiles, and the header-only template remains correctly
fail-closed with zero rows and `schema_and_provenance_pass=false`.

A temporary 3,000-row campaign with three roles x 1,000 paired samples passed:

`candidate IDs = {cand_accuracy_first, cand_knee_point, cand_resource_min},
paired_sample_sets_equal=true, bound_file_count=15,
schema_and_provenance_pass=true, errors=[]`.

Three injected failures were independently rejected:

1. a second candidate ID within `accuracy_first` triggered the one-candidate-per-
   role and three-distinct-candidate errors;
2. one target label changed for only one role triggered the paired sample-map
   error;
3. 999 rows per role triggered the minimum-1,000-inference error.

## Frozen artifact hashes

| Artifact | SHA256 |
|---|---|
| `runtime/audit_hardware_collection.txt` | `1a48570a4d2dd016e5e7cb14ae77019bb6e398793cbabbacc198eb522137ccf4` |
| `runtime/board_latency_sample_template.csv` | `cb9bd9ba58de2b8fdeedf84d5725c9b3c0985a0a503c1c76443fa2e41f4ddab4` |
| `runtime/hardware_collection_runbook_20260716.md` | `3c72d57aad423f1e96c8e01290cb2b82057c48aca0505169f5910c43559ba45b` |

## Claim boundary

This repairs a collection contract only. No AV7K325 inference row was measured,
no candidate has been programmed, and T7/F9/F10 remain `PENDING`. Passing the
latency CSV audit will still not replace the separate full-validation accuracy
audit, route/parity acceptance or external-power protocol.
