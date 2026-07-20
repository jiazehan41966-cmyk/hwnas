# T1 v2 reproducibility audit patch: 2026-07-17

## Outcome

The staged T1 replacement now joins paper registry/source audit, source smoke,
integration smoke or bounded method contract, and six dedicated environment cards
by `paper_id`. It contains five main papers plus the supplementary PLUD method,
26 columns, explicit local unified-protocol status and non-empty blockers for all
currently non-eligible rows.

This is a runtime-only staged replacement. It does not edit the canonical Python
builder or the active G1 source freeze. Canonical promotion remains a post-G1 task.

## Fail-closed rules verified

- source/environment/smoke PASS never implies formal numerical eligibility;
- every `formal_numeric_eligible=False` row has at least one explicit blocker;
- HW-PR-NAS remains an incomplete official release and its local smoke is labelled
  paper-spec/non-claimable;
- SURE, DMCL and PLUD remain isolated while redistribution licenses are unverified;
- HARP remains blocked on project complete-network LLVM graphs and grouped proxy
  evaluation;
- ESDA remains class C and `no_cross_platform_ranking`;
- CSV, Markdown and LaTeX are regenerated from the same ordered 6x26 dataset.

## Independent audit

`t1_v2_independent_audit_20260717.json.txt` reports:

- status `PASS`;
- rows `6`, columns `26`;
- main papers `5`, supplementary papers `1`;
- formal-numerical-eligible rows `0`;
- explicit-blocker rows `6`;
- input/environment bindings `16`;
- output bindings `3`;
- cross-format identity `PASS`;
- errors `[]`.

## Artifact hashes

| Artifact | SHA256 |
|---|---|
| `runtime/build_t1_reproducibility_v2.txt` | `d66ea11314e09c2bc3d0ac46d1130d3ff781381faf7c1915d642c81c1f1d46d4` |
| `runtime/audit_t1_reproducibility_v2.txt` | `9f2ef5b2b2c85891eed86b59c5e56f81b7fc6ceec26a9d563caf87e79b75950f` |
| `tables/t1_v2.csv` | `5c9c3e7935e137410113b1fabd40148e37cb9b9a6827941bca2cc5aec78d10d4` |
| `tables/t1_v2.md` | `26a49ebf7393a31884dc182abbf19a27cec72a314ae55046bcc971c4bdf4979c` |
| `tables/t1_v2.tex` | `e0f9c4b2f26ff3135944d98c8297a9cd569698b7bba5afec10b7988e995132be` |
| `tables/t1_v2_meta.json.txt` | `8aadc5038540003968e6da4bdc369da6742e9479fedd0563a554c2ff8d31bdfd` |
| `manifests/t1_v2_independent_audit_20260717.json.txt` | `cfa037e5af40c9006dcf50a020d07e276856ec69098f35f9840b700e0ecac8fa` |

## Evidence boundary

The T1 v2 audit establishes source/license/environment/reproducibility accounting,
not model superiority or paper reproduction. Numerical comparison remains blocked
until each direction completes its specified local unified protocol. The formal
`t1.*` filenames and readiness manifest must not be promoted from this staged
artifact until the canonical builder is updated and a new source freeze is made.
