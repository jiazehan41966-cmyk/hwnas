# T1 reproducibility-coverage gap audit: 2026-07-17

## Scope and claim boundary

This is a read-only reconciliation of the existing T1 bundle against the frozen
benchmark acceptance criteria. It does not change source, regenerate T1, or make
any paper reproduction claim while the formal G1 software chain is running.

Authoritative inputs inspected:

- `manifests/source_audit.json`
- `manifests/source_smoke.json`
- `environment/index.json` and the six paper-specific environment cards
- `tables/t1.csv`
- `scripts/build_benchmark_artifacts.py::_source_audit_rows`

## Evidence currently present

All five main papers have a pinned paper URL, official repository URL, observed
commit matching the registry pin, license state, paper-to-code correspondence
state, source audit, and a dedicated environment whose probe reports `PASS`.
The supplementary PLUD method has the same environment and source records.

| Paper ID | Commit pin | License | Paper-code correspondence | Source smoke | Dedicated environment probe | Formal local result |
|---|---:|---|---|---|---|---|
| `hw_pr_nas_2023` | PASS | MIT verified | partial | `BLOCKED_OFFICIAL_CODE_INCOMPLETE` | PASS | not completed |
| `sure_2024` | PASS | missing / redistribution unverified | verified | PASS | PASS | not completed |
| `harp_2023` | PASS | BSD-3-Clause verified | verified | PASS | PASS | not completed |
| `esda_2024` | PASS | MIT verified | verified | PASS | PASS | C-class design reference only |
| `dmcl_sonar_oltr_2025` | PASS | missing / redistribution unverified | partial | `PASS_SOURCE_PRESENT` | PASS | not completed |
| `plud_sonar_oltr_2024` | PASS | missing / redistribution unverified | verified | `PASS_SOURCE_PRESENT` | PASS | not completed |

`PASS` for a source smoke or environment probe means only that the pinned source
or adapter runtime is present and executable at the tested boundary. It is not a
reproduced paper result and does not make the row formally eligible.

## T1 coverage defects

The current T1 row builder copies only source-audit fields. It omits:

1. source-smoke status and the missing official HW-PR-NAS files;
2. dedicated-environment isolation status, probe status, interpreter and lock SHA;
3. local unified-protocol execution status;
4. an explicit `local_unified_protocol_not_completed` blocker when source-level
   blockers are otherwise empty;
5. the distinction between author-code reproduction, paper-spec adapter and
   C-class experimental-design reference.

Consequently, HARP and ESDA currently show `formal_eligible=False` with an empty
`blockers` cell. This is fail-closed in code but ambiguous to a reader.

## Required T1 repair after the active source freeze

The next T1 generator revision must join source audit, source smoke, environment
card and local run registry by `paper_id`. The formal columns must include at
least:

`paper_id, registry_role, direction, venue, comparability_class, repo_url,
observed_commit, pin_match, license_state, redistribution_allowed,
paper_code_correspondence, source_smoke_status, environment_probe_status,
environment_lock_sha256, local_protocol_status, execution_role,
numerical_comparison_rule, formal_eligible, blockers`.

Rules:

- HW-PR-NAS must be labelled `paper_spec_adapter`; it must never be described as
  an executable author-method reproduction at the pinned commit.
- SURE, DMCL and PLUD remain isolated and must not be redistributed while their
  repository license is unverified.
- ESDA remains class C and cannot enter cross-platform AV7K325 numerical ranking.
- A source or environment PASS cannot clear `formal_eligible` without the required
  complete local unified-protocol run and its independent audit.
- T1 CSV, Markdown and LaTeX must be regenerated from one joined source dataset;
  hand-editing only one representation is prohibited.

The source generator must not be changed during the active G1 frozen run. After
the software chain closes, revise the generator, add unit tests for the join and
fail-closed blocker behavior, create a new source freeze, and regenerate all three
T1 representations from that freeze.

## Staged v2 closure without changing the frozen source

A runtime-only staged replacement now demonstrates the required repair without
editing the canonical Python generator. `tables/t1_v2.csv`, `.md` and `.tex` are
generated from one 6-row x 26-column joined dataset. The independent audit
revalidates four top-level source files, six environment cards, six lock files and
three outputs, then reconstructs Markdown and LaTeX from the CSV rows.

Observed audit result: `PASS`, five main papers plus one supplementary paper,
six explicit-blocker rows, and zero formal-numerical-eligible rows. HW-PR-NAS is
explicitly blocked as an incomplete author release, HARP requires project
complete-network graphs and grouped evaluation, ESDA remains class C with
cross-board ranking prohibited, and all unlicensed repositories remain isolated.

This resolves the table-design and joined-evidence defect in a staged artifact,
but not the canonical-source integration requirement. After G1, migrate the same
logic into the tested canonical builder, create a new source freeze, promote the
v2 schema to the formal `t1.*` filenames, and rebuild the campaign readiness
manifest. Until then the old `t1.*` files remain the canonical snapshot and
`t1_v2.*` is the independently verified replacement candidate.
