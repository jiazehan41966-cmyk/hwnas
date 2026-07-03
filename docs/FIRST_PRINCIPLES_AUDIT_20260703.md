# HW-NAS First-Principles Audit

Status: active, evidence snapshot 2026-07-03.

This audit does not inherit earlier project conclusions. Current source code,
configuration files, raw artifacts, and reproducible checks are the evidence
base. Existing results are retained, but their claim strength is recalibrated
against the protocol that produced them.

## Decision-changing findings

### 1. Reported classification scores are not final generalization estimates

The current search and retrain path uses NKSID fold record 0 for architecture
screening, best-epoch selection, and final validation reporting. The v4 search
compared 300 candidates on those same 520 validation images, and retrain150
again selected the best epoch on the same images. There is no untouched test
set in this path and only seed 42 is present for the reported Top-K retrains.

Therefore, values such as v4 `rl_arch_135` macro_f1 `0.860728` and top1
`0.921154` remain valid descriptions of the legacy fold-0 protocol. They are
not yet defensible estimates of deployment-time generalization or evidence of
a statistically stable improvement over v3.

### 2. The supplied split is repeated 5-fold, not a single 10-fold protocol

`train_abs.txt` contains 2,617 readable samples with no missing path, duplicate
path, or exact duplicate file content. `kfold_train.txt` and `kfold_val.txt`
each contain 50 split records. Their sizes imply 5-fold validation, and every
sample appears in validation exactly 10 times: the files encode 10 repeated
5-fold cycles.

The production config uses only record 0. In that record, 498/520 validation
images (95.8%) have a same-class image with filename number `N-1` or `N+1` in
training. This is a strong image-level adjacency leakage risk. It is not proof
of acquisition-sequence leakage because mission/sequence metadata is absent.
That metadata is now required before the protocol can be called group-safe.

Machine-readable evidence:
`artifacts/first_principles_audit_20260703/evidence_summary.json`. The full
generated output remains at
`results/first_principles_audit_20260703/nksid_protocol_audit.json`.

### 3. Train-time sonar operators and HLS operators are different algorithms

PyTorch `DenoiseBlock` computes a learned depthwise feature branch plus a
learned normalized smoothing branch, then pointwise projection and an optional
residual. The admitted low-DSP HLS `denoise` template contains one spatial
branch and no residual.

PyTorch `EdgeAwareBlock` computes four trainable directional depthwise branches,
concatenates 4C channels, and fuses them. The admitted low-DSP HLS `edge`
template contains one spatial kernel per channel and fuses C channels.

Consequently, existing HLS, full-route, and COM5 artifacts establish feasibility
of simplified HLS pipelines, not deployment of the trained PyTorch operators.
The current HLS cost is especially optimistic for `edge`. The canonical
MobileNetV2 search config now explicitly loads
`hls_lut_builder/configs/operator_manifest_semantic_safe.yaml`, which blocks
both operators from new claimable searches until a matched numeric-parity and
weight-export gate exists. Historical configs and artifacts are retained
unchanged for reproducibility.

Machine-readable evidence:
`artifacts/first_principles_audit_20260703/evidence_summary.json`. The full
generated output remains at
`results/first_principles_audit_20260703/operator_semantic_parity_audit.json`.

### 4. Power and energy are not measured objectives

Search-time `power_w` is computed by a fixed linear heuristic from peak DSP,
BRAM, and LUT. `energy_mj` is that estimate multiplied by estimated latency.
Vivado power is a separate estimate. No external board power trace has been
imported, so measured power and measured energy remain unavailable and cannot
be used to claim a co-optimal solution.

### 5. Test coverage exists, but the default test command was not isolated

The project tests pass when invoked as `pytest tests`: 193 tests plus 28
subtests passed before this audit. Plain `pytest` previously collected tests
inside `reference/FBNet` and stopped at 50 missing-dependency errors. The
project also had no root packaging or dependency manifest. `pyproject.toml`
now defines the package, bounded dependencies, and project-only pytest roots.

### 6. The XC7K325T board profiles used slices as LUT capacity

The board profiles used `50,950` as `max_lut`. AMD documents `50,950` slices
and four LUTs per slice, for `203,800` 6-input LUTs. The `kintex7_xc7k325` and
`alinx_av7k325` board profiles and hardware YAML files now use `203,800`.

The active canonical MobileNetV2 search config also uses the corrected physical
capacity. Historical experiment configs and their recorded results are not
rewritten; their explicit `50,950` constraints remain part of the protocol that
produced those artifacts. This correction does not retroactively validate an
old Pareto front.

Primary source:
`https://docs.amd.com/r/en-US/ug474_7Series_CLB/7-Series-FPGA-CLB-Resources`.

## Immediate implementation changes

- Added a reproducible `pyproject.toml` and project-scoped pytest discovery.
- Added a reusable NKSID integrity/split audit and tests.
- Made corrupt NKSID samples fail closed by default instead of silently
  substituting a blank image with the original label.
- Corrected `DenoiseBlock` so the effective initial smoothing kernel is
  actually Gaussian after softmax parameterization.
- Added a semantic-safe operator policy that pauses `denoise` and `edge`.
- Corrected the XC7K325T physical LUT capacity while preserving historical
  experiment configs.
- Published a compact, reproducible audit summary under `artifacts/`.

## Verification

`python -m pytest -q` completed with `199 passed, 28 subtests passed` on
2026-07-03. Both audit scripts also reproduced the generated JSON/Markdown
files with SHA256 hashes matching
`artifacts/first_principles_audit_20260703/evidence_summary.json`.

## Revised optimization order

1. Freeze an evaluation protocol with acquisition-group-safe outer test folds.
2. Use inner folds only for architecture/epoch selection; report outer-fold
   macro_f1 and top1 with uncertainty over folds and seeds.
3. Admit only operators with PyTorch-to-fixed-point numeric parity, matching
   parameter export, HLS synthesis, and full-route evidence.
4. Calibrate latency/resource surrogates using held-out routed architectures,
   reporting prediction error rather than only LUT hit rate.
5. Add calibrated INT8 validation accuracy before full validation-set board
   inference.
6. Measure board power traces and energy per inference before including power
   in the final Pareto objectives.

Until these gates are complete, the strongest current hardware statement is:
multiple simplified full-network harnesses route and produce stable fixed-input
COM5 latency/output measurements. The strongest classification statement is:
the reported scores are single-seed, best-epoch validation results under an
image-index split with high adjacency-leakage risk.
