# FCOS/UATD experiment 2: preregistration amendment v2

The immutable machine registration is `configs/sonar_fcos_quality_prereg_v2.json`.
Its timestamp follows the first three committed AAA seed42 epochs, but precedes
inspection of new validation values or any paired outcome. V1 remains intact.
Migration verifies model, optimizer, scaler, RNG, history and committed epoch
state exactly, and counts the migrated work inside the original six-run budget.

Both best-validation-mAP and fixed epoch30 must pass all four native gates:
mean paired AP75 gain >=2 percentage points; mean NMS first-loss reduction
>=25%; strictly fewer final-no-good GTs; no increase in raw-no-good GTs.
Exactly one passing endpoint is NON_ROBUST. Epochs26..30 are saved individually
and their separately evaluated arithmetic mean is a supplementary endpoint.

Fixed score0.25 and target FPPI0.5 diagnostics are both retained. The latter
uses whole tied-score groups under floor(0.5*N) false positives and reports the
actual rate and shortfall. Native NMS/top-k stages remain unchanged. AP uses
all saved native predictions. Paired AP75 CIs use 2000 common image draws
(PCG64 seed20261012), including across late epochs; duplicate images are
independent copies in sampled order. The CI is supplementary, not a new gate.

GPU1 is reserved. GPU0/2/3 may run only when truly idle, with seed-pair affinity,
exclusive claims and cooperative yielding. Workers perform one epoch per
process. The CPU supervisor summarizes partial completions and runs final
analysis once after all six complete. No seventh run or C01c is scheduled.

From the local repository, using the existing credential-safe transport:

```powershell
.venv_cuda/Scripts/python.exe -B -m tools.sonar_fcos_quality_v2.remote stage
.venv_cuda/Scripts/python.exe -B -m tools.sonar_fcos_quality_v2.remote preflight
.venv_cuda/Scripts/python.exe -B -m tools.sonar_fcos_quality_v2.remote status
.venv_cuda/Scripts/python.exe -B -m tools.sonar_fcos_quality_v2.remote migrate
.venv_cuda/Scripts/python.exe -B -m tools.sonar_fcos_quality_v2.remote upload-review
.venv_cuda/Scripts/python.exe -B -m tools.sonar_fcos_quality_v2.remote launch
```

Migration requires the parent queue stopped and preflight passed. Launch also
requires a fresh independent review bound to execution, migration and protocol
hashes. The commands above describe the ordered procedure; do not rerun an
existing stage/preflight/launch blindly. Status is safe to repeat.

Remote namespace: `/home/jiajia/experiments/sonar_fcos_quality_20261012_01_v2`.
Local evidence: `results/sonar_fcos_quality_20261012_01_v2`.
The finite local completion collector retrieves result tables and the archived
FCOS checkpoint to `F:/sonar_fcos_quality/pretrained/20261012_01_v2` once complete.
It does not launch training or create recurring OS jobs.

Claims are restricted to development validation on UATD. Acquisition-sequence
isolation remains unverified. This joint QFL+DFL adaptation is neither an
isolated scoring ablation nor evidence for the two-class E0 lightweight model.
FCOS weights are an archive only; separate E0(B1)+new-head UATD pretraining is
deferred, requiring later authorization after experiment1's conditional result.
