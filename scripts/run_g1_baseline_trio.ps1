# G1 baseline trio: sequential frozen-protocol training on the CUDA venv.
# Order: rl_arch_135 (fastest, early signal) -> grayscale-pretrained MNV2
# (decision-critical) -> scratch MNV2. Each run resumes per (fold, seed)
# record, so re-running this script after an interruption is safe.

$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$repo = "E:\1\hwnas\hwnas"
$py = Join-Path $repo ".venv_cuda\Scripts\python.exe"
$logDir = Join-Path $repo "results\protocol"
New-Item -ItemType Directory -Force $logDir | Out-Null
$status = @{}

$common = @(
    "--data-dir", "data\NKSID",
    "--output-dir", "results\protocol",
    "--folds", "0,1,2,3,4",
    "--seeds", "42,43,44",
    "--epochs", "150",
    "--batch-size", "8",
    "--gradient-accumulation-steps", "4",
    "--amp",
    "--save-checkpoints",
    "--resume",
    "--device", "cuda"
)

$jobs = @(
    @{
        name = "g1_rl_arch_135_legacy_selected";
        args = @(
            "--candidate-path",
            "hls_lut_builder\board_harness\results\pareto_route_gate_phase0_v4_sonar_stage3_k3_lowdsp\candidates\003_rl_arch_135.candidate.json",
            "--selection-provenance", "legacy_fold0_selected"
        )
    },
    @{
        name = "g1_mobilenet_v2_grayscale_imagenet";
        args = @("--arch", "mobilenet_v2", "--pretrained",
                 "--selection-provenance", "baseline_predeclared")
    },
    @{
        name = "g1_mobilenet_v2_scratch";
        args = @("--arch", "mobilenet_v2",
                 "--selection-provenance", "baseline_predeclared")
    }
)

Set-Location $repo
foreach ($job in $jobs) {
    $name = $job.name
    $log = Join-Path $logDir "$name.launcher.log"
    "[launcher] $(Get-Date -Format s) starting $name" | Out-File -Append -Encoding utf8 $log
    & $py (Join-Path $repo "run_eval_protocol.py") @common @($job.args) `
        "--run-name" $name *>> $log
    $status[$name] = $LASTEXITCODE
    "[launcher] $(Get-Date -Format s) finished $name exit=$($status[$name])" |
        Out-File -Append -Encoding utf8 $log
}

$statusPath = Join-Path $logDir "g1_baseline_trio_status.json"
$status | ConvertTo-Json | Out-File -Encoding utf8 $statusPath
