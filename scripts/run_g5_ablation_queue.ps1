# G5 sonar four-way ablation queue (E1 of docs/SONAR_OPERATOR_G5_EXPERIMENT_PLAN.md).
#
# Self-contained chain so a single detached process owns the whole hand-off:
#   A. wait for any foreign run_eval_protocol.py process to exit
#   B. finish the missing G1 records (finish_g1_missing.ps1 skips existing ones)
#   C. rebuild missing G1 protocol summaries from records (offline aggregation)
#   D. verify all three G1 summaries exist (abort loudly if not)
#   E. run the four matched ablation variants sequentially
#
# Every training run resumes per (fold, seed) record; re-running this script
# after an interruption or reboot is safe. Variant order: control first.

$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$repo = "E:\1\hwnas\hwnas"
$py = Join-Path $repo ".venv_cuda\Scripts\python.exe"
$logDir = Join-Path $repo "results\protocol"
New-Item -ItemType Directory -Force $logDir | Out-Null
$queueLog = Join-Path $logDir "g5_ablation_queue.launcher.log"

function Write-QueueLog($message) {
    "[queue] $(Get-Date -Format s) $message" | Out-File -Append -Encoding utf8 $queueLog
}

Set-Location $repo

# --- A. never race a foreign protocol run ---
Write-QueueLog "phase A: waiting for existing run_eval_protocol.py processes"
while ($true) {
    $running = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
        Where-Object { $_.CommandLine -like "*run_eval_protocol.py*" }
    if (-not $running) { break }
    Start-Sleep -Seconds 300
}

# --- B. finish missing G1 records (script skips records already present) ---
Write-QueueLog "phase B: finishing missing G1 records via finish_g1_missing.ps1"
& (Join-Path $repo "scripts\finish_g1_missing.ps1")
Write-QueueLog "phase B done"

# --- C. rebuild missing G1 summaries from assembled records ---
$g1Runs = @(
    "g1_rl_arch_135_legacy_selected",
    "g1_mobilenet_v2_grayscale_imagenet",
    "g1_mobilenet_v2_scratch"
)
foreach ($run in $g1Runs) {
    $summary = Join-Path $logDir "$run\protocol_summary.json"
    if (-not (Test-Path $summary)) {
        Write-QueueLog "phase C: finalizing summary for $run"
        & $py (Join-Path $repo "scripts\finalize_protocol_summary.py") `
            --run-dir (Join-Path $logDir $run) *>> $queueLog
    }
}

# --- D. hard gate: all three summaries must exist before E1 starts ---
$missing = $g1Runs | Where-Object { -not (Test-Path (Join-Path $logDir "$_\protocol_summary.json")) }
if ($missing) {
    Write-QueueLog "phase D FAILED: missing summaries: $($missing -join ', '); aborting queue"
    exit 1
}
Write-QueueLog "phase D: all G1 summaries present; starting E1"

# --- E. four matched ablation variants ---
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

$variants = @("mbconv_control", "denoise", "edge", "denoise_edge")
$status = @{}

foreach ($variant in $variants) {
    $name = "g5_ablation_$variant"
    $log = Join-Path $logDir "$name.launcher.log"
    Write-QueueLog "starting $name"
    "[launcher] $(Get-Date -Format s) starting $name" | Out-File -Append -Encoding utf8 $log
    & $py (Join-Path $repo "run_eval_protocol.py") @common `
        "--candidate-path" "configs\ablation\sonar_g5_v1\$variant.candidate.json" `
        "--run-name" $name *>> $log
    $status[$name] = $LASTEXITCODE
    "[launcher] $(Get-Date -Format s) finished $name exit=$($status[$name])" |
        Out-File -Append -Encoding utf8 $log
    Write-QueueLog "finished $name exit=$($status[$name])"
}

$statusPath = Join-Path $logDir "g5_ablation_queue_status.json"
$status | ConvertTo-Json | Out-File -Encoding utf8 $statusPath
Write-QueueLog "queue complete; status written to $statusPath"
