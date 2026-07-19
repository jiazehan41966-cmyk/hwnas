param(
    [string]$DataDir = "data/NKSID",
    [string]$OutputDir = "results/protocol/g1_clean_20260711",
    [int]$Epochs = 150,
    [string]$SourceFreezeManifest
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\")).Path
$python = Join-Path $repo ".venv_cuda\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}
if (-not $SourceFreezeManifest) {
    throw "Formal G1 requires -SourceFreezeManifest from freeze_experiment_source.py."
}
$SourceFreezeManifest = (Resolve-Path -LiteralPath $SourceFreezeManifest).Path

# The pre-freeze patch run is intentionally not interrupted.  Its output is
# outside the clean root and is never copied or resumed into formal G1.
while ($true) {
    $patchProcesses = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine -match "run_eval_protocol\.py" -and
            $_.CommandLine -match "(?i)(^|[\\/ _-])_patch([\\/ _-]|$)"
        }
    if (-not $patchProcesses) {
        break
    }
    Start-Sleep -Seconds 30
}

$logDir = Join-Path $repo "logs\g1_clean_20260711"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$arguments = @(
    "scripts/run_g1_baselines.py",
    "--data-dir", $DataDir,
    "--output-dir", $OutputDir,
    "--epochs", $Epochs,
    "--source-freeze-manifest", $SourceFreezeManifest
)
& $python @arguments *>&1 |
    Tee-Object -FilePath (Join-Path $logDir "launcher.log")
exit $LASTEXITCODE
