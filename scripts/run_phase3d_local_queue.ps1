$ErrorActionPreference = "Stop"
Set-Location e:\1\hwnas\hwnas
$py = ".\.venv_cuda\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path logs\phase3d_formal | Out-Null

function Invoke-B2([string]$Dataset, [int]$Seed) {
    $folder = if ($Dataset -eq "uatd_v3") { "UATD" } else { "SCTD" }
    $summary = "results\transfer\phase3d\$folder\b2_seed$Seed.json"
    if (Test-Path $summary) {
        Write-Output "SKIP $Dataset B2 seed$Seed already complete"
        return
    }
    $log = "logs\phase3d_formal\${Dataset}_B2_seed$Seed.log"
    Write-Output "START $Dataset B2 seed$Seed $(Get-Date -Format o)"
    & $py -u scripts\run_phase3d_structural_decomposition.py --mode formal --datasets $Dataset --seeds $Seed --device cuda --resume 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "B2 $Dataset seed$Seed failed with $LASTEXITCODE" }
    Write-Output "DONE $Dataset B2 seed$Seed $(Get-Date -Format o)"
}

function Wait-RunningSeed0 {
    $summary = "results\transfer\phase3d\SCTD\b2_seed0.json"
    while (-not (Test-Path $summary)) {
        Start-Sleep -Seconds 60
        Write-Output "waiting for SCTD B2 seed0 $(Get-Date -Format o)"
    }
}

Wait-RunningSeed0
Invoke-B2 "classic_sctd_1_0" 1
Invoke-B2 "classic_sctd_1_0" 2
Invoke-B2 "uatd_v3" 0
Invoke-B2 "uatd_v3" 1
Invoke-B2 "uatd_v3" 2
Write-Output "EVALUATE $(Get-Date -Format o)"
& $py -u scripts\run_phase3d_structural_decomposition.py --mode evaluate-valid --device cuda
if ($LASTEXITCODE -ne 0) { throw "evaluate-valid failed" }
Write-Output "ANALYZE $(Get-Date -Format o)"
& $py -u scripts\run_phase3d_structural_decomposition.py --mode analyze
if ($LASTEXITCODE -ne 0) { throw "analyze failed" }
Write-Output '{"status":"COMPLETE","b2_formal_runs":"6/6","phase3d_test_accessed":false}' | Set-Content -Encoding utf8 results\transfer\phase3d\PHASE3D_QUEUE_STATUS.json
Write-Output "QUEUE COMPLETE $(Get-Date -Format o)"
