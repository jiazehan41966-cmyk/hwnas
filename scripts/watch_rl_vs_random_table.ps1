param(
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot),
    [string]$RandomRun = 'results/random_baseline_mobile_anchor_shufflenet_av7k325_200_formal',
    [string]$RlRun = 'results/rl_mobile_anchor_shufflenet_av7k325_200_formal',
    [string]$OutputDir = 'results/paper_search_comparison',
    [int]$PollSeconds = 60
)

$ErrorActionPreference = 'Stop'

Set-Location $Workspace

$outputPath = Join-Path $Workspace $OutputDir
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

$logPath = Join-Path $outputPath 'watcher.log'
$rlRoot = Join-Path $Workspace $RlRun
$summaryPath = Join-Path $rlRoot 'results\summary.json'
$runInfoPath = Join-Path $rlRoot 'run_info.json'

Add-Content -Path $logPath -Value "[$(Get-Date -Format s)] watcher started"

while (-not (Test-Path $summaryPath)) {
    if (Test-Path $runInfoPath) {
        try {
            $runInfo = Get-Content $runInfoPath -Raw | ConvertFrom-Json
            if ($runInfo.status -eq 'failed') {
                Add-Content -Path $logPath -Value "[$(Get-Date -Format s)] RL run failed; watcher exiting"
                exit 1
            }
        }
        catch {
            Add-Content -Path $logPath -Value "[$(Get-Date -Format s)] warning: failed to parse run_info.json"
        }
    }

    Start-Sleep -Seconds $PollSeconds
}

Add-Content -Path $logPath -Value "[$(Get-Date -Format s)] summary detected; generating final comparison"

python scripts/generate_paper_search_table.py `
    $RandomRun `
    $RlRun `
    --output-dir $OutputDir `
    --caption "Comparison of Random and RL search under the same 200-evaluation budget on the NKSID mobile-anchor FPGA search space." `
    --label "tab:rl_vs_random_mobile_anchor_fpga" | Tee-Object -FilePath $logPath -Append

if ($LASTEXITCODE -ne 0) {
    Add-Content -Path $logPath -Value "[$(Get-Date -Format s)] comparison generation failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Add-Content -Path $logPath -Value "[$(Get-Date -Format s)] watcher finished"
