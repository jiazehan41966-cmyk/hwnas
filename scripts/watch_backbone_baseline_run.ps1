param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot),
    [int]$PollSeconds = 60
)

$ErrorActionPreference = 'Stop'

Set-Location $Workspace

$resolvedRunDir = Resolve-Path $RunDir
$runInfoPath = Join-Path $resolvedRunDir 'run_info.json'
$summaryPath = Join-Path $resolvedRunDir 'results\backbone_summary.json'
$outputPath = Join-Path $resolvedRunDir 'results\fair_macro_backbone_table.md'
$watchLogPath = Join-Path $resolvedRunDir 'logs\watch_backbone_baseline.log'

Add-Content -Path $watchLogPath -Value "[$(Get-Date -Format s)] watcher started"

while ($true) {
    if (Test-Path $summaryPath) {
        break
    }

    if (Test-Path $runInfoPath) {
        try {
            $runInfo = Get-Content $runInfoPath -Raw | ConvertFrom-Json
            if ($runInfo.status -eq 'failed') {
                Add-Content -Path $watchLogPath -Value "[$(Get-Date -Format s)] run failed; watcher exiting"
                exit 1
            }
        }
        catch {
            Add-Content -Path $watchLogPath -Value "[$(Get-Date -Format s)] warning: failed to parse run_info.json"
        }
    }

    Start-Sleep -Seconds $PollSeconds
}

Add-Content -Path $watchLogPath -Value "[$(Get-Date -Format s)] summary detected; generating markdown table"
python scripts/generate_backbone_baseline_table.py $resolvedRunDir --output $outputPath | Out-Null
Add-Content -Path $watchLogPath -Value "[$(Get-Date -Format s)] watcher finished"
