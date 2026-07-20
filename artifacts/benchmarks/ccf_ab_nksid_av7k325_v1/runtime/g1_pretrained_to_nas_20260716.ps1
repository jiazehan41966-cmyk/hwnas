param(
    [int]$PretrainedPid = 18120
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
$python = Join-Path $repo '.venv_cuda\Scripts\python.exe'
$freeze = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\source_freeze\g1_20260715_v2\source_freeze_manifest.json'
$summary = Join-Path $repo 'results\protocol\g1_clean_20260711\g1_mobilenet_v2_grayscale_imagenet\protocol_summary.json'
$pretrainedRunDir = Join-Path $repo 'results\protocol\g1_clean_20260711\g1_mobilenet_v2_grayscale_imagenet'
$auditScript = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\runtime\audit_g1_run.py'
$auditOutput = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\manifests\g1_pretrained_independent_audit_20260716.json'
$candidate = Join-Path $repo 'hls_lut_builder\board_harness\results\pareto_route_gate_phase0_v4_sonar_stage3_k3_lowdsp\candidates\003_rl_arch_135.candidate.json'
$statusPath = Join-Path $repo 'results\protocol\g1_clean_20260711\g1_pretrained_to_nas_20260716.json'
$stdout = Join-Path $repo 'logs\g1_clean_20260711\nas_champion_20260716_stdout.log'
$stderr = Join-Path $repo 'logs\g1_clean_20260711\nas_champion_20260716_stderr.log'

function Write-Status {
    param([string]$Status, [hashtable]$Extra = @{})
    $payload = [ordered]@{
        schema_version = 1
        updated = (Get-Date).ToString('o')
        status = $Status
        pretrained_pid = $PretrainedPid
        source_freeze_manifest = $freeze
        nas_run_name = 'g1_rl_arch_135_legacy_selected'
    }
    foreach ($key in $Extra.Keys) {
        $payload[$key] = $Extra[$key]
    }
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

Write-Status -Status 'WAITING_FOR_PRETRAINED'
while (Get-Process -Id $PretrainedPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

if (-not (Test-Path -LiteralPath $summary -PathType Leaf)) {
    Write-Status -Status 'STOPPED_PRETRAINED_SUMMARY_MISSING'
    exit 20
}

$report = Get-Content -LiteralPath $summary -Raw | ConvertFrom-Json
$claimability = $report.claimability
$runCount = @($report.runs).Count
$ready = (
    $claimability.claimable -eq $true -and
    $claimability.protocol_complete -eq $true -and
    $claimability.source_freeze_verified -eq $true -and
    $claimability.observed_run_count -eq 15 -and
    $runCount -eq 15
)
if (-not $ready) {
    Write-Status -Status 'STOPPED_PRETRAINED_NOT_CLAIMABLE' -Extra @{
        claimable = $claimability.claimable
        protocol_complete = $claimability.protocol_complete
        source_freeze_verified = $claimability.source_freeze_verified
        observed_run_count = $claimability.observed_run_count
        summary_run_count = $runCount
    }
    exit 21
}

& $python $auditScript --run-dir $pretrainedRunDir --output $auditOutput --expect-pretrained true --expected-method imagenet_pretrained_mobilenet_v2
if ($LASTEXITCODE -ne 0) {
    Write-Status -Status 'STOPPED_PRETRAINED_INDEPENDENT_AUDIT_FAILED' -Extra @{
        audit_returncode = $LASTEXITCODE
        audit_output = $auditOutput
    }
    exit 24
}

& $python (Join-Path $repo 'scripts\freeze_experiment_source.py') verify --manifest $freeze
if ($LASTEXITCODE -ne 0) {
    Write-Status -Status 'STOPPED_SOURCE_FREEZE_FAILED' -Extra @{ verify_returncode = $LASTEXITCODE }
    exit 22
}
if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
    Write-Status -Status 'STOPPED_CANDIDATE_MISSING' -Extra @{ candidate = $candidate }
    exit 23
}

$arguments = @(
    'run_eval_protocol.py',
    '--data-dir', 'data/NKSID',
    '--output-dir', 'results/protocol/g1_clean_20260711',
    '--campaign-id', 'ccf_ab_nksid_av7k325_v1',
    '--paper-id', 'project_internal',
    '--folds', '0,1,2,3,4',
    '--seeds', '42,43,44',
    '--epochs', '150',
    '--batch-size', '8',
    '--gradient-accumulation-steps', '4',
    '--num-workers', '0',
    '--amp',
    '--save-checkpoints',
    '--resume',
    '--source-freeze-manifest', $freeze,
    '--candidate-path', $candidate,
    '--selection-provenance', 'legacy_fold0_selected',
    '--method-id', 'frozen_nas_champion',
    '--run-name', 'g1_rl_arch_135_legacy_selected'
)

Write-Status -Status 'STARTING_NAS' -Extra @{
    pretrained_claimable = $true
    pretrained_run_count = 15
    pretrained_independent_audit = $auditOutput
    candidate = $candidate
    stdout_log = $stdout
    stderr_log = $stderr
}
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -Wait -PassThru
if ($process.ExitCode -eq 0) {
    Write-Status -Status 'NAS_PROCESS_COMPLETED' -Extra @{ nas_returncode = 0 }
    exit 0
}
Write-Status -Status 'NAS_PROCESS_FAILED' -Extra @{ nas_returncode = $process.ExitCode }
exit $process.ExitCode
