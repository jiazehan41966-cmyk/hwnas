$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
$python = Join-Path $repo '.venv_cuda\Scripts\python.exe'
$analysisPython = 'D:\software\python\python.exe'
$surePython = Join-Path $repo '.venv_benchmarks\sure_2024_cuda\Scripts\python.exe'
$dmclPython = Join-Path $repo '.venv_benchmarks\dmcl_sonar_oltr_2025_cuda\Scripts\python.exe'
$pludPython = Join-Path $repo '.venv_benchmarks\plud_sonar_oltr_2024_cuda\Scripts\python.exe'
$freeze = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\source_freeze\g1_20260715_v2\source_freeze_manifest.json'
$candidate = Join-Path $repo 'hls_lut_builder\board_harness\results\pareto_route_gate_phase0_v4_sonar_stage3_k3_lowdsp\candidates\003_rl_arch_135.candidate.json'
$runtimeRoot = Join-Path $repo 'results\protocol\g1_clean_20260711'
$pretrainedDir = Join-Path $runtimeRoot 'g1_mobilenet_v2_grayscale_imagenet'
$nasDir = Join-Path $runtimeRoot 'g1_rl_arch_135_legacy_selected'
$scratchV2Dir = Join-Path $runtimeRoot 'g1_mobilenet_v2_scratch_v2'
$sureDir = Join-Path $repo 'results\benchmarks\ccf_ab_nksid_av7k325_v1\formal\closed\sure_same_backbone'
$robustRoot = Join-Path $repo 'results\benchmarks\ccf_ab_nksid_av7k325_v1\formal\robustness'
$openRoot = Join-Path $repo 'results\benchmarks\ccf_ab_nksid_av7k325_v1\formal\open'
$ceMspDir = Join-Path $openRoot 'ce_msp'
$dmclDir = Join-Path $openRoot 'dmcl_author_loss'
$pludDir = Join-Path $openRoot 'plud_author_loss'
$auditScript = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\runtime\audit_g1_run.py'
$openAuditScript = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\runtime\audit_open_run.txt'
$robustEvaluator = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\runtime\evaluate_sonar_corruptions.txt'
$robustArtifactBuilder = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\runtime\build_sonar_robustness_artifacts.txt'
$robustArtifactManifest = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\manifests\sonar_robustness_artifact_build.json'
$closedArtifactBuilder = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\runtime\build_closed_classification_artifacts.txt'
$closedArtifactManifest = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\manifests\closed_classification_artifact_build.json'
$openArtifactBuilder = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\runtime\build_open_set_artifacts.txt'
$openArtifactManifest = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\manifests\open_set_artifact_build.json'
$pretrainedAudit = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\manifests\g1_pretrained_independent_audit_20260716.json'
$nasAudit = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\manifests\g1_nas_independent_audit_20260716.json'
$sureAudit = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\manifests\sure_closed_independent_audit_20260716.json'
$scratchV2Audit = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\manifests\g1_scratch_v2_independent_audit_20260716.json'
$ceMspAudit = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\manifests\open_ce_msp_independent_audit_20260716.json'
$dmclAudit = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\manifests\open_dmcl_independent_audit_20260716.json'
$pludAudit = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\manifests\open_plud_independent_audit_20260716.json'
$nasDecisionPause = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\manifests\g1_nas_underperformance_pause_20260717.json.txt'
$nasDecisionApproval = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\manifests\g1_nas_underperformance_user_decision_20260717.json.txt'
$statusLog = Join-Path $runtimeRoot 'g1_persistent_runtime_20260716.txt'
$pretrainedStdout = Join-Path $repo 'logs\g1_clean_20260711\pretrained_persistent_20260716_stdout.log'
$pretrainedStderr = Join-Path $repo 'logs\g1_clean_20260711\pretrained_persistent_20260716_stderr.log'
$nasStdout = Join-Path $repo 'logs\g1_clean_20260711\nas_persistent_20260716_stdout.log'
$nasStderr = Join-Path $repo 'logs\g1_clean_20260711\nas_persistent_20260716_stderr.log'
$sureStdout = Join-Path $repo 'logs\g1_clean_20260711\sure_closed_persistent_20260716_stdout.log'
$sureStderr = Join-Path $repo 'logs\g1_clean_20260711\sure_closed_persistent_20260716_stderr.log'
$scratchV2Stdout = Join-Path $repo 'logs\g1_clean_20260711\scratch_v2_persistent_20260716_stdout.log'
$scratchV2Stderr = Join-Path $repo 'logs\g1_clean_20260711\scratch_v2_persistent_20260716_stderr.log'
$robustStdout = Join-Path $repo 'logs\g1_clean_20260711\sonar_corruption_persistent_20260716_stdout.log'
$robustStderr = Join-Path $repo 'logs\g1_clean_20260711\sonar_corruption_persistent_20260716_stderr.log'
$ceMspStdout = Join-Path $repo 'logs\g1_clean_20260711\open_ce_msp_persistent_20260716_stdout.log'
$ceMspStderr = Join-Path $repo 'logs\g1_clean_20260711\open_ce_msp_persistent_20260716_stderr.log'
$dmclStdout = Join-Path $repo 'logs\g1_clean_20260711\open_dmcl_persistent_20260716_stdout.log'
$dmclStderr = Join-Path $repo 'logs\g1_clean_20260711\open_dmcl_persistent_20260716_stderr.log'
$pludStdout = Join-Path $repo 'logs\g1_clean_20260711\open_plud_persistent_20260716_stdout.log'
$pludStderr = Join-Path $repo 'logs\g1_clean_20260711\open_plud_persistent_20260716_stderr.log'

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class HwnasExecutionState {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
'@

function Write-RuntimeStatus {
    param([string]$Status, [string]$Detail = '')
    $line = "{0}`t{1}`t{2}" -f (Get-Date).ToString('o'), $Status, $Detail
    Add-Content -LiteralPath $statusLog -Value $line -Encoding UTF8
}

function Require-UserDecisionApproval {
    param([string]$DecisionField)
    if (-not (Test-Path -LiteralPath $nasDecisionPause -PathType Leaf)) {
        return
    }
    if (-not (Test-Path -LiteralPath $nasDecisionApproval -PathType Leaf)) {
        Write-RuntimeStatus -Status 'PAUSED_PENDING_USER_DECISION' -Detail $DecisionField
        throw "User decision is required before $DecisionField; approval artifact is missing"
    }
    $pauseSha = (Get-FileHash -LiteralPath $nasDecisionPause -Algorithm SHA256).Hash.ToLower()
    $decision = Get-Content -LiteralPath $nasDecisionApproval -Raw | ConvertFrom-Json
    if (
        $decision.schema_version -ne 1 -or
        $decision.campaign_id -ne 'ccf_ab_nksid_av7k325_v1' -or
        $decision.user_decision_recorded -ne $true -or
        [string]$decision.pause_manifest_sha256 -ne $pauseSha -or
        $decision.$DecisionField -ne $true
    ) {
        Write-RuntimeStatus -Status 'PAUSED_USER_DECISION_NOT_APPROVED' -Detail $DecisionField
        throw "User decision artifact does not approve $DecisionField"
    }
    Write-RuntimeStatus -Status 'USER_DECISION_APPROVED' -Detail $DecisionField
}

function Get-Claimability {
    param([string]$RunDir)
    $summaryPath = Join-Path $RunDir 'protocol_summary.json'
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        return $null
    }
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    return $summary.claimability
}

function Test-CompleteClaimableRun {
    param([string]$RunDir)
    $claimability = Get-Claimability -RunDir $RunDir
    if ($null -eq $claimability) {
        return $false
    }
    return (
        $claimability.claimable -eq $true -and
        $claimability.protocol_complete -eq $true -and
        $claimability.source_freeze_verified -eq $true -and
        $claimability.observed_run_count -eq 15 -and
        @($claimability.missing_pairs).Count -eq 0 -and
        @($claimability.unexpected_pairs).Count -eq 0
    )
}

function Test-PassingAudit {
    param(
        [string]$AuditPath,
        [string]$RunDir,
        [string]$ExpectedMethod,
        [string]$AuditorPath
    )
    if (-not (Test-Path -LiteralPath $AuditPath -PathType Leaf)) {
        return $false
    }
    try {
        $audit = Get-Content -LiteralPath $AuditPath -Raw | ConvertFrom-Json
        $manifestPath = Join-Path $RunDir 'run_manifest.json'
        if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
            return $false
        }
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        if (
            $audit.status -ne 'PASS' -or
            $audit.record_count -ne 15 -or
            @($audit.errors).Count -ne 0 -or
            ([System.IO.Path]::GetFullPath([string]$audit.run_dir) -ne [System.IO.Path]::GetFullPath($RunDir)) -or
            $audit.expected_method -ne $ExpectedMethod -or
            $audit.run_fingerprint -ne $manifest.run_fingerprint
        ) {
            return $false
        }
        if (-not (Test-Path -LiteralPath $AuditorPath -PathType Leaf)) {
            return $false
        }
        $auditorSha = (Get-FileHash -LiteralPath $AuditorPath -Algorithm SHA256).Hash.ToLower()
        if ($null -ne $audit.auditor_sha256) {
            if ([string]$audit.auditor_sha256 -ne $auditorSha) {
                return $false
            }
        }
        elseif ((Get-Item -LiteralPath $AuditPath).LastWriteTimeUtc -lt (Get-Item -LiteralPath $AuditorPath).LastWriteTimeUtc) {
            return $false
        }

        $auditRecords = @($audit.records)
        if ($auditRecords.Count -ne 15) {
            return $false
        }
        $seen = @{}
        foreach ($auditRecord in $auditRecords) {
            $fold = [int]$auditRecord.fold
            $seed = [int]$auditRecord.seed
            $key = '{0}:{1}' -f $fold, $seed
            if ($seen.ContainsKey($key)) {
                return $false
            }
            $seen[$key] = $true
            $recordPath = Join-Path $RunDir ("run_fold{0}_seed{1}.json" -f $fold, $seed)
            if (-not (Test-Path -LiteralPath $recordPath -PathType Leaf)) {
                return $false
            }
            $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
            if (
                [string]$record.outer_predictions.sha256 -ne [string]$auditRecord.prediction_sha256 -or
                [string]$record.checkpoint.sha256 -ne [string]$auditRecord.checkpoint_sha256
            ) {
                return $false
            }
            $predictionPath = [string]$record.outer_predictions.path
            $checkpointPath = [string]$record.checkpoint.path
            if (
                -not (Test-Path -LiteralPath $predictionPath -PathType Leaf) -or
                -not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)
            ) {
                return $false
            }
            if (
                (Get-FileHash -LiteralPath $predictionPath -Algorithm SHA256).Hash.ToLower() -ne [string]$auditRecord.prediction_sha256 -or
                (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLower() -ne [string]$auditRecord.checkpoint_sha256
            ) {
                return $false
            }
            if ($null -ne $auditRecord.record_sha256) {
                if ((Get-FileHash -LiteralPath $recordPath -Algorithm SHA256).Hash.ToLower() -ne [string]$auditRecord.record_sha256) {
                    return $false
                }
            }
            elseif ((Get-Item -LiteralPath $recordPath).LastWriteTimeUtc -gt (Get-Item -LiteralPath $AuditPath).LastWriteTimeUtc) {
                return $false
            }
        }
        return $seen.Count -eq 15
    }
    catch {
        return $false
    }
}

function Test-CompleteRobustness {
    param(
        [string]$OutputDir,
        [string]$RunDir,
        [string]$AuditPath,
        [string]$MethodId
    )
    $manifestPath = Join-Path $OutputDir 'manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        return $false
    }
    try {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        $auditSha = (Get-FileHash -LiteralPath $AuditPath -Algorithm SHA256).Hash.ToLower()
        $evaluatorSha = (Get-FileHash -LiteralPath $robustEvaluator -Algorithm SHA256).Hash.ToLower()
        return (
            $manifest.status -eq 'COMPLETE_INPUT_AUDIT_PASS_PENDING_G1_LEDGER' -and
            $manifest.method_id -eq $MethodId -and
            ([System.IO.Path]::GetFullPath([string]$manifest.run_dir) -eq [System.IO.Path]::GetFullPath($RunDir)) -and
            $manifest.audit_sha256 -eq $auditSha -and
            $manifest.evaluator_sha256 -eq $evaluatorSha -and
            $manifest.existing_f_robust_definition_changed -eq $false
        )
    }
    catch {
        return $false
    }
}

function Test-CompleteRobustnessArtifacts {
    if (-not (Test-Path -LiteralPath $robustArtifactManifest -PathType Leaf)) {
        return $false
    }
    try {
        $manifest = Get-Content -LiteralPath $robustArtifactManifest -Raw | ConvertFrom-Json
        $generatorSha = (Get-FileHash -LiteralPath $robustArtifactBuilder -Algorithm SHA256).Hash.ToLower()
        if (
            $manifest.status -ne 'COMPLETE_INPUT_AUDITS_PASS_PENDING_G1_LEDGER' -or
            $manifest.generator_sha256 -ne $generatorSha -or
            @($manifest.input_manifests).Count -ne 4
        ) {
            return $false
        }
        foreach ($input in @($manifest.input_manifests)) {
            if (-not (Test-Path -LiteralPath $input.path -PathType Leaf)) {
                return $false
            }
            $observed = (Get-FileHash -LiteralPath $input.path -Algorithm SHA256).Hash.ToLower()
            if ($observed -ne $input.sha256) {
                return $false
            }
        }
        foreach ($required in @(
            'tables\t4.csv', 'tables\t4.md', 'tables\t4.tex',
            'tables\t9.csv', 'tables\t9.md', 'tables\t9.tex',
            'figures\f7.svg', 'figures\f7.pdf', 'figures\f7.png', 'figures\f7_source.csv', 'figures\f7_meta.json',
            'figures\f8.svg', 'figures\f8.pdf', 'figures\f8.png', 'figures\f8_source.csv', 'figures\f8_meta.json'
        )) {
            if (-not (Test-Path -LiteralPath (Join-Path (Split-Path $robustArtifactManifest -Parent | Split-Path -Parent) $required) -PathType Leaf)) {
                return $false
            }
        }
        return $true
    }
    catch {
        return $false
    }
}

function Test-CompleteClosedArtifacts {
    if (-not (Test-Path -LiteralPath $closedArtifactManifest -PathType Leaf)) {
        return $false
    }
    try {
        $manifest = Get-Content -LiteralPath $closedArtifactManifest -Raw | ConvertFrom-Json
        $generatorSha = (Get-FileHash -LiteralPath $closedArtifactBuilder -Algorithm SHA256).Hash.ToLower()
        if (
            $manifest.status -ne 'COMPLETE_INPUT_AUDITS_PASS_PENDING_G1_LEDGER' -or
            $manifest.generator_sha256 -ne $generatorSha -or
            $manifest.f5_status -ne 'WITHHELD_PENDING_OPEN_SET_CONFUSION_EVIDENCE'
        ) {
            return $false
        }
        foreach ($property in $manifest.inputs.PSObject.Properties) {
            $input = $property.Value
            if (-not (Test-Path -LiteralPath $input.audit_path -PathType Leaf)) {
                return $false
            }
            $observed = (Get-FileHash -LiteralPath $input.audit_path -Algorithm SHA256).Hash.ToLower()
            if ($observed -ne $input.audit_sha256) {
                return $false
            }
        }
        $artifactRoot = Split-Path (Split-Path $closedArtifactManifest -Parent) -Parent
        foreach ($required in @(
            'tables\t2.csv', 'tables\t2.md', 'tables\t2.tex',
            'figures\f6.svg', 'figures\f6.pdf', 'figures\f6.png', 'figures\f6_source.csv', 'figures\f6_meta.json',
            'source_data\t2_fold_seed_units.csv', 'source_data\t9_closed_statistics.csv'
        )) {
            if (-not (Test-Path -LiteralPath (Join-Path $artifactRoot $required) -PathType Leaf)) {
                return $false
            }
        }
        return $true
    }
    catch {
        return $false
    }
}

function Test-CompleteOpenArtifacts {
    if (-not (Test-Path -LiteralPath $openArtifactManifest -PathType Leaf)) {
        return $false
    }
    try {
        $manifest = Get-Content -LiteralPath $openArtifactManifest -Raw | ConvertFrom-Json
        $generatorSha = (Get-FileHash -LiteralPath $openArtifactBuilder -Algorithm SHA256).Hash.ToLower()
        if (
            $manifest.status -ne 'COMPLETE_INPUT_AUDITS_PASS_PENDING_G1_LEDGER' -or
            $manifest.generator_sha256 -ne $generatorSha
        ) {
            return $false
        }
        if (-not (Test-Path -LiteralPath $manifest.closed_input.path -PathType Leaf)) {
            return $false
        }
        if ((Get-FileHash -LiteralPath $manifest.closed_input.path -Algorithm SHA256).Hash.ToLower() -ne $manifest.closed_input.sha256) {
            return $false
        }
        foreach ($property in $manifest.open_inputs.PSObject.Properties) {
            $input = $property.Value
            if (-not (Test-Path -LiteralPath $input.audit_path -PathType Leaf)) {
                return $false
            }
            if ((Get-FileHash -LiteralPath $input.audit_path -Algorithm SHA256).Hash.ToLower() -ne $input.audit_sha256) {
                return $false
            }
        }
        $artifactRoot = Split-Path (Split-Path $openArtifactManifest -Parent) -Parent
        foreach ($required in @(
            'tables\t3.csv', 'tables\t3.md', 'tables\t3.tex',
            'tables\t9.csv', 'tables\t9.md', 'tables\t9.tex',
            'figures\f5.svg', 'figures\f5.pdf', 'figures\f5.png', 'figures\f5_source.csv', 'figures\f5_meta.json',
            'source_data\t3_fold_seed_units.csv'
        )) {
            if (-not (Test-Path -LiteralPath (Join-Path $artifactRoot $required) -PathType Leaf)) {
                return $false
            }
        }
        return $true
    }
    catch {
        return $false
    }
}

function Invoke-SourceFreezeVerification {
    Write-RuntimeStatus -Status 'VERIFY_SOURCE_FREEZE'
    $verificationOutput = & $python (Join-Path $repo 'scripts\freeze_experiment_source.py') verify --manifest $freeze 2>&1
    $verificationReturnCode = $LASTEXITCODE
    foreach ($line in $verificationOutput) {
        Add-Content -LiteralPath $statusLog -Value ([string]$line) -Encoding UTF8
    }
    if ($verificationReturnCode -ne 0) {
        throw "source freeze verification failed with return code $verificationReturnCode"
    }
}

function Invoke-G1Audit {
    param(
        [string]$RunDir,
        [string]$Output,
        [string]$ExpectPretrained,
        [string]$ExpectedMethod
    )
    $auditOutputLines = & $python $auditScript --run-dir $RunDir --output $Output --expect-pretrained $ExpectPretrained --expected-method $ExpectedMethod 2>&1
    $auditReturnCode = $LASTEXITCODE
    foreach ($line in $auditOutputLines) {
        Add-Content -LiteralPath $statusLog -Value ([string]$line) -Encoding UTF8
    }
    if ($auditReturnCode -ne 0) {
        throw "independent audit failed for $RunDir with return code $auditReturnCode"
    }
}

function Invoke-OpenAudit {
    param(
        [string]$RunDir,
        [string]$Output,
        [string]$ExpectedMethod,
        [bool]$RequireEnvironmentCard
    )
    $arguments = @(
        $openAuditScript,
        '--run-dir', $RunDir,
        '--output', $Output,
        '--expected-method', $ExpectedMethod
    )
    if ($RequireEnvironmentCard) {
        $arguments += '--require-environment-card'
    }
    $auditOutputLines = & $python @arguments 2>&1
    $auditReturnCode = $LASTEXITCODE
    foreach ($line in $auditOutputLines) {
        Add-Content -LiteralPath $statusLog -Value ([string]$line) -Encoding UTF8
    }
    if ($auditReturnCode -ne 0) {
        throw "independent open-set audit failed for $RunDir with return code $auditReturnCode"
    }
}

function Invoke-PretrainedRun {
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
        '--arch', 'mobilenet_v2',
        '--pretrained',
        '--selection-provenance', 'baseline_predeclared',
        '--method-id', 'imagenet_pretrained_mobilenet_v2',
        '--run-name', 'g1_mobilenet_v2_grayscale_imagenet'
    )
    Write-RuntimeStatus -Status 'PRETRAINED_PROCESS_START'
    & $python @arguments 1>> $pretrainedStdout 2>> $pretrainedStderr
    if ($LASTEXITCODE -ne 0) {
        throw "pretrained process failed with return code $LASTEXITCODE"
    }
    Write-RuntimeStatus -Status 'PRETRAINED_PROCESS_EXIT_0'
}

function Invoke-NasRun {
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
    Write-RuntimeStatus -Status 'NAS_PROCESS_START'
    & $python @arguments 1>> $nasStdout 2>> $nasStderr
    if ($LASTEXITCODE -ne 0) {
        throw "NAS process failed with return code $LASTEXITCODE"
    }
    Write-RuntimeStatus -Status 'NAS_PROCESS_EXIT_0'
}

function Invoke-ScratchV2Run {
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
        '--arch', 'mobilenet_v2',
        '--selection-provenance', 'baseline_predeclared',
        '--method-id', 'scratch_mobilenet_v2',
        '--run-name', 'g1_mobilenet_v2_scratch_v2'
    )
    Write-RuntimeStatus -Status 'SCRATCH_V2_PROCESS_START'
    & $python @arguments 1>> $scratchV2Stdout 2>> $scratchV2Stderr
    if ($LASTEXITCODE -ne 0) {
        throw "scratch v2 process failed with return code $LASTEXITCODE"
    }
    Write-RuntimeStatus -Status 'SCRATCH_V2_PROCESS_EXIT_0'
}

function Invoke-RobustnessEvaluation {
    param(
        [string]$Interpreter,
        [string]$RunDir,
        [string]$AuditPath,
        [string]$MethodId,
        [string]$OutputDir
    )
    $arguments = @(
        $robustEvaluator,
        '--run-dir', $RunDir,
        '--audit-json', $AuditPath,
        '--output-dir', $OutputDir,
        '--data-dir', 'data/NKSID',
        '--method-id', $MethodId,
        '--batch-size', '32',
        '--num-workers', '0',
        '--device', 'cuda'
    )
    Write-RuntimeStatus -Status ("SONAR_CORRUPTION_PROCESS_START_{0}" -f $MethodId)
    & $Interpreter @arguments 1>> $robustStdout 2>> $robustStderr
    if ($LASTEXITCODE -ne 0) {
        throw "sonar corruption evaluation $MethodId failed with return code $LASTEXITCODE"
    }
    Write-RuntimeStatus -Status ("SONAR_CORRUPTION_PROCESS_EXIT_0_{0}" -f $MethodId)
}

function Invoke-RobustnessArtifactBuild {
    if (-not (Test-Path -LiteralPath $analysisPython -PathType Leaf)) {
        throw "analysis Python is missing: $analysisPython"
    }
    if (-not (Test-Path -LiteralPath $robustArtifactBuilder -PathType Leaf)) {
        throw "sonar robustness artifact builder is missing: $robustArtifactBuilder"
    }
    Write-RuntimeStatus -Status 'T4_F7_F8_T9_ARTIFACT_BUILD_START'
    & $analysisPython $robustArtifactBuilder --campaign-id 'ccf_ab_nksid_av7k325_v1' --robust-root $robustRoot 1>> $robustStdout 2>> $robustStderr
    if ($LASTEXITCODE -ne 0) {
        throw "sonar robustness artifact build failed with return code $LASTEXITCODE"
    }
    Write-RuntimeStatus -Status 'T4_F7_F8_T9_ARTIFACT_BUILD_EXIT_0'
}

function Invoke-ClosedArtifactBuild {
    if (-not (Test-Path -LiteralPath $analysisPython -PathType Leaf)) {
        throw "analysis Python is missing: $analysisPython"
    }
    if (-not (Test-Path -LiteralPath $closedArtifactBuilder -PathType Leaf)) {
        throw "closed classification artifact builder is missing: $closedArtifactBuilder"
    }
    $arguments = @(
        $closedArtifactBuilder,
        '--campaign-id', 'ccf_ab_nksid_av7k325_v1',
        '--scratch-run', $scratchV2Dir,
        '--scratch-audit', $scratchV2Audit,
        '--pretrained-run', $pretrainedDir,
        '--pretrained-audit', $pretrainedAudit,
        '--nas-run', $nasDir,
        '--nas-audit', $nasAudit,
        '--sure-run', $sureDir,
        '--sure-audit', $sureAudit
    )
    Write-RuntimeStatus -Status 'T2_F6_ARTIFACT_BUILD_START'
    & $analysisPython @arguments 1>> $robustStdout 2>> $robustStderr
    if ($LASTEXITCODE -ne 0) {
        throw "closed classification artifact build failed with return code $LASTEXITCODE"
    }
    Write-RuntimeStatus -Status 'T2_F6_ARTIFACT_BUILD_EXIT_0'
}

function Invoke-OpenArtifactBuild {
    if (-not (Test-Path -LiteralPath $analysisPython -PathType Leaf)) {
        throw "analysis Python is missing: $analysisPython"
    }
    if (-not (Test-Path -LiteralPath $openArtifactBuilder -PathType Leaf)) {
        throw "open-set artifact builder is missing: $openArtifactBuilder"
    }
    $arguments = @(
        $openArtifactBuilder,
        '--campaign-id', 'ccf_ab_nksid_av7k325_v1',
        '--closed-manifest', $closedArtifactManifest,
        '--ce-run', $ceMspDir,
        '--ce-audit', $ceMspAudit,
        '--dmcl-run', $dmclDir,
        '--dmcl-audit', $dmclAudit,
        '--plud-run', $pludDir,
        '--plud-audit', $pludAudit
    )
    Write-RuntimeStatus -Status 'T3_F5_T9_ARTIFACT_BUILD_START'
    & $analysisPython @arguments 1>> $robustStdout 2>> $robustStderr
    if ($LASTEXITCODE -ne 0) {
        throw "open-set artifact build failed with return code $LASTEXITCODE"
    }
    Write-RuntimeStatus -Status 'T3_F5_T9_ARTIFACT_BUILD_EXIT_0'
}

function Invoke-FinalBenchmarkAudits {
    Write-RuntimeStatus -Status 'BENCHMARK_ARTIFACT_STATUS_REFRESH_START'
    & $analysisPython (Join-Path $repo 'scripts\build_benchmark_artifacts.py') --campaign-id 'ccf_ab_nksid_av7k325_v1' 1>> $robustStdout 2>> $robustStderr
    if ($LASTEXITCODE -ne 0) {
        throw "benchmark artifact status refresh failed with return code $LASTEXITCODE"
    }
    Write-RuntimeStatus -Status 'BENCHMARK_READINESS_REAUDIT_START'
    & $analysisPython (Join-Path $repo 'scripts\audit_benchmark_readiness.py') --campaign-id 'ccf_ab_nksid_av7k325_v1' 1>> $robustStdout 2>> $robustStderr
    if ($LASTEXITCODE -ne 0) {
        throw "benchmark readiness audit failed with return code $LASTEXITCODE"
    }
    Write-RuntimeStatus -Status 'MEASUREMENT_FIRST_LEDGER_REAUDIT_START'
    & $python (Join-Path $repo 'scripts\audit_measurement_first_gates.py') 1>> $robustStdout 2>> $robustStderr
    if ($LASTEXITCODE -ne 0) {
        throw "measurement-first ledger audit failed with return code $LASTEXITCODE"
    }
    Write-RuntimeStatus -Status 'FINAL_BENCHMARK_AUDITS_EXIT_0'
}

function Invoke-SureRun {
    $arguments = @(
        'run_eval_protocol.py',
        '--data-dir', 'data/NKSID',
        '--output-dir', 'results/benchmarks/ccf_ab_nksid_av7k325_v1/formal/closed',
        '--task', 'closed_set',
        '--adapter-id', 'sure_author_recipe',
        '--campaign-id', 'ccf_ab_nksid_av7k325_v1',
        '--paper-id', 'sure_2024',
        '--method-id', 'sure_same_backbone',
        '--folds', '0,1,2,3,4',
        '--seeds', '42,43,44',
        '--epochs', '150',
        '--batch-size', '8',
        '--gradient-accumulation-steps', '1',
        '--num-workers', '0',
        '--arch', 'mobilenet_v2',
        '--save-checkpoints',
        '--resume',
        '--source-freeze-manifest', $freeze,
        '--selection-provenance', 'baseline_predeclared',
        '--run-name', 'sure_same_backbone'
    )
    Write-RuntimeStatus -Status 'SURE_PROCESS_START'
    & $surePython @arguments 1>> $sureStdout 2>> $sureStderr
    if ($LASTEXITCODE -ne 0) {
        throw "SURE process failed with return code $LASTEXITCODE"
    }
    Write-RuntimeStatus -Status 'SURE_PROCESS_EXIT_0'
}

function Invoke-OpenRun {
    param(
        [string]$Interpreter,
        [string]$AdapterId,
        [string]$PaperId,
        [string]$MethodId,
        [string]$RunName,
        [string]$Stdout,
        [string]$Stderr
    )
    $arguments = @(
        'run_eval_protocol.py',
        '--data-dir', 'data/NKSID',
        '--output-dir', 'results/benchmarks/ccf_ab_nksid_av7k325_v1/formal/open',
        '--task', 'open_long_tail',
        '--adapter-id', $AdapterId,
        '--campaign-id', 'ccf_ab_nksid_av7k325_v1',
        '--paper-id', $PaperId,
        '--method-id', $MethodId,
        '--folds', '0,1,2,3,4',
        '--seeds', '42,43,44',
        '--epochs', '150',
        '--batch-size', '8',
        '--gradient-accumulation-steps', '1',
        '--num-workers', '0',
        '--arch', 'mobilenet_v2',
        '--save-checkpoints',
        '--resume',
        '--source-freeze-manifest', $freeze,
        '--selection-provenance', 'baseline_predeclared',
        '--run-name', $RunName
    )
    Write-RuntimeStatus -Status ("OPEN_PROCESS_START_{0}" -f $MethodId)
    & $Interpreter @arguments 1>> $Stdout 2>> $Stderr
    if ($LASTEXITCODE -ne 0) {
        throw "open-set process $MethodId failed with return code $LASTEXITCODE"
    }
    Write-RuntimeStatus -Status ("OPEN_PROCESS_EXIT_0_{0}" -f $MethodId)
}

$executionStateSet = [HwnasExecutionState]::SetThreadExecutionState([uint32]2147483649)
if ($executionStateSet -eq 0) {
    throw 'SetThreadExecutionState failed'
}

try {
    Set-Location -LiteralPath $repo
    Write-RuntimeStatus -Status 'ORCHESTRATOR_START' -Detail "pid=$PID"
    Invoke-SourceFreezeVerification

    if (-not (Test-CompleteClaimableRun -RunDir $pretrainedDir)) {
        Invoke-PretrainedRun
    } else {
        Write-RuntimeStatus -Status 'PRETRAINED_ALREADY_COMPLETE'
    }
    if (-not (Test-CompleteClaimableRun -RunDir $pretrainedDir)) {
        throw 'pretrained run exited without a complete claimable 15-unit summary'
    }
    if (-not (Test-PassingAudit -AuditPath $pretrainedAudit -RunDir $pretrainedDir -ExpectedMethod 'imagenet_pretrained_mobilenet_v2' -AuditorPath $auditScript)) {
        Write-RuntimeStatus -Status 'PRETRAINED_INDEPENDENT_AUDIT_START'
        Invoke-G1Audit -RunDir $pretrainedDir -Output $pretrainedAudit -ExpectPretrained 'true' -ExpectedMethod 'imagenet_pretrained_mobilenet_v2'
    }
    if (-not (Test-PassingAudit -AuditPath $pretrainedAudit -RunDir $pretrainedDir -ExpectedMethod 'imagenet_pretrained_mobilenet_v2' -AuditorPath $auditScript)) {
        throw 'pretrained independent audit did not produce PASS'
    }
    Write-RuntimeStatus -Status 'PRETRAINED_ACCEPTED_15_OF_15'

    Invoke-SourceFreezeVerification
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "NAS candidate is missing: $candidate"
    }
    if (-not (Test-CompleteClaimableRun -RunDir $nasDir)) {
        Require-UserDecisionApproval -DecisionField 'resume_nas_to_15'
        Invoke-NasRun
    } else {
        Write-RuntimeStatus -Status 'NAS_ALREADY_COMPLETE'
    }
    if (-not (Test-CompleteClaimableRun -RunDir $nasDir)) {
        throw 'NAS run exited without a complete claimable 15-unit summary'
    }
    if (-not (Test-PassingAudit -AuditPath $nasAudit -RunDir $nasDir -ExpectedMethod 'frozen_nas_champion' -AuditorPath $auditScript)) {
        Write-RuntimeStatus -Status 'NAS_INDEPENDENT_AUDIT_START'
        Invoke-G1Audit -RunDir $nasDir -Output $nasAudit -ExpectPretrained 'false' -ExpectedMethod 'frozen_nas_champion'
    }
    if (-not (Test-PassingAudit -AuditPath $nasAudit -RunDir $nasDir -ExpectedMethod 'frozen_nas_champion' -AuditorPath $auditScript)) {
        throw 'NAS independent audit did not produce PASS'
    }
    Write-RuntimeStatus -Status 'G1_DYNAMIC_RUNS_ACCEPTED_30_OF_30'

    Require-UserDecisionApproval -DecisionField 'continue_downstream_closed_set_chain'
    Invoke-SourceFreezeVerification
    if (-not (Test-Path -LiteralPath $surePython -PathType Leaf)) {
        throw "SURE dedicated interpreter is missing: $surePython"
    }
    if (-not (Test-CompleteClaimableRun -RunDir $sureDir)) {
        Invoke-SureRun
    } else {
        Write-RuntimeStatus -Status 'SURE_ALREADY_COMPLETE'
    }
    if (-not (Test-CompleteClaimableRun -RunDir $sureDir)) {
        throw 'SURE run exited without a complete claimable 15-unit summary'
    }
    if (-not (Test-PassingAudit -AuditPath $sureAudit -RunDir $sureDir -ExpectedMethod 'sure_same_backbone' -AuditorPath $auditScript)) {
        Write-RuntimeStatus -Status 'SURE_INDEPENDENT_AUDIT_START'
        Invoke-G1Audit -RunDir $sureDir -Output $sureAudit -ExpectPretrained 'false' -ExpectedMethod 'sure_same_backbone'
    }
    if (-not (Test-PassingAudit -AuditPath $sureAudit -RunDir $sureDir -ExpectedMethod 'sure_same_backbone' -AuditorPath $auditScript)) {
        throw 'SURE independent audit did not produce PASS'
    }
    Write-RuntimeStatus -Status 'R3_CLOSED_METHODS_DYNAMIC_RUNS_ACCEPTED_45_OF_45'

    # The original scratch run remains scientifically intact, but a failed
    # resume overwrote its generated code_patch.diff. Re-run under a new name
    # instead of weakening the independent provenance audit.
    Invoke-SourceFreezeVerification
    if (-not (Test-CompleteClaimableRun -RunDir $scratchV2Dir)) {
        Invoke-ScratchV2Run
    } else {
        Write-RuntimeStatus -Status 'SCRATCH_V2_ALREADY_COMPLETE'
    }
    if (-not (Test-CompleteClaimableRun -RunDir $scratchV2Dir)) {
        throw 'scratch v2 run exited without a complete claimable 15-unit summary'
    }
    if (-not (Test-PassingAudit -AuditPath $scratchV2Audit -RunDir $scratchV2Dir -ExpectedMethod 'scratch_mobilenet_v2' -AuditorPath $auditScript)) {
        Write-RuntimeStatus -Status 'SCRATCH_V2_INDEPENDENT_AUDIT_START'
        Invoke-G1Audit -RunDir $scratchV2Dir -Output $scratchV2Audit -ExpectPretrained 'false' -ExpectedMethod 'scratch_mobilenet_v2'
    }
    if (-not (Test-PassingAudit -AuditPath $scratchV2Audit -RunDir $scratchV2Dir -ExpectedMethod 'scratch_mobilenet_v2' -AuditorPath $auditScript)) {
        throw 'scratch v2 independent audit did not produce PASS'
    }
    Write-RuntimeStatus -Status 'R3_CLOSED_METHODS_ACCEPTED_60_OF_60'

    if (-not (Test-CompleteClosedArtifacts)) {
        Invoke-ClosedArtifactBuild
    } else {
        Write-RuntimeStatus -Status 'T2_F6_ARTIFACTS_ALREADY_COMPLETE'
    }
    if (-not (Test-CompleteClosedArtifacts)) {
        throw 'T2/F6 artifact build did not produce a matching complete manifest'
    }
    Write-RuntimeStatus -Status 'T2_F6_ARTIFACTS_ACCEPTED_F5_WITHHELD'

    if (-not (Test-Path -LiteralPath $robustEvaluator -PathType Leaf)) {
        throw "sonar corruption evaluator is missing: $robustEvaluator"
    }
    $robustRuns = @(
        @{
            Interpreter = $python
            RunDir = $scratchV2Dir
            AuditPath = $scratchV2Audit
            MethodId = 'scratch_mobilenet_v2'
            OutputDir = (Join-Path $robustRoot 'scratch_mobilenet_v2')
        },
        @{
            Interpreter = $python
            RunDir = $pretrainedDir
            AuditPath = $pretrainedAudit
            MethodId = 'imagenet_pretrained_mobilenet_v2'
            OutputDir = (Join-Path $robustRoot 'imagenet_pretrained_mobilenet_v2')
        },
        @{
            Interpreter = $python
            RunDir = $nasDir
            AuditPath = $nasAudit
            MethodId = 'frozen_nas_champion'
            OutputDir = (Join-Path $robustRoot 'frozen_nas_champion')
        },
        @{
            Interpreter = $surePython
            RunDir = $sureDir
            AuditPath = $sureAudit
            MethodId = 'sure_same_backbone'
            OutputDir = (Join-Path $robustRoot 'sure_same_backbone')
        }
    )
    foreach ($robustRun in $robustRuns) {
        if (-not (Test-CompleteRobustness -OutputDir $robustRun.OutputDir -RunDir $robustRun.RunDir -AuditPath $robustRun.AuditPath -MethodId $robustRun.MethodId)) {
            Invoke-RobustnessEvaluation -Interpreter $robustRun.Interpreter -RunDir $robustRun.RunDir -AuditPath $robustRun.AuditPath -MethodId $robustRun.MethodId -OutputDir $robustRun.OutputDir
        } else {
            Write-RuntimeStatus -Status ("SONAR_CORRUPTION_ALREADY_COMPLETE_{0}" -f $robustRun.MethodId)
        }
        if (-not (Test-CompleteRobustness -OutputDir $robustRun.OutputDir -RunDir $robustRun.RunDir -AuditPath $robustRun.AuditPath -MethodId $robustRun.MethodId)) {
            throw "sonar corruption evaluation did not produce a matching complete manifest: $($robustRun.MethodId)"
        }
    }
    Write-RuntimeStatus -Status 'T4_F7_F8_INPUTS_ACCEPTED_4_METHODS'

    if (-not (Test-CompleteRobustnessArtifacts)) {
        Invoke-RobustnessArtifactBuild
    } else {
        Write-RuntimeStatus -Status 'T4_F7_F8_T9_ARTIFACTS_ALREADY_COMPLETE'
    }
    if (-not (Test-CompleteRobustnessArtifacts)) {
        throw 'T4/F7/F8/T9 artifact build did not produce a matching complete manifest'
    }
    Write-RuntimeStatus -Status 'T4_F7_F8_T9_ARTIFACTS_ACCEPTED'

    Invoke-SourceFreezeVerification
    if (-not (Test-CompleteClaimableRun -RunDir $ceMspDir)) {
        Invoke-OpenRun -Interpreter $python -AdapterId 'builtin' -PaperId 'project_internal' -MethodId 'ce_msp' -RunName 'ce_msp' -Stdout $ceMspStdout -Stderr $ceMspStderr
    }
    if (-not (Test-CompleteClaimableRun -RunDir $ceMspDir)) {
        throw 'CE+MSP open-set run is not complete and claimable'
    }
    if (-not (Test-PassingAudit -AuditPath $ceMspAudit -RunDir $ceMspDir -ExpectedMethod 'ce_msp' -AuditorPath $openAuditScript)) {
        Write-RuntimeStatus -Status 'OPEN_CE_MSP_INDEPENDENT_AUDIT_START'
        Invoke-OpenAudit -RunDir $ceMspDir -Output $ceMspAudit -ExpectedMethod 'ce_msp' -RequireEnvironmentCard $false
    }
    if (-not (Test-PassingAudit -AuditPath $ceMspAudit -RunDir $ceMspDir -ExpectedMethod 'ce_msp' -AuditorPath $openAuditScript)) {
        throw 'CE+MSP independent audit did not produce PASS'
    }
    Write-RuntimeStatus -Status 'OPEN_CE_MSP_ACCEPTED_15_OF_15'

    Invoke-SourceFreezeVerification
    if (-not (Test-Path -LiteralPath $dmclPython -PathType Leaf)) {
        throw "DMCL dedicated interpreter is missing: $dmclPython"
    }
    if (-not (Test-CompleteClaimableRun -RunDir $dmclDir)) {
        Invoke-OpenRun -Interpreter $dmclPython -AdapterId 'dmcl_author_loss' -PaperId 'dmcl_sonar_oltr_2025' -MethodId 'dmcl_author_loss' -RunName 'dmcl_author_loss' -Stdout $dmclStdout -Stderr $dmclStderr
    }
    if (-not (Test-CompleteClaimableRun -RunDir $dmclDir)) {
        throw 'DMCL open-set run is not complete and claimable'
    }
    if (-not (Test-PassingAudit -AuditPath $dmclAudit -RunDir $dmclDir -ExpectedMethod 'dmcl_author_loss' -AuditorPath $openAuditScript)) {
        Write-RuntimeStatus -Status 'OPEN_DMCL_INDEPENDENT_AUDIT_START'
        Invoke-OpenAudit -RunDir $dmclDir -Output $dmclAudit -ExpectedMethod 'dmcl_author_loss' -RequireEnvironmentCard $true
    }
    if (-not (Test-PassingAudit -AuditPath $dmclAudit -RunDir $dmclDir -ExpectedMethod 'dmcl_author_loss' -AuditorPath $openAuditScript)) {
        throw 'DMCL independent audit did not produce PASS'
    }
    Write-RuntimeStatus -Status 'OPEN_DMCL_ACCEPTED_15_OF_15'

    Invoke-SourceFreezeVerification
    if (-not (Test-Path -LiteralPath $pludPython -PathType Leaf)) {
        throw "PLUD dedicated interpreter is missing: $pludPython"
    }
    if (-not (Test-CompleteClaimableRun -RunDir $pludDir)) {
        Invoke-OpenRun -Interpreter $pludPython -AdapterId 'plud_author_loss' -PaperId 'plud_sonar_oltr_2024' -MethodId 'plud_author_loss' -RunName 'plud_author_loss' -Stdout $pludStdout -Stderr $pludStderr
    }
    if (-not (Test-CompleteClaimableRun -RunDir $pludDir)) {
        throw 'PLUD open-set run is not complete and claimable'
    }
    if (-not (Test-PassingAudit -AuditPath $pludAudit -RunDir $pludDir -ExpectedMethod 'plud_author_loss' -AuditorPath $openAuditScript)) {
        Write-RuntimeStatus -Status 'OPEN_PLUD_INDEPENDENT_AUDIT_START'
        Invoke-OpenAudit -RunDir $pludDir -Output $pludAudit -ExpectedMethod 'plud_author_loss' -RequireEnvironmentCard $true
    }
    if (-not (Test-PassingAudit -AuditPath $pludAudit -RunDir $pludDir -ExpectedMethod 'plud_author_loss' -AuditorPath $openAuditScript)) {
        throw 'PLUD independent audit did not produce PASS'
    }
    Write-RuntimeStatus -Status 'R4_OPEN_METHODS_ACCEPTED_45_OF_45'

    if (-not (Test-CompleteOpenArtifacts)) {
        Invoke-OpenArtifactBuild
    } else {
        Write-RuntimeStatus -Status 'T3_F5_T9_ARTIFACTS_ALREADY_COMPLETE'
    }
    if (-not (Test-CompleteOpenArtifacts)) {
        throw 'T3/F5/T9 artifact build did not produce a matching complete manifest'
    }
    Write-RuntimeStatus -Status 'T3_F5_T9_ARTIFACTS_ACCEPTED'
    Invoke-FinalBenchmarkAudits
    exit 0
}
catch {
    Write-RuntimeStatus -Status 'ORCHESTRATOR_STOPPED' -Detail $_.Exception.Message
    exit 2
}
finally {
    [void][HwnasExecutionState]::SetThreadExecutionState([uint32]2147483648)
}
