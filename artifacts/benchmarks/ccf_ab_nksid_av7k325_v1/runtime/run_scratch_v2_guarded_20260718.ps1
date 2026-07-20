param([switch]$PreflightOnly)

$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
$python = Join-Path $repo '.venv_cuda\Scripts\python.exe'
$analysisPython = 'D:\software\python\python.exe'
$approval = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\manifests\scratch_v2_execution_authorization_20260718.json.txt'
$approvalConsumed = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\manifests\scratch_v2_execution_authorization_20260718.consumed.json.txt'
$expectedApprovalSha256 = 'c63d31b4ac203ce07cef40c6c1d4297f30328e3d6191ae76ae171a3f3402e3ce'
$freeze = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\source_freeze\g1_20260718_v3\source_freeze_manifest.json'
$outputRoot = Join-Path $repo 'results\protocol\g1_clean_20260718'
$runDir = Join-Path $outputRoot 'g1_mobilenet_v2_scratch_v2'
$legacyRunDir = Join-Path $repo 'results\protocol\g1_clean_20260711\g1_mobilenet_v2_scratch'
$legacyManifest = Join-Path $legacyRunDir 'run_manifest.json'
$legacyPatch = Join-Path $legacyRunDir 'code_patch.diff'
$auditScript = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\runtime\audit_g1_run.py'
$auditOutput = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\manifests\g1_scratch_v2_independent_audit_20260718.json'
$ledgerOutput = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\manifests\measurement_first_reaudit_20260718'
$logRoot = Join-Path $repo 'logs\g1_clean_20260718'
$stdoutLog = Join-Path $logRoot 'scratch_v2_stdout.log'
$stderrLog = Join-Path $logRoot 'scratch_v2_stderr.log'
$statusLog = Join-Path $logRoot $(
    if ($PreflightOnly) {
        'scratch_v2_guard_preflight_{0}.tsv' -f (Get-Date).ToString('yyyyMMdd_HHmmss')
    }
    else { 'scratch_v2_guard_status.tsv' }
)
$taskName = 'Codex_HWNAS_ScratchV2_20260718'
$process = $null

function Get-LowerSha256 {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "缺少待核验文件：$Path"
    }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLower()
}

function Write-GuardStatus {
    param([string]$Status, [string]$Detail = '')
    $line = "{0}`t{1}`t{2}" -f (Get-Date).ToString('o'), $Status, $Detail
    Add-Content -LiteralPath $statusLog -Value $line -Encoding UTF8
}

function Disable-OwnTask {
    Disable-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue | Out-Null
}

function Stop-WithChineseIncident {
    param([string]$Reason)
    if ($null -ne $process) {
        try {
            $process.Refresh()
            if (-not $process.HasExited) {
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            }
        } catch {
        }
    }
    $stamp = (Get-Date).ToString('yyyyMMdd_HHmmss')
    $incident = Join-Path $repo "artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\manifests\scratch_v2_interruption_$stamp.md"
    $completed = @(Get-ChildItem -LiteralPath $runDir -Filter 'run_fold*_seed*.json' -File -ErrorAction SilentlyContinue).Count
    $content = @"
# scratch-v2 中断记录

- 时间：$((Get-Date).ToString('o'))
- 状态：INTERRUPTED_FAIL_CLOSED
- 已发现完整单元数：$completed/15
- 原因：$Reason
- 运行目录：$runDir
- stdout：$stdoutLog
- stderr：$stderrLog

## 决策边界

监控器已经停止训练且禁止自动恢复。旧 scratch、SURE、corruption、开放集、NAS 搜索、HLS、route、COM5、板级与功耗实验均未获本授权，不能继续。下一步必须由用户判断。
"@
    Set-Content -LiteralPath $incident -Value $content -Encoding UTF8
    Write-GuardStatus -Status '已中断' -Detail $Reason
    Disable-OwnTask
    exit 2
}

function Assert-LegacyScratchUnchanged {
    param($Decision)
    if ((Get-LowerSha256 -Path $legacyManifest) -ne [string]$Decision.legacy_scratch_manifest_sha256) {
        throw '旧 scratch 的 run_manifest.json 已发生变化'
    }
    if ((Get-LowerSha256 -Path $legacyPatch) -ne [string]$Decision.legacy_scratch_patch_sha256) {
        throw '旧 scratch 的 code_patch.diff 已发生变化'
    }
}

function Assert-NewManifestPatchIntegrity {
    $manifestPath = Join-Path $runDir 'run_manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        return
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $tracked = $manifest.code_provenance.tracked_patch
    $patchPath = [string]$tracked.path
    $expected = ([string]$tracked.sha256).ToLower()
    if ($expected.Length -ne 64 -or (Get-LowerSha256 -Path $patchPath) -ne $expected) {
        throw '新 run 的 manifest-bound patch SHA256 不一致'
    }
    if ((Split-Path -Leaf $patchPath) -ne ("code_patch_{0}.diff" -f $expected)) {
        throw '新 run 未使用内容寻址 patch 文件名'
    }
}

function Assert-CompletedRecord {
    param([System.IO.FileInfo]$RecordFile, $Decision)
    $record = Get-Content -LiteralPath $RecordFile.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    $fold = [int]$record.fold
    $seed = [int]$record.seed
    if ($fold -notin @(0, 1, 2, 3, 4) -or $seed -notin @(42, 43, 44)) {
        throw "出现未授权 fold/seed：fold=$fold seed=$seed"
    }
    $macroF1 = [double]$record.outer_val.macro_f1
    if ([double]::IsNaN($macroF1) -or [double]::IsInfinity($macroF1)) {
        throw "fold=$fold seed=$seed 的 macro_f1 为非有限值"
    }
    $checkpointPath = [string]$record.checkpoint.path
    $predictionPath = [string]$record.outer_predictions.path
    if ((Get-LowerSha256 -Path $checkpointPath) -ne ([string]$record.checkpoint.sha256).ToLower()) {
        throw "fold=$fold seed=$seed 的 checkpoint SHA256 不一致"
    }
    if ((Get-LowerSha256 -Path $predictionPath) -ne ([string]$record.outer_predictions.sha256).ToLower()) {
        throw "fold=$fold seed=$seed 的逐样本预测 SHA256 不一致"
    }
    $boundFreeze = $record.provenance.source_freeze
    if (
        [string]$boundFreeze.manifest_sha256 -ne [string]$Decision.source_freeze_manifest_sha256 -or
        [string]$boundFreeze.archive_sha256 -ne [string]$Decision.source_snapshot_sha256 -or
        [string]$boundFreeze.verification_status -ne 'PASS'
    ) {
        throw "fold=$fold seed=$seed 的源码冻结绑定不一致"
    }
    $legacyRecordPath = Join-Path $legacyRunDir ("run_fold{0}_seed{1}.json" -f $fold, $seed)
    $legacyRecord = Get-Content -LiteralPath $legacyRecordPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $legacyMacroF1 = [double]$legacyRecord.outer_val.macro_f1
    $delta = [math]::Abs($macroF1 - $legacyMacroF1)
    if ($macroF1 -lt [double]$Decision.stop_policy.minimum_macro_f1_per_unit) {
        throw ("性能中断：fold={0} seed={1} macro_f1={2:N6} 低于阈值 {3:N2}" -f $fold, $seed, $macroF1, [double]$Decision.stop_policy.minimum_macro_f1_per_unit)
    }
    if ($delta -gt [double]$Decision.stop_policy.maximum_absolute_delta_from_legacy_same_pair) {
        throw ("一致性中断：fold={0} seed={1} 新旧 macro_f1 绝对差 {2:N6} 超过阈值 {3:N2}" -f $fold, $seed, $delta, [double]$Decision.stop_policy.maximum_absolute_delta_from_legacy_same_pair)
    }
    return [pscustomobject]@{
        fold = $fold
        seed = $seed
        macro_f1 = $macroF1
        legacy_macro_f1 = $legacyMacroF1
        absolute_delta = $delta
    }
}

try {
    New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
    if (Test-Path -LiteralPath $statusLog) {
        throw "状态日志已存在，拒绝覆盖：$statusLog"
    }
    Write-GuardStatus -Status '预检开始'
    if ((Get-LowerSha256 -Path $approval) -ne $expectedApprovalSha256) {
        throw '一次性批准文件 SHA256 不一致'
    }
    $decision = Get-Content -LiteralPath $approval -Raw -Encoding UTF8 | ConvertFrom-Json
    if (
        $decision.schema_version -ne 1 -or
        $decision.campaign_id -ne 'ccf_ab_nksid_av7k325_v1' -or
        $decision.authorization_status -ne 'APPROVED_ONCE' -or
        $decision.user_decision_recorded -ne $true -or
        $decision.run_name -ne 'g1_mobilenet_v2_scratch_v2' -or
        $decision.allow_sure -ne $false -or
        $decision.allow_hls_or_route -ne $false -or
        $decision.allow_board_or_com5 -ne $false -or
        $decision.allow_power -ne $false
    ) {
        throw '一次性批准文件的范围或禁止项不符合预声明'
    }
    if (Test-Path -LiteralPath $runDir) {
        throw "fresh scratch-v2 目录已经存在，拒绝自动覆盖或恢复：$runDir"
    }
    if ((Get-LowerSha256 -Path (Join-Path $repo 'run_eval_protocol.py')) -ne [string]$decision.entrypoint_sha256) {
        throw '正式分类入口 SHA256 与批准文件不一致'
    }
    if ((Get-LowerSha256 -Path (Join-Path $repo 'scripts\audit_measurement_first_gates.py')) -ne [string]$decision.measurement_ledger_sha256) {
        throw '测量优先总账脚本 SHA256 与批准文件不一致'
    }
    if ((Get-LowerSha256 -Path $freeze) -ne [string]$decision.source_freeze_manifest_sha256) {
        throw '源码冻结 manifest SHA256 与批准文件不一致'
    }
    $freezePayload = Get-Content -LiteralPath $freeze -Raw -Encoding UTF8 | ConvertFrom-Json
    if ((Get-LowerSha256 -Path ([string]$freezePayload.archive.path)) -ne [string]$decision.source_snapshot_sha256) {
        throw '源码冻结归档 SHA256 与批准文件不一致'
    }
    Assert-LegacyScratchUnchanged -Decision $decision
    & $analysisPython (Join-Path $repo 'scripts\freeze_experiment_source.py') verify --manifest $freeze 1>> (Join-Path $logRoot 'source_freeze_verify.log') 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw '源码冻结实时核验失败'
    }
    Write-GuardStatus -Status '预检通过' -Detail '仅授权 scratch-v2 15 单元'
    if ($PreflightOnly) {
        Write-GuardStatus -Status '仅预检结束' -Detail '未启动训练'
        exit 0
    }

    $arguments = @(
        'run_eval_protocol.py',
        '--data-dir', 'data/NKSID',
        '--output-dir', 'results/protocol/g1_clean_20260718',
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
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -WindowStyle Hidden -PassThru
    Write-GuardStatus -Status '训练启动' -Detail ("PID={0}" -f $process.Id)
    $seen = @{}
    while ($true) {
        Start-Sleep -Seconds 10
        $process.Refresh()
        Assert-LegacyScratchUnchanged -Decision $decision
        Assert-NewManifestPatchIntegrity
        $records = @(Get-ChildItem -LiteralPath $runDir -Filter 'run_fold*_seed*.json' -File -ErrorAction SilentlyContinue | Sort-Object Name)
        if ($records.Count -gt 15) {
            throw "完成记录超过 15 个：$($records.Count)"
        }
        foreach ($recordFile in $records) {
            if ($seen.ContainsKey($recordFile.FullName)) {
                continue
            }
            try {
                $checked = Assert-CompletedRecord -RecordFile $recordFile -Decision $decision
            } catch [System.Management.Automation.PSInvalidCastException] {
                continue
            } catch [System.ArgumentException] {
                continue
            }
            $seen[$recordFile.FullName] = $true
            Write-GuardStatus -Status '单元通过' -Detail ("fold={0} seed={1} macro_f1={2:N6} 与旧诊断差={3:N6}" -f $checked.fold, $checked.seed, $checked.macro_f1, $checked.absolute_delta)
        }
        if ($process.HasExited) {
            break
        }
    }
    # Start-Process 返回的对象在轮询 HasExited 后，ExitCode 仍可能暂时为 $null。
    # 必须显式等待并刷新；否则 PowerShell 中 `$null -ne 0` 会把成功运行误报为失败。
    $process.WaitForExit()
    $process.Refresh()
    $trainingExitCode = $process.ExitCode
    if ($null -eq $trainingExitCode) {
        throw '训练进程退出码不可用'
    }
    if ([int]$trainingExitCode -ne 0) {
        throw "训练进程非零退出：$trainingExitCode"
    }
    if ($seen.Count -ne 15) {
        throw "训练退出后仅核验 $($seen.Count)/15 个单元"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $runDir 'protocol_summary.json') -PathType Leaf)) {
        throw '训练退出后缺少 protocol_summary.json'
    }
    Write-GuardStatus -Status '训练完成' -Detail '15/15 单元已通过在线检查'

    & $analysisPython $auditScript --run-dir $runDir --output $auditOutput --expect-pretrained false --expected-method scratch_mobilenet_v2 1>> (Join-Path $logRoot 'independent_audit.log') 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw 'scratch-v2 独立审计未通过'
    }
    $audit = Get-Content -LiteralPath $auditOutput -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($audit.status -ne 'PASS' -or [int]$audit.record_count -ne 15) {
        throw 'scratch-v2 独立审计状态或记录数不符合要求'
    }
    & $analysisPython (Join-Path $repo 'scripts\audit_measurement_first_gates.py') --output-dir $ledgerOutput 1>> (Join-Path $logRoot 'measurement_ledger.log') 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw '测量优先总账脚本执行失败'
    }
    $ledger = Get-Content -LiteralPath (Join-Path $ledgerOutput 'status.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($ledger.gates.G1_accuracy_baselines.pass -ne $true) {
        throw '测量优先总账的 G1 仍未通过'
    }
    $summary = Get-Content -LiteralPath (Join-Path $runDir 'protocol_summary.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $completion = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1\manifests\scratch_v2_completion_20260718.md'
    $completionText = @"
# scratch-v2 完成记录

- 状态：PASS
- 正式单元：15/15
- mean macro_f1（平均宏 F1）：$($summary.outer_macro_f1.mean)
- mean top1（平均 Top-1 准确率）：$($summary.outer_top1.mean)
- 独立审计：PASS
- 测量优先总账 G1：PASS
- 运行目录：$runDir
- 独立审计：$auditOutput
- 总账快照：$(Join-Path $ledgerOutput 'status.json')

## 边界

本授权到此消费完毕。没有启动 SURE、corruption、开放集、NAS 新搜索、HLS、route、COM5、AV7K325 板级或功耗实验。后续关键方向由用户决定。
"@
    Set-Content -LiteralPath $completion -Value $completionText -Encoding UTF8
    Move-Item -LiteralPath $approval -Destination $approvalConsumed
    Write-GuardStatus -Status '全部完成并暂停' -Detail '独立审计与 G1 总账均为 PASS；批准已消费'
    Disable-OwnTask
    exit 0
} catch {
    Stop-WithChineseIncident -Reason $_.Exception.Message
}
