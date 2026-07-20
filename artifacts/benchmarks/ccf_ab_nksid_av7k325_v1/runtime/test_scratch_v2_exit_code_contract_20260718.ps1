$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..\..')).Path
$campaign = Join-Path $repo 'artifacts\benchmarks\ccf_ab_nksid_av7k325_v1'
$wrapper = Join-Path $campaign 'runtime\run_scratch_v2_guarded_20260718.ps1'
$jsonOutput = Join-Path $campaign 'manifests\scratch_v2_exit_code_contract_test_20260718.json'
$mdOutput = Join-Path $campaign 'manifests\scratch_v2_exit_code_contract_test_20260718.md'

function Get-ObservedExitCode {
    param([int]$RequestedExitCode)
    $child = Start-Process -FilePath 'powershell.exe' -ArgumentList @(
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        "exit $RequestedExitCode"
    ) -WindowStyle Hidden -PassThru
    $child.WaitForExit()
    $child.Refresh()
    if ($null -eq $child.ExitCode) {
        throw "子进程退出码不可用：请求值=$RequestedExitCode"
    }
    return [int]$child.ExitCode
}

$observedZero = Get-ObservedExitCode -RequestedExitCode 0
$observedSeven = Get-ObservedExitCode -RequestedExitCode 7
$nullComparisonWouldMisclassify = ($null -ne 0)
$wrapperText = Get-Content -LiteralPath $wrapper -Raw -Encoding UTF8
$staticContract = (
    $wrapperText.Contains('$process.WaitForExit()') -and
    $wrapperText.Contains('$process.Refresh()') -and
    $wrapperText.Contains('$null -eq $trainingExitCode') -and
    $wrapperText.Contains('[int]$trainingExitCode -ne 0')
)
$passed = (
    $observedZero -eq 0 -and
    $observedSeven -eq 7 -and
    $nullComparisonWouldMisclassify -and
    $staticContract
)
$payload = [ordered]@{
    schema_version = 1
    language = 'zh-CN'
    status = $(if ($passed) { 'PASS' } else { 'FAIL' })
    observed_exit_code_zero = $observedZero
    observed_exit_code_seven = $observedSeven
    powershell_null_ne_zero = $nullComparisonWouldMisclassify
    wrapper_static_contract = $staticContract
    wrapper = $wrapper.Substring($repo.Length + 1).Replace('\', '/')
    wrapper_sha256 = (Get-FileHash -LiteralPath $wrapper -Algorithm SHA256).Hash.ToLowerInvariant()
    conclusion_zh = '显式 WaitForExit、Refresh 和空值检查能够区分成功退出、非零退出与退出码不可用。'
    boundary_zh = '本测试仅验证守护脚本退出码判定，不重新运行训练，也不改变已有实验结果。'
}
$payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $jsonOutput -Encoding UTF8

$statusZh = if ($passed) { '通过' } else { '失败' }
$markdown = @"
# scratch-v2 守护脚本退出码回归测试

- 状态：**$statusZh**（`$($payload.status)`）。
- 成功子进程观测退出码：`$observedZero`。
- 非零子进程观测退出码：`$observedSeven`。
- PowerShell 中 `` `$null -ne 0 `` 的结果：`$nullComparisonWouldMisclassify`；这解释了旧逻辑为何会在训练已成功结束后误报中断。
- 修复后的静态契约：`$staticContract`，包含显式 `WaitForExit()`、`Refresh()`、空值分支和整数退出码判断。
- 守护脚本 SHA256：`$($payload.wrapper_sha256)`。

## 边界

本测试没有重新运行训练，没有修改 15 个 fold-seed 结果，也没有启动 SURE、HLS、route、COM5、板级或功耗实验。
"@
Set-Content -LiteralPath $mdOutput -Value $markdown -Encoding UTF8

Write-Output ($payload | ConvertTo-Json -Compress)
if (-not $passed) {
    exit 1
}

