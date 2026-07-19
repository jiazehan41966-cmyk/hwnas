param(
    [Parameter(Mandatory = $true)][int]$WrapperPid
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
$statusLog = Join-Path $repo 'logs\g1_clean_20260718\scratch_v2_guard_status.tsv'
$watchLog = Join-Path $repo 'logs\g1_clean_20260718\scratch_v2_process_tree_watch.tsv'
$commandMarker = 'g1_mobilenet_v2_scratch_v2'

function Write-WatchStatus {
    param([string]$Status, [string]$Detail = '')
    $line = "{0}`t{1}`t{2}" -f (Get-Date).ToString('o'), $Status, $Detail
    Add-Content -LiteralPath $watchLog -Value $line -Encoding UTF8
}

function Stop-MatchingTrainingTrees {
    $matches = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and
        $_.CommandLine -like "*$commandMarker*"
    })
    $roots = @($matches | Where-Object {
        $candidate = $_
        -not ($matches | Where-Object { $_.ProcessId -eq $candidate.ParentProcessId })
    })
    foreach ($root in $roots) {
        & taskkill.exe /PID $root.ProcessId /T /F 1>> $watchLog 2>&1
    }
    Write-WatchStatus -Status '训练进程树已终止' -Detail ("匹配进程={0} 根进程={1}" -f $matches.Count, $roots.Count)
}

if (Test-Path -LiteralPath $watchLog) {
    exit 2
}
Write-WatchStatus -Status '看门狗启动' -Detail ("wrapper PID={0}" -f $WrapperPid)

while ($true) {
    Start-Sleep -Seconds 2
    $status = ''
    if (Test-Path -LiteralPath $statusLog -PathType Leaf) {
        $status = Get-Content -LiteralPath $statusLog -Raw -Encoding UTF8
    }
    if ($status -match '全部完成并暂停') {
        Write-WatchStatus -Status '正常完成' -Detail '训练已退出且审计通过，无需终止进程'
        exit 0
    }
    if ($status -match '已中断') {
        Stop-MatchingTrainingTrees
        exit 2
    }
    if (-not (Get-Process -Id $WrapperPid -ErrorAction SilentlyContinue)) {
        Stop-MatchingTrainingTrees
        Write-WatchStatus -Status '监控器异常退出' -Detail '未发现正常完成标志，已 fail-closed'
        exit 2
    }
}
