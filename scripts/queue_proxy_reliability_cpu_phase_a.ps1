param(
    [string]$RunMatrix = "results/proxy_reliability_gate0/manifest_v2/run_matrix.jsonl",
    [double]$MinimumFreeMemoryGB = 4.0,
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = (Get-Command python -ErrorAction Stop).Source
$MatrixPath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $RunMatrix))
$RunRoot = Split-Path -Parent $MatrixPath
$QueueRoot = Join-Path $RunRoot "cpu_phase_a_queue"
$StatusPath = Join-Path $QueueRoot "queue_status.json"
$StdoutPath = Join-Path $QueueRoot "worker.stdout.log"
$StderrPath = Join-Path $QueueRoot "worker.stderr.log"
New-Item -ItemType Directory -Force -Path $QueueRoot | Out-Null

function Write-QueueStatus {
    param(
        [string]$State,
        [object[]]$BlockingProcesses,
        [double]$FreeMemoryGB,
        [Nullable[int]]$WorkerPid = $null,
        [Nullable[int]]$ExitCode = $null
    )
    $payload = [ordered]@{
        schema_version = 1
        state = $State
        updated_at = (Get-Date).ToString("o")
        run_matrix = $MatrixPath
        stage = "phase_a_signal_discovery"
        device = "cpu"
        minimum_free_memory_gb = $MinimumFreeMemoryGB
        free_memory_gb = $FreeMemoryGB
        blocking_processes = @(
            $BlockingProcesses | ForEach-Object {
                [ordered]@{
                    process_id = $_.ProcessId
                    parent_process_id = $_.ParentProcessId
                    command_line = $_.CommandLine
                }
            }
        )
        worker_pid = $WorkerPid
        exit_code = $ExitCode
        stdout = $StdoutPath
        stderr = $StderrPath
    }
    $temporary = "$StatusPath.tmp"
    $payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $StatusPath -Force
}

while ($true) {
    $blocking = @(
        Get-CimInstance Win32_Process |
            Where-Object {
                (
                    $_.Name -like "python*" -and
                    $_.CommandLine -like "*run_eval_protocol.py*"
                ) -or (
                    $_.Name -like "powershell*" -and
                    $_.ProcessId -ne $PID -and
                    $_.CommandLine -like "*run_g1_baseline_trio.ps1*"
                )
            }
    )
    $os = Get-CimInstance Win32_OperatingSystem
    $freeMemoryGB = [math]::Round($os.FreePhysicalMemory * 1KB / 1GB, 2)
    if ($blocking.Count -eq 0 -and $freeMemoryGB -ge $MinimumFreeMemoryGB) {
        break
    }
    Write-QueueStatus `
        -State "waiting_for_existing_training" `
        -BlockingProcesses $blocking `
        -FreeMemoryGB $freeMemoryGB
    Start-Sleep -Seconds $PollSeconds
}

$arguments = @(
    "scripts/run_proxy_reliability_worker.py",
    "--run-matrix", $MatrixPath,
    "--stage", "phase_a_signal_discovery",
    "--shard-index", "0",
    "--num-shards", "1",
    "--device", "cpu",
    "--no-require-cuda",
    "--num-workers", "0"
)
$worker = Start-Process `
    -FilePath $Python `
    -ArgumentList $arguments `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $StdoutPath `
    -RedirectStandardError $StderrPath `
    -WindowStyle Hidden `
    -PassThru

$os = Get-CimInstance Win32_OperatingSystem
$freeMemoryGB = [math]::Round($os.FreePhysicalMemory * 1KB / 1GB, 2)
Write-QueueStatus `
    -State "running_phase_a" `
    -BlockingProcesses @() `
    -FreeMemoryGB $freeMemoryGB `
    -WorkerPid $worker.Id

$worker.WaitForExit()
$os = Get-CimInstance Win32_OperatingSystem
$freeMemoryGB = [math]::Round($os.FreePhysicalMemory * 1KB / 1GB, 2)
Write-QueueStatus `
    -State ($(if ($worker.ExitCode -eq 0) { "completed" } else { "failed" })) `
    -BlockingProcesses @() `
    -FreeMemoryGB $freeMemoryGB `
    -WorkerPid $worker.Id `
    -ExitCode $worker.ExitCode
exit $worker.ExitCode
