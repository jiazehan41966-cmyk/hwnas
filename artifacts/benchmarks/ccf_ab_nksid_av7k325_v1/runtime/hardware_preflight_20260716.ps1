param(
    [string]$VivadoPath = '',
    [string]$HlsPath = '',
    [string]$SerialPort = 'COM5',
    [string]$InstrumentCommand = '',
    [switch]$ProbeVersions,
    [switch]$RequireReady
)

$ErrorActionPreference = 'Stop'

function Resolve-Executable {
    param(
        [string]$ExplicitPath,
        [string[]]$Names,
        [string[]]$FallbackPatterns = @()
    )
    if ($ExplicitPath) {
        if (Test-Path -LiteralPath $ExplicitPath -PathType Leaf) {
            return (Resolve-Path -LiteralPath $ExplicitPath).Path
        }
        return $null
    }
    foreach ($name in $Names) {
        $command = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -ne $command) {
            return $command.Source
        }
    }
    foreach ($pattern in $FallbackPatterns) {
        $matches = @(
            Get-Item -Path $pattern -ErrorAction SilentlyContinue |
                Where-Object { -not $_.PSIsContainer } |
                Sort-Object FullName -Descending
        )
        if ($matches.Count -gt 0) {
            return $matches[0].FullName
        }
    }
    return $null
}

function Get-ToolVersionLine {
    param([string]$Executable, [string]$Pattern)
    if (-not $Executable) {
        return $null
    }
    try {
        return @(& $Executable -version 2>&1) |
            ForEach-Object { [string]$_ } |
            Where-Object { $_ -match $Pattern } |
            Select-Object -First 1
    }
    catch {
        return $null
    }
}

$vivadoFallbacks = @(
    'F:\vivado\Vivado\*\bin\vivado.bat',
    'C:\Xilinx\Vivado\*\bin\vivado.bat',
    'D:\Xilinx\Vivado\*\bin\vivado.bat',
    'E:\Xilinx\Vivado\*\bin\vivado.bat',
    'F:\Xilinx\Vivado\*\bin\vivado.bat',
    'C:\AMD\Vivado\*\bin\vivado.bat',
    'D:\AMD\Vivado\*\bin\vivado.bat',
    'E:\AMD\Vivado\*\bin\vivado.bat',
    'F:\AMD\Vivado\*\bin\vivado.bat'
)
$hlsFallbacks = @(
    'F:\vivado\Vitis_HLS\*\bin\vitis_hls.bat',
    'C:\Xilinx\Vitis_HLS\*\bin\vitis_hls.bat',
    'D:\Xilinx\Vitis_HLS\*\bin\vitis_hls.bat',
    'E:\Xilinx\Vitis_HLS\*\bin\vitis_hls.bat',
    'F:\Xilinx\Vitis_HLS\*\bin\vitis_hls.bat',
    'C:\AMD\Vitis_HLS\*\bin\vitis_hls.bat',
    'D:\AMD\Vitis_HLS\*\bin\vitis_hls.bat',
    'E:\AMD\Vitis_HLS\*\bin\vitis_hls.bat',
    'F:\AMD\Vitis_HLS\*\bin\vitis_hls.bat'
)

$vivado = Resolve-Executable -ExplicitPath $VivadoPath -Names @('vivado') -FallbackPatterns $vivadoFallbacks
$hls = Resolve-Executable -ExplicitPath $HlsPath -Names @('vitis_hls', 'vivado_hls') -FallbackPatterns $hlsFallbacks
$port = Get-PnpDevice -Class Ports -PresentOnly -ErrorAction SilentlyContinue |
    Where-Object { $_.FriendlyName -match "\($([regex]::Escape($SerialPort))\)" } |
    Select-Object -First 1
$instrumentExecutable = $null
if ($InstrumentCommand) {
    $candidate = $InstrumentCommand.Trim().Split(' ')[0].Trim('"')
    $instrumentExecutable = Resolve-Executable -ExplicitPath $candidate -Names @($candidate)
}

$checks = [ordered]@{
    vivado_callable = [bool]$vivado
    hls_callable = [bool]$hls
    serial_port_present = ($null -ne $port -and $port.Status -eq 'OK')
    external_instrument_command_callable = [bool]$instrumentExecutable
}
$ready = -not ($checks.Values -contains $false)
$result = [ordered]@{
    schema_version = 1
    generated = (Get-Date).ToString('o')
    status = if ($ready) { 'READY_FOR_HARDWARE_COLLECTION' } else { 'NOT_READY' }
    checks = $checks
    observed = [ordered]@{
        vivado = $vivado
        vivado_version = if ($ProbeVersions) { Get-ToolVersionLine -Executable $vivado -Pattern '^vivado v' } else { $null }
        hls = $hls
        hls_version = if ($ProbeVersions) { Get-ToolVersionLine -Executable $hls -Pattern '^Vitis HLS' } else { $null }
        serial_port = $SerialPort
        serial_friendly_name = if ($port) { $port.FriendlyName } else { $null }
        serial_instance_id = if ($port) { $port.InstanceId } else { $null }
        instrument_command = if ($InstrumentCommand) { $InstrumentCommand } else { $null }
        instrument_executable = $instrumentExecutable
    }
    boundary = 'COM/UART presence is not HLS, route, bitstream, inference, or external-power evidence.'
}
$result | ConvertTo-Json -Depth 6
if ($RequireReady -and -not $ready) {
    exit 2
}
exit 0
