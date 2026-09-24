# Stop this project's processes only. Never trust stale PIDs without checking ownership.
[CmdletBinding()]
param([string]$Root, [switch]$KeepPets)
. (Join-Path $PSScriptRoot 'common.ps1')
$ProjectRoot = [IO.Path]::GetFullPath((Get-ConfigRoot -Root $Root)).TrimEnd('\')
$prefix = $ProjectRoot + '\'
function Get-OwnedProcesses {
    Get-CimInstance Win32_Process | Where-Object {
        ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) -or
        ($_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -and
         $_.CommandLine.IndexOf($prefix, [StringComparison]::OrdinalIgnoreCase) -ge 0)
    }
}
$owned = @(Get-OwnedProcesses)
$api = @($owned | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'run_server\.py|api\.server' })
if ($api.Count) {
    Write-Step '停止网页服务'
    $result = Invoke-LocalHttp -Uri 'http://127.0.0.1:8077/api/shutdown' -Method POST -TimeoutSec 5
    if ($result.Ok) {
        foreach ($proc in $api) { Wait-Process -Id $proc.ProcessId -Timeout 15 -ErrorAction SilentlyContinue }
    }
}
foreach ($proc in @(Get-OwnedProcesses)) {
    $isApi = $proc.Name -match '^python(w)?\.exe$' -and $proc.CommandLine -match 'run_server\.py|api\.server|tts_worker\.py'
    $isModel = $proc.Name -eq 'llama-server.exe'
    $isPet = $proc.Name -in @('ChatBotPet.exe', 'ChatBot.exe') -or
             ($proc.Name -match '^python(w)?\.exe$' -and $proc.CommandLine -match 'run_pet\.py') -or
             ($proc.ExecutablePath -eq (Join-Path $ProjectRoot '启动聊天机器人.exe'))
    if ($isApi -or $isModel -or ($isPet -and -not $KeepPets)) {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Ok "Stopped $($proc.Name), PID $($proc.ProcessId)"
    }
}
$vllmState = Join-Path $ProjectRoot 'data\vllm-install.json'
if (Test-Path -LiteralPath $vllmState) {
    $installed = Get-Content -LiteralPath $vllmState -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($installed.state -eq 'ready') {
        $python = Get-Python -Root $ProjectRoot
        & $python (Join-Path $ProjectRoot 'scripts\vllm_control.py') stop
    }
}
foreach ($name in @('app.pids.json', 'model-server.pids.json')) {
    $path = Join-Path $ProjectRoot "data\$name"
    if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
}
Write-Ok '本项目服务已停止'
