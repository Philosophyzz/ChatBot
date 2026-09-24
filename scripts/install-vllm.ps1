# Install/finish vLLM without rebooting Windows. Run again after a requested reboot.
[CmdletBinding()]
param([switch]$PrepareOnly)
$ErrorActionPreference = 'Stop'
$installMutex = New-Object System.Threading.Mutex($false, 'Local\ChatBot-vLLM-Install')
if (-not $installMutex.WaitOne(0)) { throw 'Another vLLM installation is already running.' }
. (Join-Path $PSScriptRoot 'common.ps1')
$ProjectRoot = Get-ConfigRoot
$python = Get-Python -Root $ProjectRoot
$statePath = Join-Path $ProjectRoot 'data\vllm-install.json'
$spec = Get-Content (Join-Path $ProjectRoot 'config\vllm.json') -Raw -Encoding UTF8 | ConvertFrom-Json
function Write-InstallState([string]$State, [string]$Detail) {
    @{state=$State; detail=$Detail; distro=$spec.distro; version=$spec.version} | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
}

# Downloads are independent of WSL boot, so can be prepared before restarting Windows.
$cache = Join-Path $ProjectRoot 'venvs\vllm-install'
New-Item -ItemType Directory -Path $cache -Force | Out-Null
$msi = Join-Path $cache 'wsl.2.7.14.0.x64.msi'
$rootfs = Join-Path $cache 'ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz'
if (-not (Test-Path -LiteralPath $msi)) {
    & curl.exe -f -L --retry 3 --output ($msi + '.part') 'https://github.com/microsoft/WSL/releases/download/2.7.14/wsl.2.7.14.0.x64.msi'
    if ($LASTEXITCODE -ne 0) { throw 'WSL installer download failed' }
    Move-Item -LiteralPath ($msi + '.part') -Destination $msi
}
$signature = Get-AuthenticodeSignature -LiteralPath $msi
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Microsoft Corporation') { throw 'WSL installer signature is invalid' }
if (-not (Test-Path -LiteralPath $rootfs)) {
    & curl.exe -f -L --retry 3 --output ($rootfs + '.part') 'https://cloud-images.ubuntu.com/wsl/releases/24.04/20240423/ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz'
    if ($LASTEXITCODE -ne 0) { throw 'Ubuntu download failed' }
    Move-Item -LiteralPath ($rootfs + '.part') -Destination $rootfs
}
# Pinned against Ubuntu's 20240423/SHA256SUMS, so a later "current" release cannot
# invalidate the already verified download when installation resumes after reboot.
$expected = '8251e27ffff381a4af5f41dcb94d867de3e0d9774a9241908ab34555d99315ea'
if (-not $expected -or (Get-FileHash -LiteralPath $rootfs -Algorithm SHA256).Hash -ne $expected) { throw 'Ubuntu SHA256 mismatch' }
if ($PrepareOnly) { Write-Host 'Signed WSL installer and verified Ubuntu image prepared.'; exit 0 }

$features = @('VirtualMachinePlatform', 'Microsoft-Windows-Subsystem-Linux')
$needsRestart = Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending'
foreach ($feature in $features) {
    $state = (Get-WindowsOptionalFeature -Online -FeatureName $feature).State
    if ($state -eq 'EnablePending') { $needsRestart = $true }
    elseif ($state -ne 'Enabled') {
        $result = Enable-WindowsOptionalFeature -Online -FeatureName $feature -All -NoRestart
        if ($result.RestartNeeded) { $needsRestart = $true }
    }
}
if ($needsRestart) {
    Write-InstallState 'reboot_required' 'WSL2 组件已启用，等待重启 Windows；随后重新运行 scripts/install-vllm.ps1。当前模型仍为原服务。'
    Write-Host 'Restart Windows, then run this script again. No reboot was performed.'
    exit 3010
}

$process = Start-Process msiexec.exe -ArgumentList @('/i', ('"' + $msi + '"'), '/qn', '/norestart') -WindowStyle Hidden -PassThru -Wait
if ($process.ExitCode -notin @(0, 1638, 3010)) { throw "WSL installation failed: $($process.ExitCode)" }
if ($process.ExitCode -eq 3010) { Write-InstallState 'reboot_required' 'WSL 安装完成，需要重启后继续。'; exit 3010 }
$distros = ((& wsl.exe --list --quiet 2>$null) -join "`n").Replace([string][char]0, '')
if ($distros -notmatch [regex]::Escape($spec.distro)) {
    $location = Join-Path $ProjectRoot 'venvs\wsl-vllm'
    & wsl.exe --import $spec.distro $location $rootfs --version 2
    if ($LASTEXITCODE -ne 0) { throw 'WSL import failed. Confirm Windows has been restarted and firmware virtualization is enabled.' }
}
Write-InstallState 'installing' '正在安装独立 Linux Python / CUDA vLLM 环境'
& wsl.exe -d $spec.distro -u root --exec apt-get update
if ($LASTEXITCODE -ne 0) { throw 'Ubuntu apt update failed' }
& wsl.exe -d $spec.distro -u root --exec apt-get install -y python3-venv python3-pip libgomp1
if ($LASTEXITCODE -ne 0) { throw 'Ubuntu dependencies failed' }
& wsl.exe -d $spec.distro -u root --exec python3 -m venv /opt/chatbot-vllm
if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed' }
& wsl.exe -d $spec.distro -u root --exec $spec.python -m pip install --upgrade pip uv
if ($LASTEXITCODE -ne 0) { throw 'pip/uv install failed' }
& wsl.exe -d $spec.distro -u root --exec /opt/chatbot-vllm/bin/uv pip install --python $spec.python ('vllm==' + $spec.version) --torch-backend=auto
if ($LASTEXITCODE -ne 0) { throw 'vLLM installation failed' }
& wsl.exe -d $spec.distro -u root --exec $spec.python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
if ($LASTEXITCODE -ne 0) { throw 'WSL CUDA device is unavailable' }
if (-not (Test-Path -LiteralPath (Join-Path $spec.models[0].path '.complete.json'))) {
    & $python (Join-Path $PSScriptRoot 'download_vllm_model.py')
    if ($LASTEXITCODE -ne 0) { throw 'Model download failed' }
}
Write-InstallState 'ready' 'vLLM 已安装；正在启动和验证模型'
# Stop only this project's services; the running desktop pet reconnects after activation.
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'stop.ps1') -KeepPets
& $python (Join-Path $PSScriptRoot 'vllm_control.py') activate
if ($LASTEXITCODE -ne 0) {
    # Activation persists the new backend only after a real completion succeeds.
    # Release the failed candidate before restoring the previous configured service.
    & $python (Join-Path $PSScriptRoot 'vllm_control.py') stop
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'start-all.ps1') -NoBrowser
    Write-InstallState 'ready' 'vLLM 已安装但模型启动失败，查看 logs/vllm.log。配置尚未切换，已尝试恢复原服务。'
    throw 'vLLM model activation failed'
}
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'start-all.ps1') -NoBrowser
if ($LASTEXITCODE -ne 0) { throw 'ChatBot restart failed' }
Write-InstallState 'ready' 'vLLM 已激活，ChatBot 使用本地 vLLM 服务'
Write-Host 'vLLM activated. ChatBot is ready.'
