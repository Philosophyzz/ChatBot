# 启动桌宠（源码方式，需要 conda 环境 chatbot）
#
#   powershell -ExecutionPolicy Bypass -File scripts\start-pet.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\start-pet.ps1 -NoBackend   # 不检查/不启动后端
#
# 桌宠是**瘦客户端**：模型推理、记忆、语音都在本地服务里（默认 8077）。
# 这个脚本只负责找到解释器、必要时把后端拉起来，然后把窗口显示出来。
[CmdletBinding()]
param(
    [string]$Root,
    [string]$Url = 'http://127.0.0.1:8077',
    [string]$Persona = '',
    [switch]$NoBackend,
    [switch]$Hidden
)

. (Join-Path $PSScriptRoot 'common.ps1')
$ProjectRoot = Get-ConfigRoot -Root $Root
$py = Get-Python -Root $ProjectRoot
if (-not $py) { Write-Err2 '找不到 Python（conda 环境 chatbot 或 venvs\main）'; exit 1 }

if (-not $NoBackend) {
    $probe = Invoke-LocalHttp -Uri "$Url/api/health" -TimeoutSec 3
    if (-not $probe.Ok) {
        Write-Warn2 '本地服务没在运行，先启动它'
        & (Join-Path $PSScriptRoot 'start-all.ps1') -Root $ProjectRoot -NoBrowser
    } else {
        Write-Ok "本地服务正常：$Url"
    }
}

$args = @((Join-Path $ProjectRoot 'run_pet.py'), '--url', $Url)
if ($Persona) { $args += @('--persona', $Persona) }
if ($NoBackend) { $args += '--no-backend' }

Write-Step '启动桌宠'
Write-Host "  解释器：$py"
Write-Host '  按住麦克风按钮说话；右键打开菜单；托盘图标可隐藏/显示；滚轮缩放；单击触摸；双击聊天；右键选择打开网页。'

if ($Hidden) {
    Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $ProjectRoot -WindowStyle Hidden | Out-Null
} else {
    & $py @args
}
