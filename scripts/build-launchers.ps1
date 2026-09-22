# 打包双击启动/停止的 exe（输出到 dist\）
#
#   powershell -ExecutionPolicy Bypass -File scripts\build-launchers.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\build-launchers.ps1 -Clean
#
# 产物：
#   dist\启动聊天机器人.exe   双击 → 起模型 + 网页 + 桌宠 + 开浏览器（已在跑则直接复用）
#   dist\停止聊天机器人.exe   双击 → 桌宠 + 服务全部优雅停止
#
# 这两个 exe 是**控制台程序**（不是窗口程序），故意如此：双击后会显示进度、
# 出错时报错就在眼前，而不是默默什么都不发生。体积也小（约 8~10MB，不含 Qt）。
#
# 它们通过 exe 所在目录（或其上一级）里的 scripts\start-all.ps1 / stop.ps1 定位项目，
# 所以把 exe 从 dist\ 挪到别处也行；实在找不到会提示用 --root 指定。
[CmdletBinding()]
param(
    [string]$Root,
    [switch]$Clean
)

. (Join-Path $PSScriptRoot 'common.ps1')
$ProjectRoot = Get-ConfigRoot -Root $Root
$py = Get-Python -Root $ProjectRoot
if (-not $py) { Write-Err2 '找不到 Python（conda 环境 chatbot）'; exit 1 }

& $py -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Step '安装 PyInstaller'
    & $py -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyinstaller
    if ($LASTEXITCODE -ne 0) { Write-Err2 'pyinstaller 安装失败'; exit 1 }
}

$launcherDir = Join-Path $ProjectRoot 'launcher'
$distDir = Join-Path $ProjectRoot 'dist'
$buildDir = Join-Path $ProjectRoot 'build'

if ($Clean) {
    foreach ($suffix in '启动聊天机器人', '停止聊天机器人') {
        $spec = Join-Path $buildDir "$suffix.spec"
        if (Test-Path -LiteralPath $spec) { Remove-Item -LiteralPath $spec -Force }
    }
}

$targets = @(
    @{ Script = 'start_chatbot.py'; Name = 'ChatBotStart' },
    @{ Script = 'stop_chatbot.py';  Name = 'ChatBotStop' }
)

foreach ($target in $targets) {
    $entry = Join-Path $launcherDir $target.Script
    if (-not (Test-Path -LiteralPath $entry)) { Write-Err2 "找不到 $entry"; exit 1 }

    Write-Step "打包 $($target.Name)（来自 $($target.Script)）"
    & $py -m PyInstaller `
        --noconfirm `
        --console `
        --onefile `
        --name $target.Name `
        --distpath $distDir `
        --workpath (Join-Path $buildDir 'launcher') `
        --specpath $buildDir `
        --exclude-module PySide6 `
        --exclude-module numpy `
        --exclude-module torch `
        $entry
    if ($LASTEXITCODE -ne 0) { Write-Err2 "$($target.Name) 打包失败"; exit 1 }
}

# 改成中文文件名（exe 不依赖自己的文件名，改名安全；--name 用中文在个别工具链上会出问题）
$renames = @(
    @{ From = 'ChatBotStart.exe'; To = '启动聊天机器人.exe' },
    @{ From = 'ChatBotStop.exe';  To = '停止聊天机器人.exe' }
)
Write-Step '改成便于双击的中文名'
foreach ($pair in $renames) {
    $from = Join-Path $distDir $pair.From
    $to = Join-Path $distDir $pair.To
    if (Test-Path -LiteralPath $from) {
        if (Test-Path -LiteralPath $to) { Remove-Item -LiteralPath $to -Force }
        Move-Item -LiteralPath $from -Destination $to -Force
        Write-Ok "$($pair.From) -> $($pair.To)"
    } else {
        Write-Warn2 "没找到 $($pair.From)"
    }
}

Write-Host ''
Write-Host '============================================================' -ForegroundColor Green
Write-Host ' 双击即可使用的三个入口（都在 dist\）：' -ForegroundColor Green
foreach ($name in '启动聊天机器人.exe', '停止聊天机器人.exe', 'ChatBotPet.exe') {
    $path = Join-Path $distDir $name
    if (Test-Path -LiteralPath $path) {
        Write-Host ("   {0,-22} {1,7:N1} MB" -f $name, ((Get-Item -LiteralPath $path).Length / 1MB))
    } else {
        Write-Host ("   {0,-22} （还没打包：跑 scripts\build-pet.ps1）" -f $name)
    }
}
Write-Host '============================================================' -ForegroundColor Green
