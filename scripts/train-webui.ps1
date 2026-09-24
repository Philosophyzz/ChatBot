# =============================================================================
# 启动 GPT-SoVITS 自带的**可视化训练页面**（Gradio WebUI）。
#
#   powershell -ExecutionPolicy Bypass -File scripts\train-webui.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\train-webui.ps1 -SkipChecks
#
# 为什么需要这个脚本：上游的 go-webui.ps1 假设仓库里自带 runtime\python.exe（官方整合包），
# 而本项目的约定是**所有依赖装在 conda 环境 chatbot 里**。webui.py 里
# `python_exec = sys.executable`，所以只要用它启动，训练/推理的子进程就都跑在同一个环境里。
#
# 页面（本机 127.0.0.1）：
#   9874  主界面：音频切分 / 降噪 / ASR 标注 / 1A 文本与特征提取 / 1B-a SoVITS 训练 / 1B-b GPT 训练
#   9872  推理（试听训练结果）
#   9873  UVR5 人声分离（去背景音乐）
#   9871  字幕打轴
#
# 训练显存 6~9GB：建议先 scripts\stop.ps1 停掉对话模型，否则 16GB 卡会紧张。
# 素材怎么准备、命令行怎么训（不想用页面时）：见 docs\训练音色.md
# =============================================================================
[CmdletBinding()]
param(
    [string]$Root,
    [string]$Language = 'zh_CN',
    [switch]$SkipChecks
)

. (Join-Path $PSScriptRoot 'common.ps1')
$ProjectRoot = Get-ConfigRoot -Root $Root
$py = Get-Python -Root $ProjectRoot
if (-not $py) { Write-Err2 '找不到 Python（conda 环境 chatbot）'; exit 1 }

$repo = Join-Path $ProjectRoot 'vendor\GPT-SoVITS'
$webui = Join-Path $repo 'webui.py'

Write-Host ''
Write-Host '============================================================' -ForegroundColor White
Write-Host ' GPT-SoVITS 可视化训练页面' -ForegroundColor White
Write-Host '============================================================' -ForegroundColor White

if (-not (Test-Path -LiteralPath $webui)) {
    Write-Err2 "找不到 $webui"
    Write-Host '    先运行：powershell -ExecutionPolicy Bypass -File scripts\install-gptsovits.ps1'
    exit 1
}

if (-not $SkipChecks) {
    Write-Step '检查训练所需文件（脚本 / 预训练基座 / G2PW / ffmpeg）'
    & $py (Join-Path $ProjectRoot 'tools\check_gptsovits.py') --repo $repo
    if ($LASTEXITCODE -ne 0) {
        Write-Err2 '训练文件不齐 —— 先跑 scripts\install-gptsovits.ps1（会断点续传）'
        exit 1
    }
    Write-Step '检查页面依赖（gradio）'
    & $py -c "import gradio, sys; print('  gradio', gradio.__version__)"
    if ($LASTEXITCODE -ne 0) {
        Write-Err2 '缺 gradio，装它：'
        Write-Host '    python -m pip install "gradio==4.44.1" -i https://pypi.tuna.tsinghua.edu.cn/simple' -ForegroundColor Yellow
        exit 1
    }
}

# 端口从上游 config.py 读，免得写死后与它对不上
$portMain = 9874
try {
    $cfgText = Get-Content -LiteralPath (Join-Path $repo 'config.py') -Raw -Encoding UTF8
    if ($cfgText -match 'webui_port_main\s*=\s*(\d+)') { $portMain = [int]$Matches[1] }
} catch { }

# --- 环境变量 ---------------------------------------------------------------
$envRoot = Split-Path -Parent $py
# ffmpeg/ffprobe 在仓库根目录（上游自带），训练与切分都要用
$env:PATH = "$repo;$envRoot;$envRoot\Scripts;$envRoot\Library\bin;$env:PATH"
# 本项目一贯的下载源约定：镜像 + 关掉 Xet（镜像不代理 Xet，会 401）
if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = 'https://hf-mirror.com' }
if (-not $env:HF_HUB_DISABLE_XET) { $env:HF_HUB_DISABLE_XET = '1' }
# 本机 Clash 在 7890：localhost 不能走代理。除了 NO_PROXY，还要把 HTTP_PROXY 之类
# **彻底删掉** —— gradio 启动时会自检"能不能连上自己的 localhost"，代理在场时这个
# 自检会失败，它就直接拒绝启动（ValueError: When localhost is not accessible…）。
$env:NO_PROXY = '127.0.0.1,localhost,::1'
$env:no_proxy = $env:NO_PROXY
foreach ($proxyVar in 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy') {
    Remove-Item "Env:$proxyVar" -ErrorAction SilentlyContinue
}
$env:PYTHONIOENCODING = 'utf-8'

Write-Step '显存现状（训练 s2 约 6~9GB，和对话模型同时跑会紧张）'
try { & nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader } catch { }
Write-Host '    提示：先 powershell -ExecutionPolicy Bypass -File scripts\stop.ps1 腾出显存更稳。' -ForegroundColor DarkGray

Write-Host ''
Write-Host '  即将打开（页面会自动在浏览器里弹出）：' -ForegroundColor Green
Write-Host "    主界面（训练）  http://127.0.0.1:$portMain"
Write-Host "    推理页面        http://127.0.0.1:9872"
Write-Host "    UVR5 人声分离   http://127.0.0.1:9873"
Write-Host ''
Write-Host '  注意：上游把页面绑在 0.0.0.0（局域网可见）。训练完按 Ctrl+C 关掉即可。' -ForegroundColor DarkGray
Write-Host '  这个窗口不要关：它就是训练进程本身。' -ForegroundColor DarkGray
Write-Host ''

Push-Location $repo
try {
    # 不直接跑上游 webui.py：本环境的 starlette 1.6 改了 TemplateResponse 的签名，
    # 而 gradio 4.44 还在用旧签名 —— 结果是端口在监听、页面却 500 打不开。
    # 这个 shim 只在当前进程里补一层兼容，主程序与依赖版本都不动。
    $shim = Join-Path $ProjectRoot 'tools\gsv_webui_shim.py'
    if (Test-Path -LiteralPath $shim) {
        & $py $shim $Language
    } else {
        Write-Warn2 '没找到兼容层 tools\gsv_webui_shim.py，直接启动上游页面（可能 500）'
        & $py $webui $Language
    }
} finally {
    Pop-Location
}
