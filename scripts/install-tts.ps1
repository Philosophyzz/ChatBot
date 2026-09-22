# 安装本地语音合成 IndexTTS2（独立环境，不污染主环境）
#
# 为什么要独立环境：indextts 这个包把 torch==2.8 / numpy==2.2.6 / keras==2.9 /
# opencv / transformers==4.52.1 / modelscope 一起钉死。直接装进主环境会把 numpy 从
# 2.4 降到 2.2，而主程序正在用 onnxruntime / ctranslate2 —— 为一个可选功能弄坏一个
# 能跑的程序不划算。所以：模型装在 venvs\tts，主程序通过 tools\tts_worker.py 这个
# JSON 子进程桥调用它（logs\tts-worker.log 里有它的输出）。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\install-tts.ps1 -Plan      # 只看计划，不下载
#   powershell -ExecutionPolicy Bypass -File scripts\install-tts.ps1            # 完整安装（约 12GB 下载）
#   powershell -ExecutionPolicy Bypass -File scripts\install-tts.ps1 -Proxy http://127.0.0.1:7890
#   powershell -ExecutionPolicy Bypass -File scripts\install-tts.ps1 -SkipWeights  # 只装环境，权重自己下
[CmdletBinding()]
param(
    [string]$Root,
    [switch]$Plan,
    [switch]$SkipTorch,
    [switch]$SkipWeights,
    [switch]$NoMirror,
    [string]$Proxy = '',
    [string]$EnvName = 'chatbot',
    [string]$TorchIndex = 'https://download.pytorch.org/whl/cu128',
    [string]$PyPiIndex = 'https://pypi.tuna.tsinghua.edu.cn/simple',
    [string]$RepoUrl = 'https://github.com/index-tts/index-tts.git',
    [string]$RepoDir = ''
)

. (Join-Path $PSScriptRoot 'common.ps1')
$ProjectRoot = Get-ConfigRoot -Root $Root
if (-not $RepoDir) { $RepoDir = Join-Path $ProjectRoot 'vendor\index-tts' }

# 与主程序同一套约定：优先 conda 环境（包都在 D 盘），没有 conda 才退回 venvs\tts。
# 两个位置 src\speech\__init__.py 都会自动发现，所以装在哪边都不用改配置。
$condaPython = Get-CondaPython -Name $EnvName -Root $ProjectRoot
$TtsVenv = Join-Path $ProjectRoot 'venvs\tts'
$TtsPy = if ($condaPython) { $condaPython } else { Join-Path $TtsVenv 'python.exe' }
$TtsLabel = if ($condaPython) { "conda 环境 $EnvName" } else { 'venvs\tts' }
$WeightsDir = Join-Path $ProjectRoot 'models\tts\IndexTTS2'
$VoiceDir = Join-Path $ProjectRoot 'models\voices'
$Worker = Join-Path $ProjectRoot 'tools\tts_worker.py'
$MakeRef = Join-Path $ProjectRoot 'tools\make_reference.py'

Write-Host ''
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host " 安装本地语音合成 IndexTTS2（独立环境：$TtsLabel）" -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ' 说明：IndexTTS2 是零样本音色克隆，必须有 5~15 秒参考音频；'
Write-Host '       它让「甜美音色 + 情感控制 + 多个人设各用不同音色」成为可能。'
Write-Host ''

if ($Plan) {
    Write-Step '安装计划（-Plan 只打印，不执行任何下载）'
    Write-Host "  1) 独立环境          conda create -n $EnvName python=3.11（没有 conda 则 uv venv `"$TtsVenv`"）"
    Write-Host "  2) 装 CUDA 版 torch  pip install torch==2.8.* torchaudio==2.8.* --index-url $TorchIndex   （约 3GB）"
    Write-Host "  3) 取源码            git clone $RepoUrl `"$RepoDir`""
    Write-Host "  4) 装 indextts       pip install -e `"$RepoDir`" -i $PyPiIndex                                （约 2GB）"
    Write-Host "  5) 下权重            hf download IndexTeam/IndexTTS-2 --local-dir `"$WeightsDir`"              （5.9GB）"
    Write-Host "  6) 参考音频          $VoiceDir\*.wav（已按人设生成；可换成你自己的录音）"
    Write-Host "  7) 自检              $TtsPy -c `"import indextts, torch`""
    Write-Host ''
    Write-Host ' 完成后：scripts\stop.ps1 然后 scripts\start-all.ps1，'
    Write-Host '         打开 http://127.0.0.1:8077 的「系统」页看「语音合成」。'
    return
}

# --- 1. 独立环境 -------------------------------------------------------------
Write-Step "1/6 准备独立 Python 环境（$TtsLabel）"
if (Test-Path -LiteralPath $TtsPy) {
    Write-Ok "已存在：$TtsPy"
} else {
    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $conda) {
        foreach ($exe in @('D:\miniconda3\Scripts\conda.exe', 'D:\anaconda3\Scripts\conda.exe',
                           (Join-Path $env:USERPROFILE 'miniconda3\Scripts\conda.exe'))) {
            if (Test-Path -LiteralPath $exe) { $conda = Get-Item -LiteralPath $exe; break }
        }
    }
    if ($conda) {
        Write-Host "    conda create -n $EnvName python=3.11" -ForegroundColor DarkGray
        & $conda.Source create -n $EnvName python=3.11 -y
        $TtsPy = (Get-CondaPython -Name $EnvName -Root $ProjectRoot)
        if (-not $TtsPy) { Write-Err2 'conda 环境创建后仍找不到 python'; exit 1 }
        $TtsLabel = "conda 环境 $EnvName"
        Write-Ok "已创建 $TtsPy"
    } else {
        $uv = Get-Command uv -ErrorAction SilentlyContinue
        if ($uv) {
            & $uv.Source venv --python 3.11 --seed $TtsVenv
            if ($LASTEXITCODE -ne 0) { Write-Err2 'uv venv 建环境失败'; exit 1 }
        } else {
            $base = Get-SystemPython
            if (-not $base) { Write-Err2 '找不到 Python，请先跑 scripts\install.ps1'; exit 1 }
            & $base -m venv $TtsVenv
            if ($LASTEXITCODE -ne 0) { Write-Err2 'python -m venv 建环境失败'; exit 1 }
        }
        Write-Ok "已创建 $TtsVenv"
    }
}

# conda 环境里 pip 可能不是最新的；先升级一次，避免解析依赖时踩旧版坑
& $TtsPy -m pip install --upgrade pip --quiet 2>&1 | Out-Null

function Install-Pip {
    param([string[]]$PipArgs, [string]$Label)
    Write-Host "     $Label" -ForegroundColor DarkGray
    $common = @()
    if ($Proxy) { $common += @('--proxy', $Proxy) }
    & $TtsPy -m pip @common @PipArgs
    return $LASTEXITCODE
}

# --- 2. torch（CUDA） --------------------------------------------------------
Write-Step '2/6 安装 CUDA 版 PyTorch（index-tts 要求 2.8.*）'
if ($SkipTorch) {
    Write-Warn2 '按参数跳过'
} else {
    $code = Install-Pip -Label "pip install torch==2.8.* torchaudio==2.8.* --index-url $TorchIndex" `
        -PipArgs @('install', '--upgrade', 'torch==2.8.*', 'torchaudio==2.8.*', '--index-url', $TorchIndex)
    if ($code -ne 0) {
        Write-Warn2 "从 $TorchIndex 安装失败。国内可换镜像重试，例如："
        Write-Host '       -TorchIndex https://mirror.sjtu.edu.cn/pytorch-wheels/cu128' -ForegroundColor DarkGray
        Write-Err2 'torch 安装失败，已停止（其余步骤依赖它）'
        exit 1
    }
    Write-Ok 'torch 安装完成'
}

# --- 3. 源码 -----------------------------------------------------------------
Write-Step '3/6 获取 index-tts 源码（该包未发布到 PyPI，只能从仓库装）'
if (Test-Path -LiteralPath (Join-Path $RepoDir 'pyproject.toml')) {
    Write-Ok "已存在：$RepoDir"
} else {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if (-not $git) { Write-Err2 '找不到 git，请先安装 Git for Windows'; exit 1 }
    $parent = Split-Path -Parent $RepoDir
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    $gitArgs = @()
    if ($Proxy) { $gitArgs += @('-c', "http.proxy=$Proxy", '-c', "https.proxy=$Proxy") }
    & $git.Source @gitArgs clone --depth 1 $RepoUrl $RepoDir
    if ($LASTEXITCODE -ne 0) {
        Write-Err2 'git clone 失败。若只是网络问题，加 -Proxy http://127.0.0.1:7890 再试。'
        exit 1
    }
    Write-Ok "已克隆到 $RepoDir"
}

# --- 4. indextts -------------------------------------------------------------
Write-Step '4/6 安装 indextts 及其依赖'
$code = Install-Pip -Label "pip install -e $RepoDir -i $PyPiIndex" `
    -PipArgs @('install', '-e', $RepoDir, '-i', $PyPiIndex)
if ($code -ne 0) { Write-Err2 'indextts 安装失败'; exit 1 }
Write-Ok 'indextts 安装完成'

# --- 5. 权重 -----------------------------------------------------------------
Write-Step '5/6 下载模型权重（IndexTeam/IndexTTS-2，约 5.9GB）'
if ($SkipWeights) {
    Write-Warn2 '按参数跳过；之后可手动执行：'
    Write-Host "       hf download IndexTeam/IndexTTS-2 --local-dir `"$WeightsDir`"" -ForegroundColor DarkGray
} elseif (Test-Path -LiteralPath (Join-Path $WeightsDir 'gpt.pth')) {
    Write-Ok "权重已存在：$WeightsDir"
} else {
    $hf = Get-Command hf -ErrorAction SilentlyContinue
    if (-not $hf) { $hf = Get-Command huggingface-cli -ErrorAction SilentlyContinue }
    if (-not $hf) {
        Write-Warn2 '找不到 hf 命令，改为用 TTS 环境里的 huggingface_hub 下载'
        $env:HF_ENDPOINT = if ($NoMirror) { 'https://huggingface.co' } else { 'https://hf-mirror.com' }
        & $TtsPy -m pip install --quiet huggingface_hub
        & $TtsPy (Join-Path $ProjectRoot 'tools\download_tts_weights.py') --dest $WeightsDir
    } else {
        if (-not $NoMirror) { $env:HF_ENDPOINT = 'https://hf-mirror.com' }
        New-Item -ItemType Directory -Path $WeightsDir -Force | Out-Null
        & $hf.Source download 'IndexTeam/IndexTTS-2' --local-dir $WeightsDir
    }
    if (-not (Test-Path -LiteralPath (Join-Path $WeightsDir 'gpt.pth'))) {
        Write-Err2 '权重没下齐（缺少 gpt.pth）。可以重跑本脚本，hf 会断点续传。'
        exit 1
    }
    Write-Ok '权重下载完成'
}

# --- 6. 参考音频 + 自检 ------------------------------------------------------
Write-Step '6/6 参考音频与自检'
if (-not (Test-Path -LiteralPath $VoiceDir)) { New-Item -ItemType Directory -Path $VoiceDir -Force | Out-Null }
$refVoice = Join-Path $VoiceDir 'default_sweet.wav'
if (Test-Path -LiteralPath $refVoice) {
    Write-Ok "参考音频已存在：$refVoice"
} else {
    Write-Warn2 '没有参考音频。IndexTTS2 是克隆模型，必须有一段音色样本。'
    if (Test-Path -LiteralPath $MakeRef) {
        & $TtsPy $MakeRef --out $refVoice
        if (Test-Path -LiteralPath $refVoice) {
            Write-Ok "已生成占位参考音频：$refVoice"
            Write-Warn2 '建议换成你自己的 5~15 秒录音（同一个文件路径），克隆出来的音色就是你的。'
        } else {
            Write-Warn2 '生成失败：请手动放一段 5~15 秒中文 WAV 到上面的路径。'
        }
    }
}

Write-Host ''
Write-Host '  自检：' -NoNewline
& $TtsPy -c "import indextts, torch; print('indextts ok, cuda =', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    Write-Err2 'TTS 环境自检失败：indextts 无法导入'
    exit 1
}

Write-Host ''
Write-Host '============================================================' -ForegroundColor Green
Write-Host ' 安装完成。接下来：' -ForegroundColor Green
Write-Host '   powershell -ExecutionPolicy Bypass -File scripts\stop.ps1'
Write-Host '   powershell -ExecutionPolicy Bypass -File scripts\start-all.ps1'
Write-Host ' 然后打开 http://127.0.0.1:8077 ，在「系统」页确认「语音合成」里'
Write-Host ' indextts 显示正常；第一次说话会加载模型（约 20~40 秒），之后就快了。'
Write-Host ' 显存紧张时它会空闲自动卸载（config.yaml: tts.indextts.idle_unload_s）。'
Write-Host '============================================================' -ForegroundColor Green
Show-Vram
