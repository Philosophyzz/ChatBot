# 安装 GPT-SoVITS（可训练的 TTS：微调自己的音色）
#
#   powershell -ExecutionPolicy Bypass -File scripts\install-gptsovits.ps1 -Plan
#   powershell -ExecutionPolicy Bypass -File scripts\install-gptsovits.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\install-gptsovits.ps1 -Proxy http://127.0.0.1:7890
#
# 为什么选 GPT-SoVITS 做"可训练"：
#   * 微调生态最成熟（1 分钟少样本能出声，10~60 分钟数据能明显更像本人）
#   * 中文/日文/英文/粤语，Windows + CUDA 有官方 PowerShell 安装脚本
#   * 推理很便宜（官方 README 实测 4060Ti RTF 0.028），适合常驻
#   * 对比：IndexTTS2 是零样本克隆，不用训练，但相似度上限取决于那 5~15 秒参考音频
#
# 环境隔离：同样单独建 conda 环境 chatbot-tts-train（包都在 D 盘），
# 训练/推理都在它里面跑，主程序环境（chatbot）保持干净。
[CmdletBinding()]
param(
    [string]$Root,
    [switch]$Plan,
    [switch]$SkipTorch,
    [switch]$SkipWeights,
    [switch]$NoUVR5,
    [switch]$NoMirror,
    [string]$Proxy = '',
    [string]$EnvName = 'chatbot',
    [string]$RepoUrl = 'https://github.com/RVC-Boss/GPT-SoVITS.git',
    [string]$RepoDir = '',
    [string]$Source = 'HF-Mirror',
    [string]$Device = 'CU128',
    [string]$NumpyPin = '2.2.6'
)

. (Join-Path $PSScriptRoot 'common.ps1')
$ProjectRoot = Get-ConfigRoot -Root $Root
if (-not $RepoDir) { $RepoDir = Join-Path $ProjectRoot 'vendor\GPT-SoVITS' }

$WeightsDir = Join-Path $RepoDir 'GPT_SoVITS\pretrained_models'
$G2PWZip = Join-Path $ProjectRoot 'models\tts\G2PWModel.zip'
$G2PWTarget = Join-Path $RepoDir 'GPT_SoVITS\text\G2PWModel'
$TrainEnvPython = Join-Path $ProjectRoot "vendor\gptsovits-python.txt"

Write-Host ''
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ' 安装 GPT-SoVITS（可微调的本地 TTS）' -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ' 用途：拿你自己的录音微调出专属音色；训练脚本见 scripts\train-tts.ps1'
Write-Host ' 对比：IndexTTS2 是零样本克隆（不训练），本方案是微调（更像本人）'
Write-Host ''

if ($Plan) {
    Write-Step '安装计划（-Plan 只打印，不下载任何东西）'
    Write-Host "  1) 独立环境     conda create -n $EnvName python=3.10        （官方测试组合：3.10 + torch 2.5/2.7 + CUDA 12.4/12.8）"
    Write-Host "  2) 取源码       git clone $RepoUrl `"$RepoDir`""
    Write-Host "  3) 装依赖       pwsh -F install.ps1 --Device $Device --Source $Source     （官方脚本；失败则退回 pip 手工装）"
    Write-Host "  4) 预训练权重   hf download lj1995/GPT-SoVITS -> GPT_SoVITS\pretrained_models   （5.3GB，含 v2Pro 的 s2Gv2ProPlus.pth）"
    Write-Host "  5) 中文文本前端  G2PWModel.zip -> GPT_SoVITS\text\G2PWModel                     （中文必装）"
    if (-not $NoUVR5) { Write-Host '  6) 人声分离     UVR5 权重（可选，素材带背景音乐时才需要）' }
    Write-Host "  7) 自检         $EnvName 里 import GPT_SoVITS 相关依赖 + torch.cuda.is_available()"
    Write-Host ''
    Write-Host ' 装完怎么用：'
    Write-Host '   1. 准备 10~60 分钟干净人声（一个人说话、无背景音乐）'
    Write-Host '   2. 整理数据集：python tools\tts_dataset.py --input <素材目录> --out <数据集目录> --speaker my_voice'
    Write-Host '   3. 训练：      powershell -ExecutionPolicy Bypass -File scripts\train-tts.ps1 -DataDir <数据集目录> -Speaker my_voice'
    Write-Host '   4. 用人设接进来：voice.backend 写 gpt_sovits（或直接改 speech.tts_preference 排第一）'
    return
}

# --- 1. 独立环境 -------------------------------------------------------------
Write-Step "1/6 准备训练环境（conda 环境 $EnvName，Python 3.10）"
$envPython = Get-CondaPython -Name $EnvName -Root $ProjectRoot
if ($envPython) {
    Write-Ok "已存在：$envPython"
} else {
    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $conda) {
        foreach ($exe in @('D:\miniconda3\Scripts\conda.exe', 'D:\anaconda3\Scripts\conda.exe',
                           (Join-Path $env:USERPROFILE 'miniconda3\Scripts\conda.exe'))) {
            if (Test-Path -LiteralPath $exe) { $conda = Get-Item -LiteralPath $exe; break }
        }
    }
    if (-not $conda) { Write-Err2 '找不到 conda；GPT-SoVITS 需要 Python 3.10 环境，请先安装 miniconda'; exit 1 }
    Write-Host "    conda create -n $EnvName python=3.10" -ForegroundColor DarkGray
    & $conda.Source create -n $EnvName python=3.10 -y
    $envPython = Get-CondaPython -Name $EnvName -Root $ProjectRoot
    if (-not $envPython) { Write-Err2 'conda 环境创建失败'; exit 1 }
    Write-Ok "已创建 $envPython"
}

# --- 2. 源码 -----------------------------------------------------------------
Write-Step '2/6 获取 GPT-SoVITS 源码'
if (Test-Path -LiteralPath (Join-Path $RepoDir 'install.ps1')) {
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
        Write-Err2 "git clone 失败。国内网络请加 -Proxy http://127.0.0.1:7890，或手动下载压缩包解到 $RepoDir"
        exit 1
    }
    Write-Ok "已克隆到 $RepoDir"
}

# --- 3. 依赖 -----------------------------------------------------------------
Write-Step '3/6 安装依赖（torch + 官方 requirements）'
$installed = $false
$official = Join-Path $RepoDir 'install.ps1'
if (-not $SkipTorch) {
    $torchIndex = 'https://download.pytorch.org/whl/cu128'
    $proxyArgs = @()
    if ($Proxy) { $proxyArgs = @('--proxy', $Proxy) }
    Write-Host "    pip install torch torchaudio --index-url $torchIndex" -ForegroundColor DarkGray
    & $envPython -m pip @proxyArgs install torch torchaudio --index-url $torchIndex
    if ($LASTEXITCODE -ne 0) {
        Write-Warn2 "从 $torchIndex 装 torch 失败；国内可试 https://mirror.sjtu.edu.cn/pytorch-wheels/cu128"
        Write-Err2 'torch 装不上就没法训练/推理，已停止'
        exit 1
    }
    Write-Ok 'torch / torchaudio 安装完成'
}

$reqExtra = Join-Path $RepoDir 'extra-req.txt'
$reqMain = Join-Path $RepoDir 'requirements.txt'
if ((Test-Path -LiteralPath $reqExtra) -and (Test-Path -LiteralPath $reqMain)) {
    Write-Host '    按仓库的 requirements 安装其余依赖' -ForegroundColor DarkGray
    & $envPython -m pip @proxyArgs install -r $reqExtra --no-deps
    & $envPython -m pip @proxyArgs install -r $reqMain -i https://pypi.tuna.tsinghua.edu.cn/simple
    $installed = ($LASTEXITCODE -eq 0)
    # GPT-SoVITS 的清单写 numpy<2.0，而 IndexTTS2（indextts）钉死 numpy==2.2.6。
    # 同一环境里两者不可能同时满足，所以装完把 numpy 抬回 $NumpyPin 再验证；
    # 若某个扩展模块是按 numpy 1.x ABI 编译的，导入时才会报错——那时就该改用独立环境训练。
    if ($NumpyPin) {
        Write-Warn2 "把 numpy 恢复到 $NumpyPin（IndexTTS2 的要求），装完务必跑 tools\check_tts_stack.py 验证"
        & $envPython -m pip @proxyArgs install "numpy==$NumpyPin" -i https://pypi.tuna.tsinghua.edu.cn/simple
    }
} elseif (Test-Path -LiteralPath $official) {
    Write-Warn2 '仓库里没有 requirements（新版可能改了结构），改用官方 install.ps1'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $official --Device $Device --Source $Source
    $installed = ($LASTEXITCODE -eq 0)
}
if (-not $installed) {
    Write-Warn2 '依赖安装可能不完整；请进入仓库目录手动执行：pip install -r requirements.txt'
}

# --- 4. 预训练权重 -----------------------------------------------------------
Write-Step '4/6 下载预训练权重（lj1995/GPT-SoVITS，约 5.3GB）'
if ($SkipWeights) {
    Write-Warn2 '按参数跳过。之后手动执行：'
    Write-Host "       `$env:HF_ENDPOINT='https://hf-mirror.com'; hf download lj1995/GPT-SoVITS --local-dir `"$WeightsDir`"" -ForegroundColor DarkGray
} elseif (Test-Path -LiteralPath (Join-Path $WeightsDir 'v2Pro\s2Gv2ProPlus.pth')) {
    Write-Ok "权重已存在：$WeightsDir"
} else {
    if (-not $NoMirror) {
        $env:HF_ENDPOINT = 'https://hf-mirror.com'
        # 镜像不代理 Xet 存储后端（会 401），必须关掉，否则下载必定失败
        $env:HF_HUB_DISABLE_XET = '1'
    }
    New-Item -ItemType Directory -Path $WeightsDir -Force | Out-Null
    $hf = Get-Command hf -ErrorAction SilentlyContinue
    if ($hf) {
        & $hf.Source download 'lj1995/GPT-SoVITS' --local-dir $WeightsDir --max-workers 8
    } else {
        & $envPython -m pip install --quiet huggingface_hub
        & $envPython (Join-Path $ProjectRoot 'tools\download_gptsovits_weights.py') --dest $WeightsDir
    }
    if (-not (Test-Path -LiteralPath (Join-Path $WeightsDir 'v2Pro\s2Gv2ProPlus.pth'))) {
        Write-Err2 '权重没下齐（缺 v2Pro/s2Gv2ProPlus.pth）。可重跑本脚本，会断点续传。'
        exit 1
    }
    Write-Ok '预训练权重就绪（v2Pro / v4 / v2final 都下下来了，训练脚本按版本取用）'
}

# --- 5. 中文文本前端 + ffmpeg ------------------------------------------------
Write-Step '5/6 中文文本前端（G2PW）与 ffmpeg'
if (Test-Path -LiteralPath $G2PWTarget) {
    Write-Ok "G2PWModel 已存在：$G2PWTarget"
} else {
    if (-not $NoMirror) { $env:HF_ENDPOINT = 'https://hf-mirror.com'; $env:HF_HUB_DISABLE_XET = '1' }
    Write-Host '    下载 G2PWModel.zip（中文多音字，中文合成必装）' -ForegroundColor DarkGray
    $ok = $false
    $hf = Get-Command hf -ErrorAction SilentlyContinue
    if ($hf) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $G2PWZip) -Force | Out-Null
        & $hf.Source download 'XXXXRT/GPT-SoVITS-Pretrained' 'G2PWModel.zip' --local-dir (Split-Path -Parent $G2PWZip) 2>&1 | Out-Null
        $ok = Test-Path -LiteralPath $G2PWZip
    }
    if (-not $ok) {
        Write-Warn2 "G2PWModel.zip 没下到。手动下载后解压到 $G2PWTarget"
        Write-Host '       https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip' -ForegroundColor DarkGray
    } else {
        Expand-Archive -LiteralPath $G2PWZip -DestinationPath (Split-Path -Parent $G2PWTarget) -Force
        if (Test-Path -LiteralPath (Join-Path (Split-Path -Parent $G2PWTarget) 'G2PWModel')) {
            Write-Ok 'G2PWModel 解压完成'
        } else {
            Write-Warn2 "解压结果不在预期位置，请确认 $G2PWTarget 存在"
        }
    }
}
foreach ($tool in 'ffmpeg.exe', 'ffprobe.exe') {
    $target = Join-Path $RepoDir $tool
    if (Test-Path -LiteralPath $target) { Write-Ok "$tool 已就位"; continue }
    $hf = Get-Command hf -ErrorAction SilentlyContinue
    if ($hf) {
        if (-not $NoMirror) { $env:HF_ENDPOINT = 'https://hf-mirror.com'; $env:HF_HUB_DISABLE_XET = '1' }
        & $hf.Source download 'lj1995/VoiceConversionWebUI' $tool --local-dir $RepoDir 2>&1 | Out-Null
        if (Test-Path -LiteralPath $target) { Write-Ok "$tool 已下载（GPT-SoVITS 切片/转码要用）" }
        else { Write-Warn2 "$tool 没下到：素材切片会失败；也可以 winget install Gyan.FFmpeg" }
    } else {
        Write-Warn2 "缺少 $tool（GPT-SoVITS 要用）；winget install Gyan.FFmpeg"
    }
}

# --- 6. 自检 -----------------------------------------------------------------
Write-Step '6/6 自检'
$check = & $envPython -c "import torch, sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'python', sys.version.split()[0])" 2>&1
Write-Host "    $check"
if ($check -notmatch 'cuda True') {
    Write-Warn2 'torch 看不到 CUDA：训练会退到 CPU（非常慢）。检查显卡驱动，或重装 CUDA 版 torch。'
}
& $envPython (Join-Path $ProjectRoot 'tools\check_gptsovits.py') --repo $RepoDir
if ($LASTEXITCODE -ne 0) { Write-Warn2 '仓库自检有缺失项，见上面输出' }

Write-Host ''
Write-Host '============================================================' -ForegroundColor Green
Write-Host ' 安装完成。下一步：' -ForegroundColor Green
Write-Host '   1. 找 10~60 分钟干净人声（一个人、无背景音乐、无混响）'
Write-Host '   2. 整理数据集：'
Write-Host '      python tools\tts_dataset.py --input <素材目录> --out <数据集目录> --speaker my_voice'
Write-Host '   3. 训练：'
Write-Host '      powershell -ExecutionPolicy Bypass -File scripts\train-tts.ps1 -DataDir <数据集目录> -Speaker my_voice'
Write-Host '   4. 接进人设：config\personas.yaml 里 voice.backend: gpt_sovits'
Write-Host '============================================================' -ForegroundColor Green
