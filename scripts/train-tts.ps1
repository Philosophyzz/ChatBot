# 微调 GPT-SoVITS：把整理好的数据集训练成你的音色
#
#   # 1) 只看计划（不训练）：确认脚本路径、参数、显存预估
#   powershell -ExecutionPolicy Bypass -File scripts\train-tts.ps1 -DataDir data\my_voice -Speaker my_voice -DryRun
#
#   # 2) 真训练（s2 声学模型 -> s1 GPT 模型）
#   powershell -ExecutionPolicy Bypass -File scripts\train-tts.ps1 -DataDir data\my_voice -Speaker my_voice
#
#   # 3) 训练完启动推理服务（人设里 voice.backend: gpt_sovits 就能用）
#   powershell -ExecutionPolicy Bypass -File scripts\train-tts.ps1 -Serve
#
# 数据集从哪来：python tools\tts_dataset.py --input <素材目录> --out <数据集目录> --speaker my_voice
#   （切片 + 静音裁剪 + 峰值归一化 + Whisper 转写 + 生成 dataset.list）
#
# 关于版本：默认微调 **v2ProPlus**。官方 README 明确说 v3/v4 需要较高质量的训练音频，
# 而 v1/v2/v2Pro 对普通素材更宽容；v2Pro 的质量与 v4 相当，显存与速度按 v2 走。
[CmdletBinding()]
param(
    [string]$Root,
    [string]$DataDir = '',
    [string]$Speaker = 'my_voice',
    [string]$RepoDir = '',
    [string]$EnvName = 'chatbot',
    [ValidateSet('v2ProPlus', 'v2Pro', 'v2', 'v4')]
    [string]$Version = 'v2ProPlus',
    [int]$SoVitsEpochs = 8,
    [int]$SoVitsBatch = 6,
    [int]$GptEpochs = 15,
    [int]$GptBatch = 6,
    [switch]$DryRun,
    [switch]$SkipS2,
    [switch]$SkipS1,
    [switch]$Serve,
    [int]$ServePort = 9090
)

. (Join-Path $PSScriptRoot 'common.ps1')
$ProjectRoot = Get-ConfigRoot -Root $Root
if (-not $RepoDir) { $RepoDir = Join-Path $ProjectRoot 'vendor\GPT-SoVITS' }

function Get-EnvPython {
    $py = Get-CondaPython -Name $EnvName -Root $ProjectRoot
    if (-not $py) {
        Write-Err2 "找不到 conda 环境 $EnvName（先跑 scripts\install-gptsovits.ps1）"
        exit 1
    }
    return $py
}

if (-not (Test-Path -LiteralPath $RepoDir)) {
    Write-Err2 "找不到 GPT-SoVITS 仓库：$RepoDir（先跑 scripts\install-gptsovits.ps1）"
    exit 1
}
$envPython = Get-EnvPython

# --- 只启动推理服务 ------------------------------------------------------------
if ($Serve) {
    Write-Step "启动 GPT-SoVITS 推理服务（端口 $ServePort）"
    $api = Join-Path $RepoDir 'api_v2.py'
    if (-not (Test-Path -LiteralPath $api)) { Write-Err2 "缺少 $api"; exit 1 }
    $log = Join-Path $ProjectRoot 'logs\gptsovits-api.log'
    $errLog = Join-Path $ProjectRoot 'logs\gptsovits-api.err.log'
    Write-Host "    日志：$log"
    Write-Host '    服务起来后，人设里把 voice.backend 设为 gpt_sovits 即可（默认连 127.0.0.1:9090）'
    & $envPython $api -a 127.0.0.1 -p $ServePort -c (Join-Path $RepoDir 'GPT_SoVITS\configs\tts_infer.yaml') 2>&1 |
        Tee-Object -FilePath $log
    exit $LASTEXITCODE
}

# --- 训练前置检查 --------------------------------------------------------------
if (-not $DataDir) { Write-Err2 '需要 -DataDir（数据集目录，含 dataset.list 与 sliced\）'; exit 1 }
$DataDir = (Resolve-Path -LiteralPath $DataDir).Path
$listFile = Join-Path $DataDir 'dataset.list'
$slicedDir = Join-Path $DataDir 'sliced'
if (-not (Test-Path -LiteralPath $listFile)) {
    Write-Err2 "数据集里没有 dataset.list：$listFile"
    Write-Host '  先运行：python tools\tts_dataset.py --input <素材目录> --out <数据集目录> --speaker my_voice' -ForegroundColor DarkGray
    exit 1
}
$lines = @(Get-Content -LiteralPath $listFile -Encoding UTF8 | Where-Object { $_.Trim() })
if ($lines.Count -lt 4) { Write-Err2 "dataset.list 只有 $($lines.Count) 行，数据太少（至少几十条切片再训练）"; exit 1 }
$totalSeconds = 0.0
$badLines = 0
foreach ($line in $lines) {
    $parts = $line.Split('|')
    if ($parts.Count -lt 4 -or -not $parts[3].Trim()) { $badLines++ }
}
Write-Host ''
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host " GPT-SoVITS 微调：$Speaker（$Version）" -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Ok "数据集 $($lines.Count) 条切片：$DataDir"
if ($badLines -gt 0) { Write-Warn2 "$badLines 行缺少文本（第 4 列），这些会被训练忽略；建议先补全标注" }

# --- 找到训练脚本与配置模板 ----------------------------------------------------
function Find-First {
    param([string[]]$Patterns, [string]$Label)
    foreach ($pattern in $Patterns) {
        $found = Get-ChildItem -LiteralPath $RepoDir -Recurse -Filter $pattern -File -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($found) { return $found.FullName }
    }
    Write-Err2 "找不到 $Label（在 $RepoDir 里试过：$($Patterns -join ', ')）"
    Write-Host '  说明：上游仓库改名/移动过脚本。请打开仓库目录确认实际文件名，或先用它的 WebUI 训练一次。' -ForegroundColor DarkGray
    return $null
}

$s2Train = Find-First -Patterns @('s2_train.py') -Label 'SoVITS 训练脚本'
$s1Train = Find-First -Patterns @('s1_train.py') -Label 'GPT 训练脚本'
$prepScripts = @()
foreach ($name in '1-get-text.py', '2-get-hubert-wav32k.py', '3-get-semantic.py') {
    $script = Get-ChildItem -LiteralPath $RepoDir -Recurse -Filter $name -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($script) { $prepScripts += $script.FullName } else { Write-Warn2 "找不到预处理脚本 $name（少样本也可跳过，但语义特征会缺失）" }
}

# 版本 -> 预训练权重
$pretrained = Join-Path $RepoDir 'GPT_SoVITS\pretrained_models'
$s2G = switch ($Version) {
    'v2ProPlus' { Join-Path $pretrained 'v2Pro\s2Gv2ProPlus.pth' }
    'v2Pro'     { Join-Path $pretrained 'v2Pro\s2Gv2Pro.pth' }
    'v2'        { Join-Path $pretrained 'gsv-v2final-pretrained\s2G2333k.pth' }
    'v4'        { Join-Path $pretrained 'gsv-v4-pretrained\s2Gv4.pth' }
}
$s2D = switch ($Version) {
    'v2ProPlus' { Join-Path $pretrained 'v2Pro\s2Dv2ProPlus.pth' }
    'v2Pro'     { Join-Path $pretrained 'v2Pro\s2Dv2Pro.pth' }
    'v2'        { Join-Path $pretrained 'gsv-v2final-pretrained\s2D2333k.pth' }
    'v4'        { Join-Path $pretrained 'gsv-v4-pretrained\s2Dv2ProPlus.pth' }
}
$s1Ckpt = Join-Path $pretrained 'gsv-v2final-pretrained\s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt'
if ($Version -eq 'v4') { $s1Ckpt = Join-Path $pretrained 's1v3.ckpt' }

$missingWeights = @()
foreach ($item in @(
        @{ Path = $s2G; Label = "生成器 $Version $s2G" },
        @{ Path = $s2D; Label = "判别器 $Version $s2D" },
        @{ Path = $s1Ckpt; Label = 'GPT(s1) 预训练' },
        @{ Path = (Join-Path $pretrained 'chinese-hubert-base\pytorch_model.bin'); Label = 'HuBERT' },
        @{ Path = (Join-Path $pretrained 'chinese-roberta-wwm-ext-large\pytorch_model.bin'); Label = 'RoBERTa' })) {
    if (-not (Test-Path -LiteralPath $item.Path)) { $missingWeights += $item.Label }
}
if ($missingWeights.Count -gt 0) {
    Write-Err2 "缺少预训练权重：$($missingWeights -join '；')"
    Write-Host '  补齐：powershell -ExecutionPolicy Bypass -File scripts\install-gptsovits.ps1' -ForegroundColor DarkGray
    exit 1
}

# --- 生成训练配置（沿用仓库模板，只改路径与超参） ------------------------------
$runDir = Join-Path $DataDir 'gptsovits'
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
$s2Config = Join-Path $runDir "s2_$Version.json"
$s1Config = Join-Path $runDir "s1_$Version.yaml"

$s2Template = Get-ChildItem -LiteralPath (Join-Path $RepoDir 'GPT_SoVITS\configs') -Filter 's2*.json' -File -ErrorAction SilentlyContinue | Select-Object -First 1
$s1Template = Get-ChildItem -LiteralPath (Join-Path $RepoDir 'GPT_SoVITS\configs') -Filter 's1*.yaml' -File -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $s2Template -or -not $s1Template) {
    Write-Err2 '找不到训练配置模板（GPT_SoVITS\configs\s2*.json / s1*.yaml）'
    exit 1
}

if (-not $DryRun) {
    $python = @'
import json, re, sys, pathlib
s2_src, s1_src, data_dir, speaker, version = sys.argv[1:6]
s2_out, s1_out = sys.argv[6:8]
sovits_epochs, sovits_batch, gpt_epochs, gpt_batch = (int(v) for v in sys.argv[8:12])
s2_g, s2_d, s1_ckpt, hubert, roberta = sys.argv[12:17]

data = pathlib.Path(data_dir)
s2 = json.loads(pathlib.Path(s2_src).read_text(encoding="utf-8"))
s2.update({
    "train": {
        "log_interval": 50,
        "eval_interval": 200,
        "seed": 1234,
        "epochs": sovits_epochs,
        "learning_rate": 0.0001,
        "lr_decay": 0.999875,
        "save_every_epoch": max(1, sovits_epochs // 2),
        "if_save_latest": True,
        "if_cache_data_in_gpu": False,
        "batch_size": sovits_batch,
    },
    "data": {
        "max_wav_value": 32768.0,
        "sampling_rate": 32000,
        "filter_length": 2048,
        "hop_length": 640,
        "win_length": 2048,
        "n_mel_channels": 128,
        "mel_fmin": 0.0,
        "mel_fmax": None,
        "add_blank": True,
        "n_speakers": 1,
        "cleaned_text": True,
        "exp_dir": str(data / "logs" / speaker),
    },
    "model": {
        "inter_channels": 192,
        "hidden_channels": 192,
        "filter_channels": 768,
        "n_heads": 2,
        "n_layers": 6,
        "kernel_size": 3,
        "p_dropout": 0.1,
        "resblock": "1",
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_rates": [10, 8, 2, 2, 2],
        "upsample_initial_channel": 512,
        "upsample_kernel_sizes": [16, 16, 8, 4, 4],
        "n_layers_q": 3,
        "use_spectral_norm": False,
        "gin_channels": 512,
        "semantic_frame_rate": "25hz",
        "freeze_quantizer": True,
    },
    "s2_ckpt_dir": str(data / "logs" / speaker),
    "s2G_path": s2_g,
    "s2D_path": s2_d,
})
pathlib.Path(s2_out).write_text(json.dumps(s2, ensure_ascii=False, indent=2), encoding="utf-8")

s1_text = pathlib.Path(s1_src).read_text(encoding="utf-8")
s1 = {
    "train": {
        "seed": 1234,
        "epochs": gpt_epochs,
        "batch_size": gpt_batch,
        "learning_rate": 0.01,
        "save_every_n_epoch": max(1, gpt_epochs // 3),
        "precision": "16-mixed",
        "if_save_latest": True,
        "if_save_every_weights": True,
        "half_weights_save_dir": str(data / "logs" / speaker / "weights"),
        "exp_name": speaker,
    },
    "pretrained": {"s1_ckpt_dir": s1_ckpt},
    "data": {
        "max_eval_sample": 8,
        "max_sec": 54,
        "num_workers": 1,
        "pad_val": 0.0,
    },
    "s1_ckpt_dir": str(data / "logs" / speaker),
    "output_dir": str(data / "logs" / speaker),
    "bert_pretrained_dir": roberta,
    "hubert_path": hubert,
}
# 模板里已有的键优先保留，避免上游新增字段被我们覆盖掉
if s1_text.strip():
    try:
        import yaml
        template = yaml.safe_load(s1_text) or {}
        for section, values in template.items():
            if isinstance(values, dict):
                merged = dict(values)
                merged.update(s1.get(section, {}))
                s1[section] = merged
            else:
                s1.setdefault(section, values)
    except Exception as exc:
        print(f"  [warn] 读取 s1 模板失败，使用内置配置：{exc}", file=sys.stderr)
import yaml
pathlib.Path(s1_out).write_text(yaml.safe_dump(s1, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(f"  配置已生成：{s2_out} / {s1_out}")
'@
    $pythonFile = Join-Path $runDir 'make_configs.py'
    $python | Set-Content -LiteralPath $pythonFile -Encoding UTF8
    & $envPython $pythonFile $s2Template.FullName $s1Template.FullName $DataDir $Speaker $Version `
        $s2Config $s1Config $SoVitsEpochs $SoVitsBatch $GptEpochs $GptBatch `
        $s2G $s2D $s1Ckpt (Join-Path $pretrained 'chinese-hubert-base') (Join-Path $pretrained 'chinese-roberta-wwm-ext-large')
    if ($LASTEXITCODE -ne 0) { Write-Err2 '生成训练配置失败'; exit 1 }
}

# --- 计划输出 -----------------------------------------------------------------
Write-Step '训练计划'
Write-Host "  环境      : $envPython"
Write-Host "  数据集    : $DataDir（$($lines.Count) 条）"
Write-Host "  版本      : $Version"
Write-Host "  SoVITS(s2): epochs=$SoVitsEpochs batch=$SoVitsBatch  -> $s2Config"
Write-Host "  GPT(s1)   : epochs=$GptEpochs batch=$GptBatch        -> $s1Config"
Write-Host "  输出      : $(Join-Path $DataDir 'logs')"
Write-Host "  显存预估  : s2 约 6~9GB / s1 约 5~8GB（16GB 卡足够；与对话模型同时跑会紧张）"
Write-Host "  时间预估  : 10 分钟素材 ≈ 20~60 分钟（取决于 batch 与显卡；以实际日志为准）"
Write-Host "  预处理    : $($prepScripts.Count) 个脚本将被调用（文本/HuBERT/语义特征）"

if ($DryRun) {
    Write-Host ''
    Write-Host '  -DryRun：以上只是计划，没有执行任何训练。' -ForegroundColor Yellow
    Write-Host '  去掉 -DryRun 即开始训练。' -ForegroundColor Yellow
    if ($prepScripts.Count -gt 0) {
        Write-Host '  将要执行：'
        foreach ($script in $prepScripts) { Write-Host "    $envPython $script" -ForegroundColor DarkGray }
    }
    Write-Host "    $envPython $s2Train --config $s2Config" -ForegroundColor DarkGray
    Write-Host "    $envPython $s1Train --config_file $s1Config" -ForegroundColor DarkGray
    exit 0
}

# --- 预处理（文本/HuBERT/语义特征） -------------------------------------------
$logDir = Join-Path $DataDir 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$trainLog = Join-Path $ProjectRoot 'logs\tts-train.log'

function Invoke-Step {
    param([string]$Label, [string]$Exe, [string[]]$Arguments)
    Write-Step $Label
    $stamp = Get-Date -Format 'HH:mm:ss'
    Add-Content -LiteralPath $trainLog -Encoding UTF8 -Value "`n===== [$stamp] $Label ====="
    & $Exe @Arguments 2>&1 | Tee-Object -FilePath $trainLog -Append
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Err2 "$Label 失败（exit $code）。完整日志：$trainLog"
        return $false
    }
    return $true
}

if ($prepScripts.Count -eq 3 -and $Version -ne 'v4') {
    $prepOk = $true
    foreach ($script in $prepScripts) {
        $prepOk = Invoke-Step -Label "预处理 $(Split-Path -Leaf $script)" -Exe $envPython -Arguments @($script)
        if (-not $prepOk) { break }
    }
    if (-not $prepOk) {
        Write-Warn2 '预处理失败：可以先在 GPT-SoVITS 的 WebUI 里跑一次「1A 训练集格式化」，确认它的路径约定后再回来'
    }
} else {
    Write-Warn2 '跳过了自动预处理（脚本不全或 v4）：若训练脚本报缺 2-name2text.txt，请先在 WebUI 跑 1A'
}

if (-not $SkipS2) {
    Invoke-Step -Label '训练 SoVITS(s2)' -Exe $envPython -Arguments @($s2Train, '--config', $s2Config) | Out-Null
}
if (-not $SkipS1) {
    Invoke-Step -Label '训练 GPT(s1)' -Exe $envPython -Arguments @($s1Train, '--config_file', $s1Config) | Out-Null
}

# --- 收集产物 -----------------------------------------------------------------
Write-Step '收集训练产物'
$weights = Get-ChildItem -LiteralPath $logDir -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '\.(pth|ckpt)$' }
if ($weights) {
    $voiceDir = Join-Path $ProjectRoot "models\tts\$Speaker"
    New-Item -ItemType Directory -Path $voiceDir -Force | Out-Null
    foreach ($file in $weights) {
        Copy-Item -LiteralPath $file.FullName -Destination $voiceDir -Force
        Write-Ok "$($file.Name)  ($([math]::Round($file.Length/1MB,1))MB)"
    }
    Write-Host "  产物已复制到：$voiceDir" -ForegroundColor Green
} else {
    Write-Warn2 "没找到训练产物（*.pth / *.ckpt），请看日志：$trainLog"
}

Write-Host ''
Write-Host '============================================================' -ForegroundColor Green
Write-Host " $Speaker 训练流程结束" -ForegroundColor Green
Write-Host ' 接下来：'
Write-Host "   1) 启动推理服务：powershell -ExecutionPolicy Bypass -File scripts\train-tts.ps1 -Serve"
Write-Host "   2) 人设里指定后端：voice.backend: gpt_sovits（参考音频/文本仍要填，推理需要）"
Write-Host "   3) 想让它成为默认音色：config.yaml 的 speech.tts_preference 把 gpt_sovits 排第一"
Write-Host '============================================================' -ForegroundColor Green
