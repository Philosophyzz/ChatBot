# =============================================================================
# 自检脚本：从环境到全链路的体检，出问题能明确告诉你是哪一层坏了
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\verify.ps1            # 完整自检
#   powershell -ExecutionPolicy Bypass -File scripts\verify.ps1 -Quick     # 只看服务状态
#   powershell -ExecutionPolicy Bypass -File scripts\verify.ps1 -SkipTests # 跳过单元测试
#   powershell -ExecutionPolicy Bypass -File scripts\verify.ps1 -Bench     # 额外做一次真实推理测速
# =============================================================================

[CmdletBinding()]
param(
    [string]$Root,
    [switch]$Quick,
    [switch]$SkipTests,
    [switch]$Bench
)

. (Join-Path $PSScriptRoot 'common.ps1')

$ProjectRoot = Get-ConfigRoot -Root $Root
$failures = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]

function Test-Item {
    param([string]$Name, [bool]$Ok, [string]$Detail = '')
    if ($Ok) {
        Write-Ok "$Name $Detail"
    } else {
        Write-Err2 "$Name $Detail"
        $failures.Add($Name) | Out-Null
    }
}

Write-Host ''
Write-Host '============================================================' -ForegroundColor White
Write-Host ' 环境与链路自检' -ForegroundColor White
Write-Host " $ProjectRoot" -ForegroundColor White
Write-Host '============================================================' -ForegroundColor White

# --- 1. 硬件与磁盘 -----------------------------------------------------------
Write-Step '硬件与磁盘'
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $gpuInfo = & nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version --format=csv,noheader 2>$null
    Write-Ok "GPU: $gpuInfo"
} else {
    Test-Item 'nvidia-smi' $false '未找到，无法确认 GPU 状态'
}
$free = Get-FreeSpaceGB -Path $ProjectRoot
Test-Item '磁盘可用空间' ($free -gt 5) "($free GB)"

# --- 2. Python 环境 ----------------------------------------------------------
Write-Step 'Python 环境'
$python = Get-Python -Root $ProjectRoot
Test-Item 'Python 解释器' ([bool]$python) "$python"
if ($python) {
    $pyVersion = & $python --version 2>&1
    Write-Host "    $pyVersion"
    $probe = & $python -c "import sys; print(sys.version_info >= (3, 10))" 2>&1
    Test-Item 'Python 版本 >= 3.10' ($probe -match 'True')
}

if (-not $Quick -and $python) {
    Write-Step 'Python 依赖'
    # 探测逻辑放在独立 .py 文件里：用 python -c 传内联脚本时，命令行会按 ANSI
    # 代码页传递，非 ASCII 内容到 Python 手里已经损坏（本项目真实踩过）。
    $moduleProbe = Join-Path $ProjectRoot 'tests\check_module.py'
    $required = @(
        @('fastapi',  'fastapi'),
        @('uvicorn',  'uvicorn[standard]'),
        @('httpx',    'httpx'),
        @('pydantic', 'pydantic'),
        @('numpy',    'numpy'),
        @('yaml',     'PyYAML'),
        @('multipart','python-multipart'),
        @('websockets', 'websockets')
    )
    foreach ($entry in $required) {
        & $python $moduleProbe $entry[0] | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "$($entry[1])"
        } else {
            Write-Warn2 "$($entry[1]) 未安装（pip install $($entry[1])）"
            $warnings.Add($entry[1]) | Out-Null
        }
    }
    foreach ($optional in @(@('faster_whisper', '语音识别'), @('edge_tts', '在线语音合成'))) {
        & $python $moduleProbe $optional[0] | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "$($optional[1])（$($optional[0])）"
        } else {
            Write-Warn2 "$($optional[1])不可用：$($optional[0]) 未安装"
            $warnings.Add($optional[0]) | Out-Null
        }
    }
}

# --- 3. 代码可导入性 ---------------------------------------------------------
if (-not $Quick -and $python) {
    Write-Step '代码导入自检'
    & $python (Join-Path $ProjectRoot 'tests\check_imports.py')
    if ($LASTEXITCODE -eq 0) {
        Write-Ok '全部模块导入成功'
    } else {
        $failures.Add('模块导入') | Out-Null
    }
}

# --- 4. 单元测试 -------------------------------------------------------------
if (-not $Quick -and -not $SkipTests -and $python) {
    Write-Step '单元测试（不需要 GPU 和模型权重）'
    Push-Location $ProjectRoot
    try {
        & $python -m pytest tests -q --no-header 2>&1 | ForEach-Object { Write-Host "    $_" }
        if ($LASTEXITCODE -ne 0) {
            # pytest 未安装时给出明确指引，而不是报一堆红
            & $python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('pytest') else 1)" 2>$null
            if ($LASTEXITCODE -ne 0) {
                Write-Warn2 'pytest 未安装：pip install pytest'
                $warnings.Add('pytest') | Out-Null
            } else {
                $failures.Add('单元测试') | Out-Null
            }
        } else {
            Write-Ok '单元测试全部通过'
        }
    } finally {
        Pop-Location
    }
}

# --- 5. 模型文件 -------------------------------------------------------------
Write-Step '模型文件'
$ggufDir = Join-Path (Get-ModelsRoot -Root $ProjectRoot) 'gguf'
$ggufFiles = @()
if (Test-Path -LiteralPath $ggufDir) {
    $ggufFiles = @(Get-ChildItem -LiteralPath $ggufDir -Filter '*.gguf' -ErrorAction SilentlyContinue)
}
if ($ggufFiles.Count -gt 0) {
    foreach ($file in $ggufFiles) {
        Write-Ok "$($file.Name)  $([math]::Round($file.Length / 1GB, 2)) GB"
    }
} else {
    Write-Warn2 'models\gguf 下没有模型文件'
    Write-Host '    运行：scripts\download-models.ps1 -Mirror'
    $warnings.Add('模型权重') | Out-Null
}

$serverExe = Get-LlamaServerPath -Root $ProjectRoot
Test-Item 'llama-server.exe' ([bool]$serverExe) "$serverExe"

# --- 6. 服务状态 -------------------------------------------------------------
Write-Step '服务状态'
$services = @(
    @{ Name = '对话模型'; Port = 8080; Path = '/health' },
    @{ Name = '嵌入模型'; Port = 8081; Path = '/health' },
    @{ Name = '重排模型'; Port = 8082; Path = '/health' },
    @{ Name = '网页服务'; Port = 8077; Path = '/api/health' }
)
$upCount = 0
foreach ($service in $services) {
    $uri = "http://127.0.0.1:$($service.Port)$($service.Path)"
    try {
        $response = Invoke-WebRequest -Uri $uri -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Ok "$($service.Name) 端口 $($service.Port) 正常"
            $upCount++
        } else {
            Write-Warn2 "$($service.Name) 端口 $($service.Port) 返回 $($response.StatusCode)"
        }
    } catch {
        if ($service.Name -in @('嵌入模型', '重排模型')) {
            Write-Warn2 "$($service.Name) 端口 $($service.Port) 未启动（可选，记忆会降级但仍可用）"
        } elseif ($service.Name -eq '网页服务') {
            Write-Warn2 "$($service.Name) 未启动：scripts\start-all.ps1"
        } else {
            Write-Warn2 "$($service.Name) 未启动：scripts\start-models.ps1"
        }
    }
}

# --- 7. 真实推理测速（可选）--------------------------------------------------
if ($Bench -and $upCount -gt 0) {
    Write-Step '真实推理测速'
    $body = @{
        model    = 'qwen3.6-27b'
        messages = @(@{ role = 'user'; content = '用一句话介绍你自己。' })
        max_tokens = 64
        stream   = $false
    } | ConvertTo-Json -Depth 6
    try {
        $started = Get-Date
        $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8080/v1/chat/completions' -Method Post `
            -Body $body -ContentType 'application/json' -TimeoutSec 300
        $elapsed = ((Get-Date) - $started).TotalSeconds
        $text = $response.choices[0].message.content
        Write-Ok ("用时 {0:N1}s ｜ 输出: {1}" -f $elapsed, $text.Trim())
        if ($response.timings) {
            Write-Host ("    预填充 {0:N1} tok/s ｜ 生成 {1:N1} tok/s ｜ 命中缓存 {2} tokens" -f `
                $response.timings.prompt_per_second, $response.timings.predicted_per_second, $response.timings.cache_n) `
                -ForegroundColor DarkGray
            Write-Host '    若生成速度低于 3 tok/s：减小 --n-gpu-layers 反而可能更快（避免显存抖动），' -ForegroundColor DarkGray
            Write-Host '    或改用 -Tier Qwen3.6-14B-A3B-FableVibes-Q4_K_M 把权重全部放进显存。' -ForegroundColor DarkGray
        }
    } catch {
        Write-Warn2 "测速失败：$($_.Exception.Message)"
    }
}

# --- 汇总 -------------------------------------------------------------------
Write-Host ''
Write-Host '============================================================' -ForegroundColor White
if ($failures.Count -eq 0) {
    Write-Host ' 自检通过' -ForegroundColor Green
} else {
    Write-Host " 自检发现 $($failures.Count) 个问题" -ForegroundColor Red
    foreach ($item in $failures) { Write-Host "   - $item" -ForegroundColor Red }
}
if ($warnings.Count -gt 0) {
    Write-Host " 可选组件缺失（不影响核心功能）：$($warnings -join ', ')" -ForegroundColor Yellow
}
Write-Host '============================================================' -ForegroundColor White
Write-Host ''
if (-not $Quick) { Show-Vram }

if ($failures.Count -gt 0) { exit 1 }
exit 0
