# =============================================================================
# 启动 llama.cpp 模型服务（三个独立实例）
#
# 为什么是三个而不是一个多模型路由实例：
#   对话 / 嵌入 / 重排需要不同参数（上下文长度、pooling 方式、显存预算），
#   分开跑端口隔离、互不影响，排障时能明确知道是哪一个服务出了问题。
#
# 端口分配（与 config/config.yaml 中的 base_url 一一对应）：
#   8080  对话模型    llm.base_url
#   8081  嵌入模型    embedding.base_url
#   8082  重排模型    reranker.base_url
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\start-models.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\start-models.ps1 -Tier balanced-14b
#   powershell -ExecutionPolicy Bypass -File scripts\start-models.ps1 -NoSupport
#   powershell -ExecutionPolicy Bypass -File scripts\start-models.ps1 -ChatOnly
#
# 显存调优（16GB 卡必读）：
#   27B Q4 权重约 16.5GB，装不进 12.5GB 可用显存，所以 -NGpuLayers 默认 28，
#   其余层放内存。速度慢但质量高。想看/调这个数字：
#     1) 启动后执行  nvidia-smi
#     2) 若 "memory.free" 还有 1GB 以上富余，把 -NGpuLayers 加 2~4 再试
#     3) 若报 OOM 或显存打满导致卡顿，减 2~4
#   想要满速就换档：-Tier balanced-14b（权重全进显存）
# =============================================================================

[CmdletBinding()]
param(
    [string]$Root,
    [string]$Tier,
    [int]$NGpuLayers = -999,      # -999 表示用 models.json 里的默认值
    [int]$ContextSize = 0,        # 0 表示用 models.json 里的默认值
    [switch]$NoSupport,
    [switch]$ChatOnly,
    [switch]$Foreground
)

. (Join-Path $PSScriptRoot 'common.ps1')

$ProjectRoot = Get-ConfigRoot -Root $Root
$serverExe = Get-LlamaServerPath -Root $ProjectRoot
if (-not $serverExe) {
    Write-Err2 "找不到 llama-server.exe：$ProjectRoot\bin\llama.cpp\"
    Write-Host '    请先运行 scripts\install.ps1，或手动下载 llama.cpp 的 Windows CUDA 构建。'
    exit 1
}

$modelsConfigPath = Join-Path $ProjectRoot 'config\models.json'
$modelsConfig = Get-Content -LiteralPath $modelsConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $Tier) { $Tier = $modelsConfig.default_tier }

$tierSpec = $modelsConfig.chat_tiers | Where-Object { $_.id -eq $Tier } | Select-Object -First 1
if (-not $tierSpec) {
    $available = ($modelsConfig.chat_tiers | ForEach-Object { $_.id }) -join ', '
    Write-Err2 "未知档位：$Tier（可选：$available）"
    exit 1
}

$ggufDir = Join-Path $ProjectRoot 'models\gguf'
$chatModel = Join-Path $ggufDir $tierSpec.local_name
if (-not (Test-Path -LiteralPath $chatModel)) {
    Write-Err2 "模型文件不存在：$chatModel"
    Write-Host "    请先运行: scripts\download-models.ps1 -Tier $Tier"
    exit 1
}

if ($NGpuLayers -eq -999) { $NGpuLayers = [int]$tierSpec.n_gpu_layers }
if ($ContextSize -eq 0) { $ContextSize = [int]$tierSpec.ctx }

$logsDir = Join-Path $ProjectRoot 'logs'
$pids = @{}

function Start-LlamaServer {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$ModelPath,
        [Parameter(Mandatory)][int]$Port,
        [Parameter(Mandatory)][string]$Alias,
        [int]$Ctx = 8192,
        [int]$GpuLayers = -1,
        [string[]]$ExtraArgs = @()
    )

    $existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Warn2 "$Name 端口 $Port 已被占用，跳过启动（可能已在运行）"
        return $null
    }

    $logFile = Join-Path $logsDir "$Name.log"
    $errFile = Join-Path $logsDir "$Name.err.log"

    # 固定参数来自 config\models.json 的 default_server_args —— 引擎里的运行时切换
    # （src\llm\supervisor.py，Web 界面 / 桌宠菜单用）读的是同一个键，这样脚本与引擎
    # 不会各自演化出两套参数。文件里没有这个键时，下面的兜底列表与原实现完全一致。
    $defaultArgs = @()
    if ($modelsConfig.default_server_args) { $defaultArgs = @($modelsConfig.default_server_args) }
    elseif ($modelsConfig.defaultServerArgs) { $defaultArgs = @($modelsConfig.defaultServerArgs) }
    if (-not $defaultArgs) {
        Write-Warn2 "config\models.json 里没有 default_server_args，使用脚本内置的兜底参数"
        $defaultArgs = @(
            # 全量卸载时 FlashAttention 更快更省显存；部分卸载时同样有益
            '--flash-attn', 'on',
            # KV cache 量化到 q8：上下文越长收益越大，是本机跑长对话的关键
            '--cache-type-k', 'q8_0',
            '--cache-type-v', 'q8_0',
            # 对话场景并行请求数为 1，多余的 slot 只会浪费 KV 显存
            '--parallel', '1',
            '--cont-batching',
            # 思考预算 = 0（关闭思考）。这是本项目的关键调优，实测数据（9B、全量 GPU、
            # 带人设系统提示词、同一个问题）：
            #   默认（不限制）           14~20 秒   思考 4278~6053 字
            #   --reasoning-budget 256     2.4 秒   思考 ~780 字
            #   --reasoning-budget 0     0.3~0.5 秒  思考 0        <- 采用
            # 关掉思考后回答反而更自然口语，因为推理预算不再被"我该如何扮演这个角色"
            # 这类自我盘问吃光。将来要做复杂推理任务时，把 0 改成 256 即可。
            '--reasoning-budget', '0',
            '--jinja',
            '--metrics'
        )
    }
    # 不要加 --cache-reuse：新版 llama.cpp 对部分模型会打印
    # "cache_reuse is not supported by this context, it will be disabled" 并忽略它。

    # 注意：PowerShell 里 $args 是自动变量，这里必须换名，否则会静默出错。
    $serverArgs = @(
        '--model', $ModelPath,
        '--alias', $Alias,
        '--host', '127.0.0.1',
        '--port', "$Port",
        '--ctx-size', "$Ctx",
        '--n-gpu-layers', "$GpuLayers"
    ) + $defaultArgs + $ExtraArgs

    Write-Host "    启动 $Name (端口 $Port, ctx $Ctx, ngl $GpuLayers)"
    $process = Start-Process -FilePath $serverExe -ArgumentList $serverArgs `
        -RedirectStandardOutput $logFile -RedirectStandardError $errFile `
        -PassThru -WindowStyle Hidden
    return $process
}

Write-Host ''
Write-Host '============================================================' -ForegroundColor White
Write-Host " 启动模型服务（档位: $Tier）" -ForegroundColor White
Write-Host " 上下文 $ContextSize ｜ GPU 层数 $NGpuLayers" -ForegroundColor White
Write-Host '============================================================' -ForegroundColor White

Write-Step '启动前的显存状态'
Show-Vram
Write-Host '    提示：16GB 卡上若已有浏览器/通讯软件占用 3~4GB，27B 档位请优先减小 -NGpuLayers。'

# --- 对话模型 ---------------------------------------------------------------
# --alias 必须与 config/config.yaml 的 llm.model 完全一致，否则请求会被模型服务
# 以「模型不存在」拒绝（llama-server 只响应它自己注册的别名）。
# 这里用档位 id 作别名（quality-27b / balanced-14b / fast-9b）：它稳定、与档位一一
# 对应。早期硬编码的 'qwen3.6-27b' 一换档位就对不上。
$chatAlias = $Tier
Write-Step "启动对话模型（别名 $chatAlias）"
$chat = Start-LlamaServer -Name "llama-chat" -ModelPath $chatModel -Port 8080 `
    -Alias $chatAlias -Ctx $ContextSize -GpuLayers $NGpuLayers
if ($chat) {
    $pids['chat'] = $chat.Id
    Write-Ok "llama-chat PID $($chat.Id)  别名 $chatAlias  日志 logs\llama-chat.log"
    if ($chatAlias -ne 'quality-27b') {
        Write-Host "    记得同步 config\config.yaml 的 llm.model: $chatAlias" -ForegroundColor DarkGray
    }
}

# --- 辅助模型 ---------------------------------------------------------------
if (-not $ChatOnly -and -not $NoSupport) {
    foreach ($spec in $modelsConfig.support_models) {
        if ($spec.kind -eq 'python') { continue }
        $path = Join-Path $ggufDir $spec.local_name
        if (-not (Test-Path -LiteralPath $path)) {
            Write-Warn2 "$($spec.id) 未下载，跳过。$($spec.notes)"
            continue
        }
        Write-Step "启动辅助模型：$($spec.id)"
        $extra = @()
        if ($spec.server_args) { $extra = @($spec.server_args) }
        $proc = Start-LlamaServer -Name "llama-$($spec.id)" -ModelPath $path `
            -Port ([int]$spec.port) -Alias $spec.local_name.Replace('.gguf', '') `
            -Ctx 8192 -GpuLayers -1 -ExtraArgs $extra
        if ($proc) {
            $pids[$spec.id] = $proc.Id
            Write-Ok "$($spec.id) PID $($proc.Id)  端口 $($spec.port)"
        }
    }
}

# --- 等待就绪 ---------------------------------------------------------------
Write-Step '等待服务就绪（首次加载 27B 模型可能需要 1~3 分钟）'
$ports = @(8080)
if ($pids.ContainsKey('embedding')) { $ports += 8081 }
if ($pids.ContainsKey('reranker')) { $ports += 8082 }

$deadline = (Get-Date).AddMinutes(6)
$ready = @{}
while ((Get-Date) -lt $deadline -and $ready.Count -lt $ports.Count) {
    foreach ($port in $ports) {
        if ($ready.ContainsKey($port)) { continue }
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -UseBasicParsing -TimeoutSec 4
            if ($response.StatusCode -eq 200) {
                $ready[$port] = $true
                Write-Ok "端口 $port 已就绪"
            }
        } catch {
            # 模型还在加载，属正常
        }
    }
    if ($ready.Count -lt $ports.Count) {
        $pending = ($ports | Where-Object { -not $ready.ContainsKey($_) }) -join ', '
        Write-Host "`r    等待端口: $pending …" -NoNewline -ForegroundColor DarkGray
        Start-Sleep -Seconds 3
    }
}
Write-Host ''

foreach ($port in $ports) {
    if (-not $ready.ContainsKey($port)) {
        Write-Err2 "端口 $port 未在超时时间内就绪，请查看 logs\ 下对应日志"
    }
}

Write-Step '启动后的显存状态'
Show-Vram

# 保存 PID，便于 stop 脚本清理
$pidFile = Join-Path $ProjectRoot 'data\model-server.pids.json'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $pidFile) | Out-Null
$pids | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding UTF8
Write-Ok "PID 已记录到 data\model-server.pids.json"

Write-Host ''
Write-Host '模型服务就绪。可用接口：' -ForegroundColor White
Write-Host '  对话  http://127.0.0.1:8080/v1/chat/completions'
if ($pids.ContainsKey('embedding')) { Write-Host '  嵌入  http://127.0.0.1:8081/v1/embeddings' }
if ($pids.ContainsKey('reranker')) { Write-Host '  重排  http://127.0.0.1:8082/v1/rerank' }
Write-Host ''
Write-Host ' 接着运行：powershell -ExecutionPolicy Bypass -File scripts\start-all.ps1' -ForegroundColor Green
Write-Host ''

if ($Foreground) {
    Write-Host '按 Ctrl+C 结束所有模型服务…'
    try {
        while ($true) { Start-Sleep -Seconds 5 }
    } finally {
        foreach ($key in $pids.Keys) {
            Stop-Process -Id $pids[$key] -Force -ErrorAction SilentlyContinue
        }
    }
}
