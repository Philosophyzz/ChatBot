# =============================================================================
# 一键启动：模型服务 + 网页界面，并自动打开浏览器
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\start-all.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\start-all.ps1 -Tier balanced-14b
#   powershell -ExecutionPolicy Bypass -File scripts\start-all.ps1 -Mock        # 不加载模型，跑模拟模式
#   powershell -ExecutionPolicy Bypass -File scripts\start-all.ps1 -NoBrowser   # 不自动开浏览器
#   powershell -ExecutionPolicy Bypass -File scripts\start-all.ps1 -Lan         # 允许局域网访问
#   powershell -ExecutionPolicy Bypass -File scripts\start-all.ps1 -Restart     # 先停掉已有实例再启动
#
# 重复执行是安全的：已在运行的服务会被检测到并复用，不会重复启动。
# 结束后关闭所有服务：scripts\stop.ps1
#
# 维护提示：本文件必须保存为「UTF-8 with BOM」。Windows PowerShell 5.1 对没有 BOM
# 的 .ps1 会按系统 ANSI 代码页解码，中文会变成乱码 —— 本项目真实踩过这个坑。
# 需要修改时请整体重写文件，不要用 Get-Content -Raw / Set-Content 往返读写。
# =============================================================================

[CmdletBinding()]
param(
    [string]$Root,
    [string]$Tier,
    [int]$Port = 0,
    [switch]$Mock,
    [switch]$NoBrowser,
    [switch]$Lan,
    [switch]$SkipModels,
    [switch]$Restart
)

. (Join-Path $PSScriptRoot 'common.ps1')

$ProjectRoot = Get-ConfigRoot -Root $Root
$python = Get-Python -Root $ProjectRoot
if (-not $python) {
    Write-Err2 '找不到 Python 环境，请先运行 scripts\install.ps1'
    exit 1
}

if ($Port -eq 0) { $Port = 8077 }

# --- 辅助：检测某端口是否已有服务在监听 --------------------------------------
function Get-PortListener {
    param([int]$PortNumber)
    $conn = Get-NetTCPConnection -LocalPort $PortNumber -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    return $conn
}

function Test-HttpOk {
    param([string]$Uri, [int]$TimeoutSec = 4)
    try {
        $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec $TimeoutSec
        return ($response.StatusCode -eq 200)
    } catch {
        return $false
    }
}

Write-Host ''
Write-Host '============================================================' -ForegroundColor White
Write-Host ' 本地 LLM 聊天机器人' -ForegroundColor White
Write-Host " 项目目录: $ProjectRoot" -ForegroundColor White
Write-Host '============================================================' -ForegroundColor White

# --- 0. 可选：先停掉已有实例 -------------------------------------------------
if ($Restart) {
    Write-Step '按 -Restart 要求先停止已有服务'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'stop.ps1') `
        -Root $ProjectRoot 2>&1 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    Start-Sleep -Seconds 2
}

# --- 1. 模型服务 -------------------------------------------------------------
if (-not $Mock -and -not $SkipModels) {
    if (Test-HttpOk -Uri 'http://127.0.0.1:8080/health' -TimeoutSec 3) {
        Write-Ok '检测到对话模型服务已在运行（端口 8080），直接复用'
    } else {
        $listener = Get-PortListener -PortNumber 8080
        if ($listener) {
            Write-Warn2 "端口 8080 已被 PID $($listener.OwningProcess) 占用，但 /health 未响应；可能仍在加载模型。"
            Write-Host '    等待它就绪（最多 3 分钟）…' -ForegroundColor DarkGray
            $deadline = (Get-Date).AddMinutes(3)
            $ready = $false
            while ((Get-Date) -lt $deadline) {
                if (Test-HttpOk -Uri 'http://127.0.0.1:8080/health' -TimeoutSec 3) { $ready = $true; break }
                Start-Sleep -Seconds 3
            }
            if ($ready) {
                Write-Ok '模型服务已就绪'
            } else {
                Write-Warn2 '端口被占用但模型服务始终未就绪。可执行 scripts\stop.ps1 后再试。'
            }
        } else {
            Write-Step '启动模型服务（后台）'
            $modelArgs = @('-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'start-models.ps1'))
            if ($Root) { $modelArgs += @('-Root', $Root) }
            if ($Tier) { $modelArgs += @('-Tier', $Tier) }
            $modelProc = Start-Process -FilePath 'powershell.exe' -ArgumentList $modelArgs -PassThru -WindowStyle Minimized
            Write-Host "    模型服务启动进程 PID $($modelProc.Id)，日志见 logs\llama-chat.log"
            Write-Warn2 '首次加载 27B 模型需要 1~3 分钟；网页此时可以打开，但提问要等模型就绪。'
        }
    }
} elseif ($Mock) {
    Write-Warn2 '模拟模式（-Mock）：不加载本地模型，用于验证界面与记忆链路'
}

# --- 2. 网页服务 -------------------------------------------------------------
Write-Step '启动网页服务'

# 端口冲突是重复执行本脚本最常见的结果，必须明确诊断而不是抛一个 WinError 10048。
$apiPid = $null
$apiProc = $null
$apiAlreadyUp = Test-HttpOk -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 4
if ($apiAlreadyUp) {
    $listener = Get-PortListener -PortNumber $Port
    $apiPid = if ($listener) { $listener.OwningProcess } else { $null }
    Write-Ok "网页服务已在运行（端口 $Port，PID $apiPid），直接复用"
    Write-Host '    如需重启请加 -Restart，或先运行 scripts\stop.ps1' -ForegroundColor DarkGray
} else {
    $listener = Get-PortListener -PortNumber $Port
    if ($listener) {
        Write-Err2 "端口 $Port 已被 PID $($listener.OwningProcess) 占用，但它不响应 /api/health。"
        Write-Host "    处理方式（任选其一）：" -ForegroundColor Yellow
        Write-Host "      a) 停掉占用者：powershell -ExecutionPolicy Bypass -File scripts\stop.ps1"
        Write-Host "      b) 换一个端口：powershell -ExecutionPolicy Bypass -File scripts\start-all.ps1 -Port 8078"
        Write-Host "      c) 查看是谁：Get-Process -Id $($listener.OwningProcess) | Select-Object Name, Path"
        exit 1
    }

    $apiArgs = @((Join-Path $ProjectRoot 'run_server.py'), '--port', "$Port")
    if ($Mock) { $apiArgs += '--mock' }
    if ($Lan) {
        $apiArgs += @('--host', '0.0.0.0')
        Write-Warn2 '已开启局域网访问：同网段设备可直接打开，注意本服务没有登录鉴权。'
    }

    $apiLog = Join-Path $ProjectRoot 'logs\api.log'
    $apiErr = Join-Path $ProjectRoot 'logs\api.err.log'

    # 用 run_server.py 而不是 `-m api.server`：后者要求工作目录必须是 src，
    # 且会在 stdout 里产生一条 'found in sys.modules' 的 RuntimeWarning 噪声。
    $apiProc = Start-Process -FilePath $python -ArgumentList $apiArgs `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $apiLog -RedirectStandardError $apiErr `
        -PassThru -WindowStyle Hidden
    $apiPid = $apiProc.Id
    Write-Host "    PID $apiPid  日志 logs\api.log"
}

# --- 3. 等待接口就绪 ---------------------------------------------------------
# 记忆子系统启动时要探测嵌入/重排服务，若它们未运行会有连接超时，主服务因此需要
# 十几秒才监听端口。所以这里的等待时间放宽到 120 秒，并在超时时给出可执行的诊断。
Write-Step '等待接口就绪'
$url = "http://127.0.0.1:$Port/"
$deadline = (Get-Date).AddSeconds(120)
$ok = $apiAlreadyUp
$waited = 0
while (-not $ok -and (Get-Date) -lt $deadline) {
    if ($apiProc -and $apiProc.HasExited) {
        Write-Err2 "网页服务已退出（退出码 $($apiProc.ExitCode)），最近日志："
        foreach ($file in @($apiErr, $apiLog)) {
            if (Test-Path -LiteralPath $file) {
                Write-Host "    --- $([System.IO.Path]::GetFileName($file)) 末尾 ---" -ForegroundColor DarkGray
                Get-Content -LiteralPath $file -Tail 20 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
            }
        }
        exit 1
    }
    if (Test-HttpOk -Uri "${url}api/health" -TimeoutSec 4) { $ok = $true; break }
    Start-Sleep -Milliseconds 900
    $waited += 0.9
}

if ($ok) {
    if ($apiAlreadyUp) { Write-Ok '接口已就绪（复用已有实例）' } else { Write-Ok "接口已就绪（用时约 $([Math]::Round($waited,1)) 秒）" }
} else {
    Write-Err2 '接口在 120 秒内未就绪。'
    if (Test-Path -LiteralPath $apiErr) {
        Write-Host '    logs\api.err.log 末尾：' -ForegroundColor DarkGray
        Get-Content -LiteralPath $apiErr -Tail 15 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    }
}

# --- 4. 记录 PID -------------------------------------------------------------
if ($apiPid -and -not $apiAlreadyUp) {
    $pidFile = Join-Path $ProjectRoot 'data\app.pids.json'
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $pidFile) | Out-Null
    @{ api = $apiPid; port = $Port } | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding UTF8
}

# --- 5. 提示记忆检索是否降级 --------------------------------------------------
if (-not (Test-HttpOk -Uri 'http://127.0.0.1:8081/health' -TimeoutSec 3)) {
    Write-Host ''
    Write-Warn2 '嵌入服务（8081）未运行：记忆检索已降级为本地哈希嵌入，语义召回质量下降。'
    Write-Host '    启用完整记忆检索（需先下载 0.6GB 模型）：' -ForegroundColor DarkGray
    Write-Host '      powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1 -Mirror -Support' -ForegroundColor DarkGray
}

# --- 6. 打开浏览器 -----------------------------------------------------------
$lanIp = $null
if ($Lan) {
    $lanIp = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' -and $_.PrefixOrigin -ne 'WellKnown' } |
        Select-Object -First 1 -ExpandProperty IPAddress)
}

Write-Host ''
Write-Host '============================================================' -ForegroundColor Green
Write-Host ' 已启动' -ForegroundColor Green
Write-Host " 本机访问: $url" -ForegroundColor Green
if ($lanIp) { Write-Host " 局域网访问: http://$lanIp`:$Port/" -ForegroundColor Green }
Write-Host '============================================================' -ForegroundColor Green
Write-Host ''
Write-Host ' 关闭服务：powershell -ExecutionPolicy Bypass -File scripts\stop.ps1'
Write-Host ' 查看状态：powershell -ExecutionPolicy Bypass -File scripts\verify.ps1 -Quick'
Write-Host ''

if (-not $NoBrowser) { Start-Process $url }
