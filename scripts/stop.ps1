# 停止所有由本项目启动的服务（模型服务 + 网页服务）
#
# 网页服务优先走"优雅退出"：POST /api/shutdown 让 uvicorn 正常跑完 lifespan 收尾，
# 从而关闭 SQLite 连接并 checkpoint WAL。Windows 没有可用的 SIGTERM，
# Stop-Process -Force 是直接杀进程，数据库不会被关闭（实测会留下几百 KB 到几 MB 的
# -wal 文件，而且 data\memory.sqlite3 在一段时间内仍被占用）。所以先优雅，超时才强杀。
[CmdletBinding()]
param([string]$Root)

. (Join-Path $PSScriptRoot 'common.ps1')
$ProjectRoot = Get-ConfigRoot -Root $Root

Write-Step '停止网页服务'
$appPidFile = Join-Path $ProjectRoot 'data\app.pids.json'
$killed = 0
$apiPid = $null
if (Test-Path -LiteralPath $appPidFile) {
    $info = Get-Content -LiteralPath $appPidFile -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($info.api) { $apiPid = [int]$info.api }
}

# 端口取自配置，避免和 -Port 启动的实例对不上
$port = 8077
$configFile = Join-Path $ProjectRoot 'config\config.yaml'
if (Test-Path -LiteralPath $configFile) {
    $match = Select-String -LiteralPath $configFile -Pattern '^\s*port:\s*(\d+)' | Select-Object -First 1
    if ($match) { $port = [int]$match.Matches[0].Groups[1].Value }
}

$graceful = $false
$probe = Invoke-LocalHttp -Uri "http://127.0.0.1:$port/api/health" -TimeoutSec 3
if ($probe.Ok) {
    $result = Invoke-LocalHttp -Uri "http://127.0.0.1:$port/api/shutdown" -Method 'POST' -TimeoutSec 5
    if ($result.Ok) {
        # 等待端口真正释放：uvicorn 要先停收新连接，再跑 lifespan 收尾
        $deadline = (Get-Date).AddSeconds(10)
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Milliseconds 400
            $check = Invoke-LocalHttp -Uri "http://127.0.0.1:$port/api/health" -TimeoutSec 2
            if (-not $check.Ok) { $graceful = $true; break }
        }
        if ($graceful) { Write-Ok '网页服务已优雅退出（数据库已关闭、WAL 已合并）' }
        else { Write-Warn2 '优雅退出超时，改为强制结束' }
    } else {
        Write-Warn2 "优雅退出请求失败（$($result.Body)），改为强制结束"
    }
}

if (-not $graceful) {
    if ($apiPid) {
        Stop-Process -Id $apiPid -Force -ErrorAction SilentlyContinue
        Write-Ok "已停止 api PID $apiPid"
        $killed++
    }
    # 兜底：按命令行特征清理残留进程（例如脚本被 Ctrl+C 中断时）
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*api.server*' -or $_.CommandLine -like '*run_server.py*' } |
        ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            Write-Ok "已停止残留 api 进程 PID $($_.ProcessId)"
            $killed++
        }
}
if (Test-Path -LiteralPath $appPidFile) { Remove-Item -LiteralPath $appPidFile -Force -ErrorAction SilentlyContinue }

Write-Step '停止模型服务'
$modelPidFile = Join-Path $ProjectRoot 'data\model-server.pids.json'
if (Test-Path -LiteralPath $modelPidFile) {
    $pids = Get-Content -LiteralPath $modelPidFile -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($name in $pids.PSObject.Properties.Name) {
        $id = $pids.$name
        if ($id) {
            Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
            Write-Ok "已停止 $name PID $id"
            $killed++
        }
    }
    Remove-Item -LiteralPath $modelPidFile -Force -ErrorAction SilentlyContinue
}
Get-Process -Name 'llama-server' -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    Write-Ok "已停止 llama-server PID $($_.Id)"
    $killed++
}

Write-Host ''
if ($killed -eq 0 -and -not $graceful) { Write-Warn2 '没有发现正在运行的服务' } else { Write-Ok '服务已全部停止' }
Show-Vram
