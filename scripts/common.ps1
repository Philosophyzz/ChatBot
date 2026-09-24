# =============================================================================
# 路径与公共函数（被其它脚本 dot-source 引用）
# 目标：所有可写内容都落在项目根目录（默认 D:\Harness\ChatBot），绝不满 C 盘。
# =============================================================================

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# 控制台按 UTF-8 输出。Windows PowerShell 5.1 默认用系统 ANSI 代码页（中文环境是 GBK），
# 一旦输出被重定向或管道捕获，中文就会变成乱码 —— 脚本自己提示的信息反而看不懂。
#
# 三处都要设，缺一个就会在某种场景下漏出乱码：
#   [Console]::OutputEncoding  —— 写控制台
#   $OutputEncoding            —— PowerShell 传给外部程序的管道编码
#   $PSDefaultParameterValues  —— Out-File / Set-Content 的默认编码，
#                                 也就是 `> file` 与 -RedirectStandardOutput 走的那条路
try {
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [Console]::OutputEncoding = $utf8NoBom
    $OutputEncoding = $utf8NoBom
    $PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
    $PSDefaultParameterValues['Set-Content:Encoding'] = 'utf8'
} catch {
    # 极少数宿主（如某些 CI 包装器）不允许改编码，忽略即可
}

function Get-ProjectRoot {
    param([string]$Override)
    if ($Override) { return (Resolve-Path -LiteralPath $Override).Path }
    if ($env:CHATBOT_ROOT) { return (Resolve-Path -LiteralPath $env:CHATBOT_ROOT).Path }
    # 本文件位于 <root>\scripts\，因此根目录是上一级
    return (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
}

function Get-ConfigRoot {
    <#
      解析真实项目根目录：config\local.yaml 里的 paths.root 优先级最高。
      这样即使把代码仓库放在别处、模型放到 D 盘，脚本也能找对地方。
    #>
    param([string]$Root)
    $base = Get-ProjectRoot -Override $Root
    $local = Join-Path $base 'config\local.yaml'
    if (Test-Path -LiteralPath $local) {
        foreach ($line in Get-Content -LiteralPath $local -Encoding UTF8) {
            if ($line -match '^\s*root\s*:\s*"?([^"#]+?)"?\s*$') {
                $candidate = $Matches[1].Trim()
                if ($candidate) { return $candidate }
            }
        }
    }
    return $base
}

function Get-CondaPython {
    <#
      找到 conda 环境里的 python.exe。

      直接问 conda（`conda env list --json`）而不是猜路径：conda 可能装在 C 盘或 D 盘，
      envs_dirs 也可能被 .condarc 改过，猜路径迟早猜错。只在真的需要时调用一次。
    #>
    param([string]$Name = 'chatbot', [string]$Root)
    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $conda) {
        foreach ($exe in @(
            (Join-Path $env:USERPROFILE 'miniconda3\Scripts\conda.exe'),
            (Join-Path $env:USERPROFILE 'anaconda3\Scripts\conda.exe'),
            'D:\miniconda3\Scripts\conda.exe',
            'D:\anaconda3\Scripts\conda.exe',
            'C:\ProgramData\miniconda3\Scripts\conda.exe',
            'C:\ProgramData\anaconda3\Scripts\conda.exe'
        )) {
            if (Test-Path -LiteralPath $exe) { $conda = Get-Item -LiteralPath $exe; break }
        }
    }
    if ($conda) {
        # conda 会往 stderr 打 WARNING（例如 conda-pypi 的提示）。在调用方设置了
        # $ErrorActionPreference = 'Stop' 的脚本里，原生 stderr 会被 PowerShell 当成
        # 致命错误并掐断整个脚本；所以这里局部放宽，并且把 stderr 一起吞掉。
        $previous = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $json = & $conda.Source env list --json 2>&1 | Out-String
            $parsed = $json | ConvertFrom-Json
            foreach ($envPath in $parsed.envs) {
                if ((Split-Path -Leaf $envPath) -eq $Name) {
                    $candidate = Join-Path $envPath 'python.exe'
                    if (Test-Path -LiteralPath $candidate) { return $candidate }
                }
            }
        } catch {
            # conda 坏了不该让整个脚本挂掉，继续走后面的兜底路径
        } finally {
            $ErrorActionPreference = $previous
        }
    }
    # conda 命令不可用（例如没进 PATH）时的常见位置
    foreach ($envPath in @(
        "D:\conda-envs\$Name",
        (Join-Path $env:USERPROFILE ".conda\envs\$Name"),
        "D:\miniconda3\envs\$Name",
        "C:\ProgramData\miniconda3\envs\$Name"
    )) {
        $candidate = Join-Path $envPath 'python.exe'
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    return $null
}

function Get-SystemPython {
    <#
      只找 PATH 上的 python。建 venv 时必须用系统解释器，用 Get-Python 会把
      "conda 环境里的 python" 当成系统 python，然后在 conda 环境里再套一个 venv。
    #>
    foreach ($name in @('python', 'python3', 'py')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    return $null
}

function Get-Python {
    <#
      解析用哪个解释器跑项目代码。顺序：

        1. 参数 -Preferred / 环境变量 CHATBOT_PYTHON（显式指定永远优先）
        2. conda 环境 chatbot（当前推荐：包都在 D 盘，见 README「运行环境」）
        3. venvs\main（scripts\install.ps1 建的 venv，保留兼容）
        4. PATH 上的 python / python3 / py

      返回可执行文件路径；都没有则返回 $null。
    #>
    param([string]$Root, [string]$Preferred = '')

    foreach ($explicit in @($Preferred, $env:CHATBOT_PYTHON)) {
        if ($explicit -and (Test-Path -LiteralPath $explicit)) { return $explicit }
    }

    $condaPython = Get-CondaPython -Name 'chatbot' -Root $Root
    if ($condaPython) { return $condaPython }

    $venvPython = Join-Path $Root 'venvs\main\Scripts\python.exe'
    if (Test-Path -LiteralPath $venvPython) { return $venvPython }

    foreach ($name in @('python', 'python3', 'py')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    return $null
}

function Test-CommandExists {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Write-Step {
    param([string]$Message)
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "    [OK] $Message" -ForegroundColor Green
}

function Write-Warn2 {
    param([string]$Message)
    Write-Host "    [!]  $Message" -ForegroundColor Yellow
}

function Write-Err2 {
    param([string]$Message)
    Write-Host "    [X]  $Message" -ForegroundColor Red
}

function Get-FreeSpaceGB {
    param([string]$Path)
    $drive = (Split-Path -Qualifier (Resolve-Path -LiteralPath $Path).Path).TrimEnd(':')
    $info = New-Object System.IO.DriveInfo($drive)
    return [math]::Round($info.AvailableFreeSpace / 1GB, 1)
}

function Invoke-Download {
    <#
      带进度提示的下载。GitHub 直连不稳定时自动改用镜像前缀重试。
    #>
    param(
        [Parameter(Mandatory)][string]$Uri,
        [Parameter(Mandatory)][string]$OutFile,
        [string[]]$MirrorPrefixes = @(),
        [int]$Retries = 3
    )
    $parent = Split-Path -Parent $OutFile
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    $candidates = @($Uri)
    foreach ($prefix in $MirrorPrefixes) { $candidates += ($prefix + $Uri) }

    foreach ($candidate in $candidates) {
        for ($attempt = 1; $attempt -le $Retries; $attempt++) {
            try {
                Write-Host "    下载: $candidate (第 $attempt 次)" -ForegroundColor DarkGray
                # GitHub 与镜像都需要 TLS1.2；Windows PowerShell 5.1 默认可能不启用
                [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
                $job = Start-Job -ScriptBlock {
                    param($u, $o)
                    $ProgressPreference = 'SilentlyContinue'
                    Invoke-WebRequest -Uri $u -OutFile $o -UseBasicParsing -TimeoutSec 1800
                } -ArgumentList $candidate, $OutFile
                while ($job.State -eq 'Running') {
                    Start-Sleep -Seconds 2
                    $size = 0
                    if (Test-Path -LiteralPath $OutFile) {
                        $size = (Get-Item -LiteralPath $OutFile).Length
                    }
                    if ($size -gt 0) {
                        Write-Host ("`r    已下载 {0:N1} MB" -f ($size / 1MB)) -NoNewline -ForegroundColor DarkGray
                    }
                }
                $result = Receive-Job -Job $job
                Remove-Job -Job $job -Force
                Write-Host ''
                if ($result -ne $null -or (Test-Path -LiteralPath $OutFile)) {
                    return $true
                }
            } catch {
                Write-Host ''
                Write-Warn2 "下载失败：$($_.Exception.Message)"
                Start-Sleep -Seconds (2 * $attempt)
            }
        }
    }
    return $false
}

function Get-GithubReleaseAsset {
    <#
      从 GitHub Releases 里找匹配的资产。凭据可选（提高 API 限额）。
      返回 @{ Tag = ''; Url = ''; Name = '' } 或 $null

      -Tag 用于锁定到指定 release：配套的资产（例如主包与 CUDA 运行时包）必须来自
      **同一个** release，否则 DLL 与 ggml-cuda.dll 的 ABI 可能不匹配。
    #>
    param(
        [Parameter(Mandatory)][string]$Repo,
        [Parameter(Mandatory)][string]$Pattern,
        [int]$MaxReleases = 15,
        [string]$Tag,
        [switch]$PreferStable
    )
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $headers = @{ 'User-Agent' = 'chatbot-setup' }
    if ($env:GITHUB_TOKEN) { $headers['Authorization'] = "Bearer $($env:GITHUB_TOKEN)" }

    if ($Tag) {
        # 指定 release 时直接用 by-tag 端点，避免分页拿不到老 release。
        $url = "https://api.github.com/repos/$Repo/releases/tags/$Tag"
    } else {
        $url = "https://api.github.com/repos/$Repo/releases?per_page=$MaxReleases"
    }
    try {
        $payload = Invoke-RestMethod -Uri $url -Headers $headers -TimeoutSec 60
    } catch {
        Write-Warn2 "无法访问 GitHub API：$($_.Exception.Message)"
        return $null
    }
    # by-tag 返回单个对象，列表端点返回数组；统一成数组处理。
    $releases = if ($payload -is [array]) { $payload } else { @($payload) }

    foreach ($release in $releases) {
        if ($PreferStable -and $release.prerelease) { continue }
        foreach ($asset in $release.assets) {
            if ($asset.name -match $Pattern) {
                return @{ Tag = $release.tag_name; Url = $asset.browser_download_url; Name = $asset.name }
            }
        }
    }
    return $null
}

function Get-LlamaServerPath {
    param([string]$Root)
    $exe = Join-Path $Root 'bin\llama.cpp\llama-server.exe'
    if (Test-Path -LiteralPath $exe) { return $exe }
    return $null
}

function Invoke-LocalHttp {
    <#
      调用本机接口，绕开系统代理。

      这台机器上跑着 Clash（127.0.0.1:7890），Invoke-RestMethod 会走系统代理，于是
      连 127.0.0.1 的请求也可能被代理拦下并返回 502——看起来像"服务挂了"，其实服务
      好得很。所以这里直接用 .NET HttpClient 并显式 UseProxy = $false，与 Python 侧
      的 trust_env=False 是同一个决定。

      返回 @{ Ok = $true/$false; Status = <int>; Body = <string> }
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [string]$Method = 'GET',
        [int]$TimeoutSec = 5
    )

    # Windows PowerShell 5.1 跑在 .NET Framework 上，System.Net.Http 不是默认加载的
    # 程序集：不先 Add-Type 就会报 "Cannot find type [System.Net.Http.HttpClientHandler]"。
    if (-not ('System.Net.Http.HttpClient' -as [type])) {
        Add-Type -AssemblyName System.Net.Http -ErrorAction SilentlyContinue
    }

    $handler = New-Object System.Net.Http.HttpClientHandler
    $handler.UseProxy = $false
    $client = New-Object System.Net.Http.HttpClient($handler)
    $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSec)
    try {
        $request = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::$Method, $Uri)
        $response = $client.SendAsync($request).GetAwaiter().GetResult()
        $body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        return @{ Ok = $response.IsSuccessStatusCode; Status = [int]$response.StatusCode; Body = $body }
    } catch {
        return @{ Ok = $false; Status = 0; Body = $_.Exception.Message }
    } finally {
        $client.Dispose()
        $handler.Dispose()
    }
}

function Show-Vram {
    <#
      打印 GPU 显存占用。调 --n-gpu-layers 时必须看这个数字，别猜。
    #>
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) { Write-Warn2 '未找到 nvidia-smi，无法读取显存'; return }
    try {
        $lines = & nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
        foreach ($line in $lines) { Write-Host "    GPU: $line" -ForegroundColor DarkGray }
    } catch {
        Write-Warn2 "读取显存失败：$($_.Exception.Message)"
    }
}

function Get-ModelsRoot {
    param([string]$Root)
    $python = Get-Python -Root $Root
    if (-not $python) { throw 'Python is required to resolve the configured model directory' }
    $code = "import sys; from pathlib import Path; sys.path.insert(0,str(Path(sys.argv[1])/'src')); from core.config import load_config; print(load_config(Path(sys.argv[1])).paths.models_dir)"
    $result = & $python -c $code $Root
    if ($LASTEXITCODE -ne 0) { throw 'Could not read paths.models_dir' }
    return ([string]($result | Select-Object -Last 1)).Trim()
}
