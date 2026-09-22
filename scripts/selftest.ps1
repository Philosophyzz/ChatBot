# =============================================================================
# 全量验证（一次跑完，输出写入 logs\selftest.txt）
#
# 与 verify.ps1 的区别：verify.ps1 面向用户做环境体检；本脚本面向交付验收，
# 依次执行 编译 → 清理 → 零依赖自检 → 单元测试 → 配置检查 → 冒烟启动，
# 任何一步失败都会在结尾汇总。
#
# 注意：本文件含中文，必须保存为「UTF-8 with BOM」，否则 Windows PowerShell 5.1
# 会按 GBK 解码，导致脚本里的中文字符串损坏（这一坑在本项目里真实踩过）。
# =============================================================================

[CmdletBinding()]
param([string]$Root)

$ErrorActionPreference = 'Continue'
$ProjectRoot = if ($Root) { $Root } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }

# 解释器解析交给 common.ps1：它知道 conda 环境 chatbot 与 venvs\main 的优先级，
# 这里硬编码路径会在换了运行环境之后悄悄跑到另一个 Python 上（并给出假的绿色结果）。
. (Join-Path $PSScriptRoot 'common.ps1')
# common.ps1 是给交互式脚本用的：它把 ErrorActionPreference 设成 Stop 并打开 StrictMode。
# 本脚本要**故意**把子进程 stderr 收进变量（很多检查会往 stderr 写日志，例如
# "unknown persona requested"），Stop 会把这种正常输出当成致命错误，整个自检在第 2 步
# 就断掉。所以 dot-source 之后必须显式恢复成本脚本需要的宽松模式。
$ErrorActionPreference = 'Continue'
Set-StrictMode -Off
$py = Get-Python -Root $ProjectRoot
if (-not $py) { Write-Host '找不到 python（conda 环境 chatbot 或 venvs\main）' -ForegroundColor Red; exit 1 }
Write-Host "  解释器: $py" -ForegroundColor DarkGray

# 输出编码必须显式设置：common.ps1 里已经设过一遍，但本脚本可能在没有它的环境下被
# 拷走单独运行，所以这里再设一次。
try {
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [Console]::OutputEncoding = $utf8NoBom
    $OutputEncoding = $utf8NoBom
    $PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
    $PSDefaultParameterValues['Set-Content:Encoding'] = 'utf8'
} catch {
    # 某些宿主不允许改编码，忽略
}

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$log = Join-Path $ProjectRoot 'logs\selftest.txt'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $log) | Out-Null

$results = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]
$steps = @()

function Invoke-Step {
    param([string]$Name, [string[]]$CommandArgs, [string]$WorkDir)
    Write-Host ''
    Write-Host "==> $Name" -ForegroundColor Cyan
    $previous = Get-Location
    if ($WorkDir) { Set-Location -LiteralPath $WorkDir }
    try {
        $output = & $py @CommandArgs 2>&1 | Out-String
        $code = $LASTEXITCODE
    } finally {
        Set-Location -LiteralPath $previous
    }
    $script:results.Add("===== $Name (exit=$code) =====`n$output")
    Write-Host $output
    $script:steps += [pscustomobject]@{ Name = $Name; Exit = $code }
    return $code
}

function Test-ModelRepoReachability {
    <#
      模型仓库清单校验需要访问 HuggingFace，属于**外部依赖**，不应让整份自检失败。
      先走国内镜像；仍不通就认定为网络原因并跳过（记为警告而非失败）。
      返回 @{ Status = 'ok' | 'failed' | 'skipped'; Args = @() }
    #>
    $probe = Join-Path $ProjectRoot 'tests\verify_models.py'
    $mirrorArgs = @('-B', $probe, '--mirror')
    Write-Host ''
    Write-Host '==> 5) 模型可用性（镜像源）' -ForegroundColor Cyan
    $output = & $py @mirrorArgs 2>&1 | Out-String
    $code = $LASTEXITCODE
    $script:results.Add("===== 5) 模型可用性 mirror (exit=$code) =====`n$output")
    Write-Host $output
    if ($code -eq 0) {
        $script:steps += [pscustomobject]@{ Name = '5) 模型可用性'; Exit = 0 }
        return
    }

    # 镜像也失败：区分「仓库真的没了」和「网络不通」。
    if ($output -match 'urlopen error|WinError 10060|timed out|Max retries|Temporary failure') {
        Write-Warn2 '无法访问模型仓库（网络原因），跳过本项校验'
        $script:warnings.Add('模型清单校验（网络不可达）') | Out-Null
        $script:steps += [pscustomobject]@{ Name = '5) 模型可用性（网络不通，已跳过）'; Exit = 0 }
    } else {
        $script:steps += [pscustomobject]@{ Name = '5) 模型可用性'; Exit = $code }
    }
}

Invoke-Step -Name '1) 编译检查' -CommandArgs @(
    '-B', '-c',
    "import pathlib,sys`nroot=pathlib.Path(r'$ProjectRoot')`nn=0;bad=[]`nfor base in ('src','tests','scripts'):`n    for p in sorted((root/base).rglob('*.py')):`n        n+=1`n        try: compile(p.read_text(encoding='utf-8'),str(p),'exec')`n        except SyntaxError as e: bad.append(f'{p.relative_to(root)}:{e.lineno}:{e.msg}')`nprint(f'compile {n} files: '+('OK' if not bad else 'FAILED'))`n[print('  '+b) for b in bad]"
) | Out-Null

Invoke-Step -Name '2) 零依赖自检' -CommandArgs @('-B', (Join-Path $ProjectRoot 'tests\smoke_stdlib.py')) | Out-Null

Invoke-Step -Name '3) 单元测试' -CommandArgs @(
    '-B', '-m', 'pytest', 'tests', '-q', '--no-header', '-p', 'no:cacheprovider', '--tb=short', '-rf'
) -WorkDir $ProjectRoot | Out-Null

Invoke-Step -Name '4) 配置检查' -CommandArgs @('-B', (Join-Path $ProjectRoot 'scripts\check_config.py')) | Out-Null

Test-ModelRepoReachability

Invoke-Step -Name '6) 接口冒烟（模拟模型，不占显存）' -CommandArgs @(
    '-B', (Join-Path $ProjectRoot 'tests\smoke_http.py')
) | Out-Null

$results -join "`n" | Set-Content -LiteralPath $log -Encoding UTF8

Write-Host ''
Write-Host '============================================================' -ForegroundColor White
$failed = $steps | Where-Object { $_.Exit -ne 0 }
foreach ($step in $steps) {
    $color = if ($step.Exit -eq 0) { 'Green' } else { 'Red' }
    Write-Host ("  {0,-34} exit={1}" -f $step.Name, $step.Exit) -ForegroundColor $color
}
Write-Host "  日志: $log" -ForegroundColor DarkGray
Write-Host '============================================================' -ForegroundColor White
if ($failed) { exit 1 }
exit 0
