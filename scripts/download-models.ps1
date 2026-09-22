# =============================================================================
# 下载模型权重（全部落在项目 models\ 目录，即 D 盘）
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1 -Tier balanced-14b -Mirror
#   powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1 -List
#
# 参数：
#   -Tier <id>     对话模型档位（quality-27b / balanced-14b / fast-9b）
#   -Support       同时下载嵌入与重排模型（记忆检索质量更好）
#   -Mirror        使用 hf-mirror.com 镜像（国内网络推荐）
#   -All           下载全部档位
# =============================================================================

[CmdletBinding()]
param(
    [string]$Root,
    [string]$Tier,
    [switch]$Support,
    [switch]$Mirror,
    [switch]$All,
    [switch]$List
)

. (Join-Path $PSScriptRoot 'common.ps1')

$ProjectRoot = Get-ConfigRoot -Root $Root
$python = Get-Python -Root $ProjectRoot
if (-not $python) {
    Write-Err2 '找不到 Python。请先运行 scripts\install.ps1'
    exit 1
}

Write-Step '磁盘空间检查'
$free = Get-FreeSpaceGB -Path $ProjectRoot
Write-Host "    $ProjectRoot 可用: $free GB"

$arguments = @((Join-Path $PSScriptRoot 'download_models.py'))
if ($List) { $arguments += '--list' }
if ($Tier) { $arguments += @('--tier', $Tier) }
if ($All) { $arguments += '--all-tiers' }
if ($Support) { $arguments += '--support' }
if ($Mirror) { $arguments += '--mirror' }

# 让 Python 侧也用 UTF-8 输出，避免中文乱码
$env:PYTHONIOENCODING = 'utf-8'
& $python @arguments
$code = $LASTEXITCODE

if ($code -eq 0 -and -not $List) {
    Write-Step '校验下载结果'
    $ggufDir = Join-Path $ProjectRoot 'models\gguf'
    if (Test-Path -LiteralPath $ggufDir) {
        $files = Get-ChildItem -LiteralPath $ggufDir -Filter '*.gguf' -ErrorAction SilentlyContinue
        if ($files) {
            foreach ($file in $files) {
                $size = [math]::Round($file.Length / 1GB, 2)
                Write-Ok "$($file.Name)  $size GB"
            }
        } else {
            Write-Warn2 'models\gguf 下没有 .gguf 文件'
        }
    }
    Write-Host ''
    Write-Host ' 下一步：powershell -ExecutionPolicy Bypass -File scripts\verify.ps1' -ForegroundColor White
}
exit $code
