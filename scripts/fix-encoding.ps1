# 修正脚本编码（改完脚本跑一下）
#
# 两条硬规则（tests\test_core.py 里有测试盯着）：
#   *.ps1 必须有 UTF-8 BOM —— Windows PowerShell 5.1 对没有 BOM 的文件按 ANSI(GBK) 解码，
#          中文会变成乱码，甚至把字符串引号吃掉导致解析失败。
#   *.py  必须没有 BOM —— Python 源码带 BOM 在某些工具链里会出问题。
#
# 很多编辑器/工具保存时会静默丢掉 BOM，所以这条命令值得在改完脚本后跑一次：
#   powershell -ExecutionPolicy Bypass -File scripts\fix-encoding.ps1
[CmdletBinding()]
param([string]$Root, [switch]$CheckOnly)

. (Join-Path $PSScriptRoot 'common.ps1')
$ProjectRoot = Get-ConfigRoot -Root $Root

$utf8Bom = [System.Text.UTF8Encoding]::new($true)
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$fixed = 0
$problems = 0

Write-Step '检查 PowerShell 脚本（需要 BOM）'
foreach ($file in Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'scripts') -Filter '*.ps1' -File) {
    $bytes = [System.IO.File]::ReadAllBytes($file.FullName)
    $hasBom = ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF)
    if ($hasBom) { continue }
    $problems++
    if ($CheckOnly) { Write-Warn2 "$($file.Name) 缺少 BOM"; continue }
    $text = [System.IO.File]::ReadAllText($file.FullName, $utf8NoBom)
    [System.IO.File]::WriteAllText($file.FullName, $text, $utf8Bom)
    Write-Ok "已补 BOM：$($file.Name)"
    $fixed++
}
if ($problems -eq 0) { Write-Ok '全部脚本都有 BOM' }

Write-Step '检查 Python 源码（不应有 BOM）'
$pyFiles = @()
foreach ($dir in @('src', 'tests', 'tools')) {
    $path = Join-Path $ProjectRoot $dir
    if (Test-Path -LiteralPath $path) {
        $pyFiles += Get-ChildItem -LiteralPath $path -Filter '*.py' -Recurse -File
    }
}
$pyProblems = 0
foreach ($file in $pyFiles) {
    $bytes = [System.IO.File]::ReadAllBytes($file.FullName)
    if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
        $pyProblems++
        if ($CheckOnly) { Write-Warn2 "$($file.Name) 不该有 BOM"; continue }
        $text = [System.IO.File]::ReadAllText($file.FullName, $utf8Bom)
        [System.IO.File]::WriteAllText($file.FullName, $text, $utf8NoBom)
        Write-Ok "已去掉 BOM：$($file.Name)"
        $fixed++
    }
}
if ($pyProblems -eq 0) { Write-Ok "检查了 $($pyFiles.Count) 个 Python 文件，都没有 BOM" }

Write-Host ''
if ($CheckOnly) {
    if ($problems + $pyProblems -gt 0) { Write-Err2 "发现 $($problems + $pyProblems) 个编码问题"; exit 1 }
    Write-Ok '编码检查通过'
} else {
    if ($fixed -gt 0) { Write-Ok "共修正 $fixed 个文件" } else { Write-Ok '无需修正' }
}
