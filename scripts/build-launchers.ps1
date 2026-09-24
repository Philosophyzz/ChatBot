# 兼容旧构建命令：现在只构建根目录的一个启动程序。
[CmdletBinding()]
param([string]$Root, [switch]$Clean)
& (Join-Path $PSScriptRoot 'build-pet.ps1') -Root $Root -Clean:$Clean
exit $LASTEXITCODE
