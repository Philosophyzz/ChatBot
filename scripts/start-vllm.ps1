[CmdletBinding()]
param([string]$Root, [string]$Tier)
. (Join-Path $PSScriptRoot 'common.ps1')
$ProjectRoot = Get-ConfigRoot -Root $Root
$python = Get-Python -Root $ProjectRoot
$arguments = @((Join-Path $PSScriptRoot 'vllm_control.py'), 'start')
if ($Tier) { $arguments += @('--model', $Tier) }
& $python @arguments
exit $LASTEXITCODE
