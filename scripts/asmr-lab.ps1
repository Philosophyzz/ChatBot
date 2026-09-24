param([ValidateSet('train','generate','experiment','compare')][string]$Action = 'train', [Parameter(ValueFromRemainingArguments=$true)][string[]]$ExtraArgs)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$ProjectRoot = Get-ConfigRoot
$python = Get-Python -Root $ProjectRoot
& $python (Join-Path $ProjectRoot 'tools/asmr_lab.py') $Action @ExtraArgs
exit $LASTEXITCODE
