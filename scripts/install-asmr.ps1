[CmdletBinding()]
param([switch]$SkipDownloads)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$ProjectRoot = Get-ConfigRoot
$python = Get-Python -Root $ProjectRoot
& $python -c 'import torch,transformers,scipy,soundfile; assert torch.__version__ == "2.8.0+cu128", "Expected the existing chatbot CUDA 12.8 runtime"'
if ($LASTEXITCODE -ne 0) { throw 'Base runtime check failed; do not downgrade the active chat environment.' }
$environment = Join-Path $ProjectRoot 'venvs/asmr'
if (-not (Test-Path (Join-Path $environment 'Scripts/python.exe'))) {
    & $python -m venv --system-site-packages $environment
    if ($LASTEXITCODE -ne 0) { throw 'ASMR environment creation failed' }
}
$asmrPython = Join-Path $environment 'Scripts/python.exe'
& $asmrPython -m pip install --no-deps -r (Join-Path $ProjectRoot 'requirements-asmr.txt')
if ($LASTEXITCODE -ne 0) { throw 'ASMR dependencies failed' }
& $asmrPython -m pip install torchvision==0.23.0+cu128 --no-deps --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'Matching torchvision install failed' }
$vendor = Join-Path $ProjectRoot 'vendor/AudioX'
if (-not (Test-Path $vendor)) {
    & git clone https://github.com/ZeyueT/AudioX.git $vendor
    if ($LASTEXITCODE -ne 0) { throw 'AudioX source download failed' }
    & git -C $vendor checkout 3bdfb7081636b9e62224039e37dadaa264dc781f
}
$revision = & git -C $vendor rev-parse HEAD
if ($revision -ne '3bdfb7081636b9e62224039e37dadaa264dc781f') { throw 'AudioX source revision differs from the tested version; preserve it and inspect before switching.' }
Push-Location $ProjectRoot
try {
    & $asmrPython -c 'import sys;sys.path.insert(0,"vendor/AudioX");from audiox.models.diffusion import create_diffusion_cond_from_config;print("AudioX imports OK")'
    if ($LASTEXITCODE -ne 0) { throw 'AudioX import check failed' }
    if (-not $SkipDownloads) {
        & $python scripts/download_asmr.py all
        if ($LASTEXITCODE -ne 0) { throw 'ASMR asset download failed' }
        & $python tools/asmr_dataset.py
        if ($LASTEXITCODE -ne 0) { throw 'ASMR dataset preparation failed' }
    }
} finally { Pop-Location }
