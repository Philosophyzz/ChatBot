# =============================================================================
# 一键安装：Python 运行环境 + 依赖 + llama.cpp 推理引擎 + ffmpeg
#
# 用法（在项目目录下打开 PowerShell 执行）：
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1
#
# 可选开关：
#   -Root D:\Harness\ChatBot   指定项目根目录（默认自动推断）
#   -SkipLlama                 不下载 llama.cpp（已装过或想用 Ollama）
#   -NoSpeech                  不安装语音依赖（省约 300MB）
#   -Force                     强制继续 / 重建虚拟环境
#
# 全部内容写入项目根目录：venvs\（Python 环境）、bin\（推理引擎）、models\（权重）
#
# 维护提示：本文件必须保存为「UTF-8 with BOM」。Windows PowerShell 5.1 对没有 BOM
# 的 .ps1 会按系统 ANSI 代码页解码，中文会变成乱码 —— 本项目真实踩过这个坑。
# 需要修改时请整体重写文件，不要用 Get-Content -Raw / Set-Content 往返读写。
# =============================================================================

[CmdletBinding()]
param(
    [string]$Root,
    [switch]$SkipLlama,
    [switch]$NoSpeech,
    [switch]$Force
)

. (Join-Path $PSScriptRoot 'common.ps1')

$ProjectRoot = Get-ConfigRoot -Root $Root
Write-Host ''
Write-Host '============================================================' -ForegroundColor White
Write-Host ' 本地 LLM 聊天机器人 —— 环境安装' -ForegroundColor White
Write-Host " 项目根目录: $ProjectRoot" -ForegroundColor White
Write-Host '============================================================' -ForegroundColor White

# --- 0. 磁盘空间检查 ---------------------------------------------------------
Write-Step '检查磁盘空间'
$free = Get-FreeSpaceGB -Path $ProjectRoot
Write-Host "    可用空间: $free GB"
if ($free -lt 40) {
    Write-Warn2 '剩余空间不足 40GB。27B 模型 + 语音模型约需 25GB，建议先清理。'
    if (-not $Force) {
        $answer = Read-Host '    仍要继续吗？(y/N)'
        if ($answer -notmatch '^[yY]') { Write-Err2 '已取消'; exit 1 }
    }
} else {
    Write-Ok "空间充足（$free GB）"
}

# --- 1. 目录结构 -------------------------------------------------------------
Write-Step '创建目录结构'
$dirs = @('config', 'data', 'logs', 'bin', 'bin\llama.cpp', 'venvs', 'models',
          'models\gguf', 'models\tts', 'models\voices', 'models\cache', 'models\whisper', 'web\assets')
foreach ($dir in $dirs) {
    $path = Join-Path $ProjectRoot $dir
    if (-not (Test-Path -LiteralPath $path)) { New-Item -ItemType Directory -Force -Path $path | Out-Null }
}
Write-Ok "已就绪 $($dirs.Count) 个目录"

# --- 2. Python 运行环境 ------------------------------------------------------
# 优先复用已经存在的 conda 环境 chatbot（推荐用法：所有依赖都在 D 盘，
# 见 README「运行环境」）；没有才建 venvs\main。两条路都会装上同一份
# requirements.txt / requirements-speech.txt，不会出现依赖漂移。
Write-Step '准备 Python 运行环境'
$condaPython = Get-CondaPython -Name 'chatbot' -Root $ProjectRoot
$venvDir = Join-Path $ProjectRoot 'venvs\main'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'

if ($condaPython -and -not $Force) {
    $venvPython = $condaPython
    $pythonLabel = 'conda 环境 chatbot'
    Write-Ok "复用 $pythonLabel：$condaPython"
} else {

if ($Force -and (Test-Path -LiteralPath $venvDir)) {
    Write-Warn2 '按 -Force 要求删除旧虚拟环境'
    Remove-Item -LiteralPath $venvDir -Recurse -Force
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    $hasUv = Test-CommandExists 'uv'
    if ($hasUv) {
        Write-Host '    使用 uv 创建（更快）'
        # --seed 很关键：uv 默认不往 venv 里装 pip，而这个脚本后面要用 pip 装依赖。
        # 不加 --seed 会得到 "No module named pip"。
        & uv venv --python 3.11 --seed $venvDir
        if (-not (Test-Path -LiteralPath $venvPython)) {
            Write-Warn2 'uv --seed 创建失败，改用不带 pip 的方式（稍后自动补齐）'
            & uv venv --python 3.11 $venvDir
        }
        if (-not (Test-Path -LiteralPath $venvPython)) {
            Write-Warn2 'uv 创建失败，回退到系统 python venv'
            $hasUv = $false
        }
    }
    if (-not $hasUv) {
        $systemPython = Get-SystemPython
        if (-not $systemPython) {
            Write-Err2 '未找到 Python。请先安装 Python 3.10+（winget install Python.Python.3.12）'
            exit 1
        }
        Write-Host "    使用 $systemPython 创建虚拟环境"
        & $systemPython -m venv $venvDir
    }
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Err2 '虚拟环境创建失败'
    exit 1
}
$pythonLabel = 'venvs\main'
}
$version = & $venvPython --version
Write-Ok "$version @ $venvPython（$pythonLabel）"

# --- 2b. 确保 pip 可用 ------------------------------------------------------
# 旧版 uv venv 不带 --seed、部分精简 Python 发行版也不带 ensurepip，
# 所以这里主动补一次，让后面无论走 uv 还是 pip 都有退路。
function Test-VenvPip {
    param([string]$Python)
    & $Python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('pip') else 1)" 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Install-PipIntoVenv {
    param([string]$Python)
    Write-Warn2 '虚拟环境里没有 pip，正在补齐…'
    if (Test-CommandExists 'uv') {
        & uv pip install --python $Python pip --quiet 2>&1 | Out-Null
        if (Test-VenvPip -Python $Python) { return $true }
    }
    & $Python -m ensurepip --upgrade 2>&1 | Out-Null
    if (Test-VenvPip -Python $Python) { return $true }
    & $Python -m ensurepip --default-pip 2>&1 | Out-Null
    return (Test-VenvPip -Python $Python)
}

if (-not (Test-VenvPip -Python $venvPython)) {
    if (-not (Install-PipIntoVenv -Python $venvPython)) {
        Write-Warn2 '无法补上 pip；将改用 uv pip 直接安装依赖（不影响使用）'
    }
}
if (Test-VenvPip -Python $venvPython) {
    Write-Ok 'pip 可用'
    & $venvPython -m pip install --upgrade pip --quiet 2>&1 | Out-Null
} else {
    Write-Host '    提示：该环境无 pip，依赖将通过 uv 安装'
}

# --- 3. Python 依赖 ---------------------------------------------------------
Write-Step '安装 Python 依赖（约 200MB，首次需要几分钟）'

# 依赖清单只有一份：requirements.txt（+ 语音扩展 requirements-speech.txt）。
# 以前把清单内联在这个脚本里，结果手工建环境（例如 conda）时没人知道要装什么，
# 两边迟早漂移。
$requirementFiles = @(Join-Path $ProjectRoot 'requirements.txt')
if (-not $NoSpeech) {
    $requirementFiles += Join-Path $ProjectRoot 'requirements-speech.txt'
}
foreach ($file in $requirementFiles) {
    if (-not (Test-Path -LiteralPath $file)) {
        Write-Err2 "缺少依赖清单：$file"
        exit 1
    }
}
$requirements = @()
foreach ($file in $requirementFiles) {
    $requirements += (Get-Content -LiteralPath $file -Encoding UTF8 |
        Where-Object { $_.Trim() -and -not $_.Trim().StartsWith('#') } |
        ForEach-Object { $_.Trim() })
}
$names = ($requirementFiles | ForEach-Object { Split-Path -Leaf $_ }) -join ' + '
Write-Host "    清单: $names" -ForegroundColor DarkGray

# 使用国内镜像会快很多；失败自动回退官方源。
# 注意 uv 与 pip 的镜像参数不同：uv 用 UV_DEFAULT_INDEX 环境变量，pip 用 -i。
$indexes = @(
    'https://pypi.tuna.tsinghua.edu.cn/simple',
    'https://mirrors.aliyun.com/pypi/simple',
    ''
)
$installed = $false
$useUv = (Test-CommandExists 'uv')

foreach ($index in $indexes) {
    $label = if ($index) { $index } else { 'PyPI 官方' }
    Write-Host "    源: $label"

    if ($useUv) {
        if ($index) { $env:UV_DEFAULT_INDEX = $index } else { Remove-Item Env:\UV_DEFAULT_INDEX -ErrorAction SilentlyContinue }
        & uv pip install --python $venvPython @requirements
        if ($LASTEXITCODE -eq 0) { $installed = $true; break }
        Write-Warn2 'uv 安装失败，改用 pip 重试'
        $useUv = $false
    }

    $pipArgs = @('-m', 'pip', 'install') + $requirements
    if ($index) { $pipArgs += @('-i', $index, '--trusted-host', ([Uri]$index).Host) }
    & $venvPython @pipArgs
    if ($LASTEXITCODE -eq 0) { $installed = $true; break }
    Write-Warn2 '该源安装失败，尝试下一个'
}
if (-not $installed) {
    Write-Err2 'Python 依赖安装失败。可手动执行其中一条：'
    Write-Host "    uv pip install --python `"$venvPython`" $($requirements -join ' ')"
    Write-Host "    $venvPython -m pip install $($requirements -join ' ')"
    exit 1
}
Write-Ok 'Python 依赖安装完成'

# 立刻验证：装完能不能真的 import。只报「安装成功」而不验证，等于把问题留给用户。
# 校验逻辑放在独立 .py 文件里，不用 python -c 内联脚本 —— 内联脚本里的非 ASCII
# 内容会在 PowerShell 与 Python 之间经历一次编码往返（命令行按 ANSI 代码页传递），
# 中文被破坏后 Python 直接 SyntaxError。本项目真实踩过这个坑。
Write-Step '校验依赖可否导入'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
& $venvPython (Join-Path $ProjectRoot 'tests\verify_env.py')
if ($LASTEXITCODE -ne 0) {
    Write-Err2 '依赖校验未通过（修复命令见上）。修复后请重新运行本脚本。'
    if (-not $Force) { exit 1 }
    Write-Warn2 '按 -Force 要求继续执行'
}

# --- 4. llama.cpp 推理引擎 ---------------------------------------------------
if (-not $SkipLlama) {
    Write-Step '获取 llama.cpp（本地推理引擎）'
    $llamaDir = Join-Path $ProjectRoot 'bin\llama.cpp'
    $serverExe = Join-Path $llamaDir 'llama-server.exe'

    if (Test-Path -LiteralPath $serverExe) {
        Write-Ok "已存在: $serverExe"
    } else {
        # ggml-org 的 Windows 资产分两类，**两类都要**，缺一类会静默退回 CPU：
        #   llama-bNNNNN-bin-win-cuda-13.4-x64.zip  可执行文件（含 ggml-cuda.dll）
        #   cudart-llama-bin-win-cuda-13.4-x64.zip  CUDA 运行时 DLL（cudart64_*.dll、
        #                                           cublas64_*.dll、cublasLt64_*.dll 等）
        # 只装前者时 `llama-server --list-devices` 输出 "(none)"，模型全部跑在 CPU 上，
        # 而且**不报任何错** —— 本项目真实踩过：27B 因此只有 3.6 tok/s，
        # 而且调 --n-gpu-layers 完全无效（层数变了显存和速度都不动）。
        # arm64 包必须排除（本机是 x64）。
        $plan = @(
            @{ Name = 'CUDA 13.4 x64'; Bin = '^llama-b\d+-bin-win-cuda-13\.\d+-x64\.zip$'; Cudart = '^cudart-llama-bin-win-cuda-13\.4-x64\.zip$' },
            @{ Name = 'CUDA 12.4 x64'; Bin = '^llama-b\d+-bin-win-cuda-12\.\d+-x64\.zip$'; Cudart = '^cudart-llama-bin-win-cuda-12\.4-x64\.zip$' },
            @{ Name = 'Vulkan x64';    Bin = '^llama-b\d+-bin-win-vulkan-x64\.zip$';        Cudart = $null },
            @{ Name = 'CPU x64';       Bin = '^llama-b\d+-bin-win-cpu-x64\.zip$';           Cudart = $null }
        )

        $binAsset = $null
        $cudartAsset = $null
        $chosenName = ''
        foreach ($option in $plan) {
            $candidate = Get-GithubReleaseAsset -Repo 'ggml-org/llama.cpp' -Pattern $option.Bin -MaxReleases 25
            if ($candidate) {
                $binAsset = $candidate
                $chosenName = $option.Name
                if ($option.Cudart) {
                    # 运行时包必须来自同一个 release，否则 DLL 与 ggml-cuda.dll 的 ABI 可能不匹配。
                    $cudartAsset = Get-GithubReleaseAsset -Repo 'ggml-org/llama.cpp' `
                        -Pattern $option.Cudart -MaxReleases 25 -Tag $candidate.Tag
                }
                break
            }
        }

        if (-not $binAsset) {
            Write-Warn2 '未能从 GitHub 自动获取 llama.cpp 构建'
            Write-Host '    可手动下载后解压到 bin\llama.cpp\：'
            Write-Host '    https://github.com/ggml-org/llama.cpp/releases'
            Write-Host '    CUDA 版需要**两个**包：llama-bNNNNN-bin-win-cuda-13.4-x64.zip 与'
            Write-Host '    cudart-llama-bin-win-cuda-13.4-x64.zip（后者提供 cudart64_*.dll 等运行时）'
        } else {
            Write-Host "    选用 $chosenName 构建: $($binAsset.Tag)"
            if ($chosenName -like 'CUDA*' -and -not $cudartAsset) {
                Write-Warn2 '同一 release 里没找到配套的 cudart 运行时包，GPU 可能无法启用'
            }

            $targets = @($binAsset)
            if ($cudartAsset) { $targets += $cudartAsset }
            $extracted = 0

            foreach ($target in $targets) {
                $zip = Join-Path $env:TEMP $target.Name
                Write-Host "    下载 $($target.Name)"
                $ok = Invoke-Download -Uri $target.Url -OutFile $zip `
                    -MirrorPrefixes @('https://ghfast.top/', 'https://gh-proxy.com/')
                if (-not $ok) {
                    Write-Warn2 "下载失败：$($target.Name)"
                    continue
                }
                Write-Host '    解压中…'
                $tmp = Join-Path $env:TEMP ("llama-unzip-" + [Guid]::NewGuid().ToString('N'))
                Expand-Archive -LiteralPath $zip -DestinationPath $tmp -Force

                # 两个包都解压到同一个目标目录，这样 ggml-cuda.dll 与它需要的运行时 DLL 会在一起。
                # 压缩包内可能是平铺的，也可能带一层子目录；cudart 包里没有 exe，改用首个 DLL 定位。
                $source = $null
                $found = Get-ChildItem -LiteralPath $tmp -Recurse -Filter 'llama-server.exe' |
                    Select-Object -First 1
                if ($found) {
                    $source = $found.Directory.FullName
                } else {
                    $anyDll = Get-ChildItem -LiteralPath $tmp -Recurse -Filter '*.dll' | Select-Object -First 1
                    if ($anyDll) { $source = $anyDll.Directory.FullName }
                }
                if ($source) {
                    Copy-Item -Path (Join-Path $source '*') -Destination $llamaDir -Recurse -Force
                    $extracted++
                } else {
                    Write-Warn2 "压缩包内容无法识别：$($target.Name)"
                }
                Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
                Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
            }

            if (Test-Path -LiteralPath $serverExe) {
                Write-Ok "已安装: $serverExe（解压了 $extracted 个包）"
            } else {
                Write-Err2 '解压后仍未找到 llama-server.exe，请手动安装'
            }
        }
    }

    if (Test-Path -LiteralPath $serverExe) {
        # 自检：确认能启动、依赖 DLL 齐全。
        # 注意：llama-server 把版本信息写到 stderr（不是 stdout），而 PowerShell 在
        # $ErrorActionPreference='Stop' 下会把原生命令的 stderr 当作终止错误 ——
        # 于是「启动成功」会被误报成失败。这里临时关掉 Stop，只看退出码与文本。
        Write-Host '    自检 llama-server --version …'
        $savedPref = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $output = & $serverExe --version 2>&1
            $code = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $savedPref
        }
        $text = ($output | Out-String)
        if ($code -eq 0 -and $text -match 'version') {
            $versionLine = ($output | Where-Object { "$_" -match 'version:' } | Select-Object -First 1)
            if (-not $versionLine) { $versionLine = '退出码 0' }
            Write-Ok "llama-server 可用（$("$versionLine".Trim())）"
        } else {
            Write-Warn2 "llama-server 启动自检未通过（退出码 $code）"
            ($output | Select-Object -First 4) | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray }
            Write-Host '    常见原因：缺少 Visual C++ 运行库（winget install Microsoft.VCRedist.2015+.x64），'
            Write-Host '              或下载的构建与系统不匹配（例如下到了 arm64 包）。'
        }

        # GPU 可用性检查 —— 这是最容易被忽略的一步。
        # 只装了 llama-b*-bin-win-cuda-*.zip、没装 cudart 运行时的典型症状是：
        # llama-server 能正常启动，但 --list-devices 输出 "(none)"，于是所有计算静默
        # 退回 CPU。表现是"能跑但极慢"（27B 只有 3.6 tok/s），而且调 --n-gpu-layers
        # 完全没有效果。不主动检查的话，用户会以为是模型太大或参数没调好。
        Write-Host '    检查 GPU 后端 …'
        $savedPref = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $devOut = & $serverExe --list-devices 2>&1
        } finally {
            $ErrorActionPreference = $savedPref
        }
        $devText = ($devOut | Out-String)
        $gpuLines = $devOut | Where-Object { "$_" -match 'CUDA|Vulkan|ROCm|Metal' }
        if ($gpuLines) {
            Write-Ok "GPU 后端可用：$(($gpuLines | Select-Object -First 1).ToString().Trim())"
        } else {
            Write-Warn2 'llama-server 看不到任何 GPU 设备，模型将全部跑在 CPU 上（会非常慢）。'
            Write-Host '    若是 CUDA 构建，通常说明缺少 CUDA 运行时 DLL。修复方式：' -ForegroundColor Yellow
            Write-Host '      1) 确认这两个包都解压到了 bin\llama.cpp\：'
            Write-Host '         llama-bNNNNN-bin-win-cuda-13.4-x64.zip'
            Write-Host '         cudart-llama-bin-win-cuda-13.4-x64.zip'
            Write-Host '      2) 或改用不依赖 CUDA 运行时的 Vulkan 构建（性能略低但省事）'
            Write-Host '      重新运行本脚本并加 -Force 可强制重新下载安装。'
        }
    }
} else {
    Write-Warn2 '已跳过 llama.cpp 下载（-SkipLlama）'
}

# --- 5. ffmpeg（可选但推荐）-------------------------------------------------
Write-Step '检查 ffmpeg（浏览器录音转码用）'
if (Test-CommandExists 'ffmpeg') {
    Write-Ok "已安装: $((Get-Command ffmpeg).Source)"
} else {
    Write-Warn2 '未安装 ffmpeg'
    Write-Host '    不影响：前端会直接上传 16kHz WAV，无需转码。'
    Write-Host '    如需支持任意格式音频，可执行：winget install Gyan.FFmpeg'
}

# --- 6. 提示后续步骤 ---------------------------------------------------------
Write-Host ''
Write-Host '============================================================' -ForegroundColor Green
Write-Host ' 环境安装完成' -ForegroundColor Green
Write-Host '============================================================' -ForegroundColor Green
Write-Host ''
Write-Host '下一步（按顺序执行）：' -ForegroundColor White
Write-Host '  1) 下载模型（约 18GB，可断点续传）：'
Write-Host '     powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1 -Mirror'
Write-Host ''
Write-Host '  2) 先跑一次自检（不需要模型，验证代码链路）：'
Write-Host '     powershell -ExecutionPolicy Bypass -File scripts\verify.ps1'
Write-Host ''
Write-Host '  3) 启动（自动拉起模型服务 + 网页界面）：'
Write-Host '     powershell -ExecutionPolicy Bypass -File scripts\start-all.ps1'
Write-Host ''

# 显式成功退出码：PowerShell 会让脚本继承上一条原生命令的退出码，
# 于是「其实成功了」的安装过程可能对外报 1，被自动化误判为失败。
exit 0
