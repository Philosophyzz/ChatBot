# 把桌宠打包成单文件 exe（dist\ChatBotPet.exe）
#
#   powershell -ExecutionPolicy Bypass -File scripts\build-pet.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\build-pet.ps1 -OneDir     # 目录版，启动更快
#   powershell -ExecutionPolicy Bypass -File scripts\build-pet.ps1 -SkipIcon   # 不生成图标
#
# 说明：
#  * 用 PyInstaller 打包，产物在 dist\；构建缓存也放在项目里（build\），不碰 C 盘。
#  * exe 是**瘦客户端**：它需要本地服务在跑（8077）。桌宠右键菜单里有「启动本地服务」，
#    所以即使后端没起也能自己拉起来，不需要先去开终端。
#  * 图标由桌宠自己画出来（同样是代码生成，不引入美术资源）：先离屏渲染 PNG，再包成 ICO。
[CmdletBinding()]
param(
    [string]$Root,
    [string]$Name = 'ChatBotPet',
    [switch]$OneDir,
    [switch]$SkipIcon,
    [switch]$Console,
    [switch]$Clean
)

. (Join-Path $PSScriptRoot 'common.ps1')
$ProjectRoot = Get-ConfigRoot -Root $Root
$py = Get-Python -Root $ProjectRoot
if (-not $py) { Write-Err2 '找不到 Python（conda 环境 chatbot）'; exit 1 }

$entry = Join-Path $ProjectRoot 'run_pet.py'
if (-not (Test-Path -LiteralPath $entry)) { Write-Err2 "找不到入口文件 $entry"; exit 1 }

# PyInstaller 装在解释器所在环境里
& $py -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Step '安装 PyInstaller'
    & $py -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyinstaller
    if ($LASTEXITCODE -ne 0) { Write-Err2 'pyinstaller 安装失败'; exit 1 }
}

$iconPath = Join-Path $ProjectRoot 'data\pet.ico'
if (-not $SkipIcon) {
    Write-Step '生成图标（由桌宠自己渲染，无需美术资源）'
    & $py (Join-Path $ProjectRoot 'tools\make_pet_icon.py') --out $iconPath
    if (-not (Test-Path -LiteralPath $iconPath)) {
        Write-Warn2 '图标生成失败，继续用默认图标'
        $iconPath = ''
    }
} else {
    $iconPath = ''
}

if ($Clean) {
    foreach ($dir in 'build', 'dist') {
        $path = Join-Path $ProjectRoot $dir
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Recurse -Force; Write-Ok "已清理 $dir" }
    }
}

# 正在运行的桌宠会**锁住** dist\ChatBotPet.exe：PyInstaller 一路跑完，最后 os.remove
# 覆盖产物时才报"拒绝访问"，等于白等十分钟。所以打包前先把它关掉（这是你要重建的程序，
# 关掉它是预期行为，但要说清楚，别让人以为桌宠自己消失了）。
$runningPets = Get-Process -Name $Name -ErrorAction SilentlyContinue
if ($runningPets) {
    Write-Step "先关掉正在运行的桌宠（$($runningPets.Count) 个）—— 它锁着 dist\$Name.exe，不关会打包失败"
    $runningPets | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

Write-Step "打包 $Name.exe（单文件约 60~90MB，首次需要几分钟）"
$windowMode = if ($Console) { '--console' } else { '--windowed' }
$pyiArgs = @(
    '-m', 'PyInstaller',
    '--noconfirm',
    $windowMode,                                    # 默认不弹控制台窗口
    '--name', $Name,
    '--distpath', (Join-Path $ProjectRoot 'dist'),
    '--workpath', (Join-Path $ProjectRoot 'build'),
    '--specpath', (Join-Path $ProjectRoot 'build'),
    '--paths', (Join-Path $ProjectRoot 'src'),      # core / pet 包在 src 下
    '--hidden-import', 'pet',
    '--hidden-import', 'pet.window',
    '--hidden-import', 'pet.client',
    '--hidden-import', 'pet.audio',
    # pet.single_instance 是函数内 import（只在启动/切人设时才用到），显式声明免得
    # 打出来的 exe 少了单实例守卫 —— 那样双击两次就会开出两只一模一样的桌宠。
    '--hidden-import', 'pet.single_instance',
    # 同上：免提状态机 / 形象处理 / 本地设置都是运行中才 import 的。漏了它们，
    # 用户点「一直听着」或「更换形象」时会蹦 ModuleNotFoundError。
    '--hidden-import', 'pet.hands_free',
    '--hidden-import', 'pet.skin',
    '--hidden-import', 'pet.settings',
    # 录音和播放在函数里延迟导入，必须显式打进桌宠。
    '--hidden-import', 'sounddevice',
    '--hidden-import', 'soundfile',
    '--hidden-import', 'xml.etree.ElementTree',
    '--add-data', ((Join-Path $ProjectRoot 'src\pet\assets') + ';pet/assets'),
    # 不要用 --collect-submodules PySide6：它会把 Qt3D / WebEngine / Charts 等
    # 几百 MB 一起塞进包里（实测 268MB）。PyInstaller 自带的 PySide6 钩子已经会带上
    # 真正 import 到的模块与平台插件，配合下面的 --exclude 才能压到 60~90MB。
    '--exclude-module', 'PySide6.QtWebEngineCore',
    '--exclude-module', 'PySide6.QtWebEngineWidgets',
    '--exclude-module', 'PySide6.QtQml',
    '--exclude-module', 'PySide6.QtQuick',
    '--exclude-module', 'PySide6.QtMultimedia',
    '--exclude-module', 'PySide6.QtNetwork',
    '--exclude-module', 'PySide6.Qt3DCore',
    '--exclude-module', 'PySide6.QtCharts',
    '--exclude-module', 'PySide6.QtDataVisualization',
    '--exclude-module', 'matplotlib',
    '--exclude-module', 'tkinter'
)

# 桌宠是**瘦客户端**：它只通过 HTTP 跟 8077 上的服务说话，自己不做任何模型推理。
# 但 PyInstaller 会顺着 src\plugs\__init__.py（那里 import speech.stt / speech.tts 来注册
# 内置插件）一路分析下去，把 torch 整套 CUDA 运行库也打进来 —— 实测产出 3.3GB 的 exe
# （光 torch_cuda.dll 就 981MB），启动慢到没法用。这些包在桌宠进程里永远不会被 import，
# 显式排除即可把体积压回 ~130MB。
$heavyExcludes = @(
    'torch', 'torchvision', 'torchaudio', 'accelerate', 'transformers', 'diffusers',
    'faster_whisper', 'ctranslate2', 'tokenizers', 'sentencepiece', 'onnxruntime',
    'modelscope', 'datasets', 'oss2', 'aliyunsdkcore', 'huggingface_hub', 'safetensors',
    'cv2', 'numba', 'llvmlite', 'scipy', 'sklearn', 'pandas', 'pyarrow',
    'librosa', 'audioread', 'av', 'pydub',
    'indextts', 'pytorch_lightning', 'tensorboard', 'wandb',
    'sympy', 'networkx'
)
foreach ($module in $heavyExcludes) { $pyiArgs += @('--exclude-module', $module) }
Write-Ok "已排除 $($heavyExcludes.Count) 个桌宠用不到的重量级依赖（torch / whisper / opencv …）"

# conda 环境把 C 运行库放在 <env>\Library\bin，而不是 DLLs\，PyInstaller 不会自动带上。
# 每一种缺失都是一次"exe 双击没反应"：
#   libssl/libcrypto → `import _ssl: DLL load failed`（httpx 要走 https）
#   libexpat         → `import pyexpat: DLL load failed` —— 启动钩子 pyi_rth_pkgres 会
#                      加载 pkg_resources，它 import pyexpat，于是 exe 连界面都出不来
#                      （只在无控制台的 --windowed 版本上表现为"弹一个 Unhandled exception"）
#   ffi*             → `import _ctypes: DLL load failed`。conda 的 ffi 叫 ffi-8.dll，
#                      **不叫** libffi-8.dll，所以按 libffi* 根本找不到；桌宠的单实例守卫
#                      用 ctypes，一进去就炸 —— 当时自检没走到 ctypes，于是漏到了用户手上。
#                      现在自检里有 --import-check 专门盯这一整类缺失。
#   sqlite3/liblzma  → _sqlite3 / _lzma 同理，属于以后可能踩的坑，一起带上（几百 KB）
$envRoot = Split-Path -Parent $py
$libBin = Join-Path $envRoot 'Library\bin'
$runtimeDllPatterns = @(
    'libssl*.dll', 'libcrypto*.dll', 'libffi*.dll', 'ffi*.dll',
    'libexpat*.dll', 'sqlite3*.dll', 'liblzma*.dll', 'libbz2*.dll', 'zlib*.dll'
)
$bundledDlls = @()
foreach ($pattern in $runtimeDllPatterns) {
    foreach ($dll in Get-ChildItem -LiteralPath $libBin -Filter $pattern -File -ErrorAction SilentlyContinue) {
        $pyiArgs += @('--add-binary', "$($dll.FullName);.")
        $bundledDlls += $dll.Name
    }
}
if ($bundledDlls.Count -eq 0) { Write-Warn2 "在 $libBin 里没找到运行库 DLL，exe 很可能起不来" }
else { Write-Ok "已捆绑 $($bundledDlls.Count) 个运行库 DLL：$($bundledDlls -join ', ')" }
$mustHave = @(
    @{ Label = 'libexpat'; Patterns = @('libexpat*') },
    @{ Label = 'openssl'; Patterns = @('libssl*') },
    # conda 的 ffi 叫 ffi-8.dll（cpython 官方才叫 libffi-8.dll），两个都认 —— 否则这里
    # 每次构建都报一条假的"没找到 libffi"，常亮的警告等于没有警告。
    @{ Label = 'libffi'; Patterns = @('ffi-*', 'libffi*') }
)
foreach ($need in $mustHave) {
    # 循环变量别叫 $name：PowerShell 变量不区分大小写，那样会覆盖上面的 $Name 参数，
    # 构建最后就会去找 dist\zlib.dll.exe 这种根本不存在的产物（真踩过）。
    $hit = $bundledDlls | Where-Object { $dllName = $_; $need.Patterns | Where-Object { $dllName -like $_ } }
    if (-not $hit) { Write-Warn2 "没有找到 $($need.Patterns -join ' / ') —— exe 启动时可能报 DLL load failed" }
}

if ($OneDir) { $pyiArgs += '--onedir' } else { $pyiArgs += '--onefile' }
if ($iconPath) { $pyiArgs += @('--icon', $iconPath) }
$pyiArgs += $entry

& $py @pyiArgs
if ($LASTEXITCODE -ne 0) { Write-Err2 '打包失败'; exit 1 }

$exe = if ($OneDir) { Join-Path $ProjectRoot "dist\$Name\$Name.exe" } else { Join-Path $ProjectRoot "dist\$Name.exe" }
if (-not (Test-Path -LiteralPath $exe)) { Write-Err2 "没有找到产物 $exe"; exit 1 }

$mb = [math]::Round((Get-Item -LiteralPath $exe).Length / 1MB, 1)
# 体积是"有没有误打包推理依赖"的唯一外部信号：出过一次 3.3GB 的事故，所以这里硬性卡住，
# 免得又交给用户一个启动要半分钟的 exe。
$limitMb = 400
if (-not $OneDir -and $mb -gt $limitMb) {
    Write-Err2 "exe 有 $mb MB，超过 ${limitMb}MB 上限 —— 多半又把 torch/whisper 之类的推理依赖打进来了。"
    Write-Host '  看看是谁在膨胀：' -ForegroundColor Yellow
    Write-Host "    & `"$py`" tools\exe_size_report.py --top 30" -ForegroundColor Yellow
    Write-Host '  然后把对应的包补进本脚本的 $heavyExcludes。' -ForegroundColor Yellow
    exit 1
}

# 冒烟测试：真的启动一次 exe，让它渲染一帧并写报告。后端没跑时个别检查会失败，那是正常
# 的 —— 这里只验证"exe 能起来、Qt 能加载、能把结果写到文件"。
#
# 为什么不能只写 Start-Process -Wait：exe 启动失败时 PyInstaller 会弹一个模态错误框并
# 一直等人点确定，-Wait 就永远不返回（实测卡死过一次构建）。所以这里带超时，并且把
# "有没有写出报告"当成唯一判据。
$report = Join-Path $ProjectRoot 'logs\pet-build-selftest.txt'
$smokePng = Join-Path $ProjectRoot 'logs\pet-build.png'
$importReport = Join-Path $ProjectRoot 'logs\pet-build-imports.txt'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $report) | Out-Null

# 第一步：运行时依赖能不能导入。这一步抓的是"缺 DLL"这类只在冻结环境里才暴露的问题
# （ffi-8.dll 就漏过一次，用户双击后只看到"桌宠启动失败"）。
if (Test-Path -LiteralPath $importReport) { Remove-Item -LiteralPath $importReport -Force }
Write-Step '冒烟测试 1/2：导入检查（运行时依赖是否齐全）'
$proc = Start-Process -FilePath $exe -ArgumentList '--import-check', '--report', $importReport -WindowStyle Hidden -PassThru
if (-not $proc.WaitForExit(120000)) {
    Write-Err2 '导入检查超时 —— exe 可能弹了模态错误框'
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Get-Process -Name $Name -ErrorAction SilentlyContinue | Stop-Process -Force
    exit 1
}
if (-not (Test-Path -LiteralPath $importReport)) { Write-Err2 '导入检查没有写出报告'; exit 1 }
$importText = Get-Content -LiteralPath $importReport -Raw -Encoding UTF8
if ($importText -notmatch 'IMPORT CHECK PASS') {
    Write-Err2 'exe 缺运行时依赖（下面这些模块导不进来）：'
    $importText -split "`n" | Where-Object { $_ -match '\[FAIL\]' } | ForEach-Object { Write-Host $_ -ForegroundColor Yellow }
    Write-Host '  缺 DLL 通常在 <env>\Library\bin，把对应文件名加进本脚本的 $runtimeDllPatterns。' -ForegroundColor Yellow
    exit 1
}
Write-Ok '导入检查通过（含 ctypes/ssl/sqlite3/expat 与 Qt）'

# 第二步：真的渲染一帧并走一遍后端链路。后端没跑时个别检查会失败，那是正常的 —— 这里
# 只验证"exe 能起来、Qt 能加载、能把结果写到文件"。
#
# 为什么不能只写 Start-Process -Wait：exe 启动失败时 PyInstaller 会弹一个模态错误框并
# 一直等人点确定，-Wait 就永远不返回（实测卡死过一次构建）。所以这里带超时。
if (Test-Path -LiteralPath $report) { Remove-Item -LiteralPath $report -Force }
Write-Step '冒烟测试 2/2：启动 exe（渲染一帧 + 走一遍后端链路）'
# 不加 --offscreen：打包好的 exe 不一定带 Qt 的 offscreen 插件，而自检本身不会 show()
# 任何窗口，用原生平台插件即可（顺便验证 windows 平台插件真的被打进去了）。
$smokeTimeoutSec = 180
$proc = Start-Process -FilePath $exe -ArgumentList '--screenshot', $smokePng, '--report', $report -WindowStyle Hidden -PassThru
if (-not $proc.WaitForExit($smokeTimeoutSec * 1000)) {
    Write-Err2 "exe 在 $smokeTimeoutSec 秒内没退出 —— 多半是启动失败后弹了模态错误框"
    Write-Host '  读一下框里的文字：' -ForegroundColor Yellow
    Write-Host "    & `"$py`" tools\inspect_windows.py --process $Name" -ForegroundColor Yellow
    Write-Host "    & `"$py`" tools\read_dialog.py     # 打印对话框正文（含缺失的 DLL / 模块名）" -ForegroundColor Yellow
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Get-Process -Name $Name -ErrorAction SilentlyContinue | Stop-Process -Force
    if (-not (Test-Path -LiteralPath $report)) { exit 1 }
    Write-Warn2 '不过报告已经写出来了，继续检查报告内容'
}
if (-not (Test-Path -LiteralPath $report)) {
    Write-Err2 "exe 启动失败，没有写出报告 —— 看 logs\pet-crash.log，或直接跑一次 exe 看弹窗"
    exit 1
}
$reportText = Get-Content -LiteralPath $report -Raw -Encoding UTF8
$pass = ($reportText -match 'PET SELFTEST PASS')
$offscreenFail = ($reportText -match '\[FAIL\] (角色渲染|内置形象)')
if ($offscreenFail) {
    Write-Err2 'exe 起来了，但离屏渲染失败（Qt 平台插件或绘制依赖缺失）'
    Write-Host $reportText
    exit 1
}
if ($pass) { Write-Ok 'exe 冒烟测试通过（渲染 + 后端链路都在）' }
else { Write-Warn2 'exe 能启动并渲染，但后端链路没测通（服务没跑时属正常）；报告：logs\pet-build-selftest.txt' }

Write-Host ''
Write-Host '============================================================' -ForegroundColor Green
Write-Ok "已生成 $exe（$mb MB）"
Write-Host ' 双击即可打开桌宠；它会显示在屏幕右下角，并出现在系统托盘里。'
Write-Host ' 后端没跑时：双击 dist\启动聊天机器人.exe（服务没在运行时，桌宠右键菜单里也会多出一项）。'
Write-Host '============================================================' -ForegroundColor Green
