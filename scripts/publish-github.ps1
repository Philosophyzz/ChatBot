# =============================================================================
# 把项目安全地提交到本地 git 仓库，并打印推送到 GitHub 的后续步骤。
#
#   powershell -ExecutionPolicy Bypass -File scripts\publish-github.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\publish-github.ps1 -Message "修复桌宠气泡位置"
#   powershell -ExecutionPolicy Bypass -File scripts\publish-github.ps1 -Remote https://github.com/you/ChatBot.git -Push
#
# 为什么要有这个脚本，而不是让你手敲 git：
#   * models/ 有 33.6GB、vendor/ 5.8GB，data/ 里是**你真实的对话与记忆库** ——
#     推错了撤不干净，所以提交前强制跑一遍 tools\check_repo_hygiene.py；
#   * 没配 git 身份时提交会直接失败（而且邮箱会公开出现在提交记录里），
#     脚本会停下来告诉你该怎么设，不替你猜邮箱。
#
# 推送到远端需要你自己的 GitHub 账号（PAT 或 SSH），脚本不碰凭据。
# 详细说明见 docs\上传GitHub.md。
# =============================================================================
[CmdletBinding()]
param(
    [string]$Root,
    [string]$Message = '',
    [string]$Remote = '',
    [switch]$Push,
    [switch]$SkipCheck
)

. (Join-Path $PSScriptRoot 'common.ps1')
$ProjectRoot = Get-ConfigRoot -Root $Root
$py = Get-Python -Root $ProjectRoot

Write-Host ''
Write-Host '============================================================' -ForegroundColor White
Write-Host ' 准备提交到 git（模型与私人数据不会进去）' -ForegroundColor White
Write-Host '============================================================' -ForegroundColor White

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Err2 '没有找到 git。安装：winget install --id Git.Git -e'
    exit 1
}

Push-Location $ProjectRoot
try {
    # --- 1. 仓库与身份 --------------------------------------------------------
    if (-not (Test-Path (Join-Path $ProjectRoot '.git'))) {
        Write-Step '初始化仓库（默认分支 main）'
        & git init -b main | Out-Null
        if ($LASTEXITCODE -ne 0) {
            # 老版本 git 不支持 -b
            & git init | Out-Null
            & git branch -M main 2>$null
        }
    } else {
        Write-Ok '已经是 git 仓库'
        # 习惯用 main：GitHub 的默认分支名，免得推上去还要改
        $current = (& git branch --show-current) 2>$null
        if ($current -and $current -ne 'main') {
            Write-Step "把分支 $current 改名为 main"
            & git branch -M main
        }
    }

    $gitName = (& git config user.name) 2>$null
    $gitEmail = (& git config user.email) 2>$null
    if (-not $gitName -or -not $gitEmail) {
        Write-Err2 '还没有配置 git 身份，提交会被拒绝。请先执行（把名字/邮箱换成你自己的）：'
        Write-Host ''
        Write-Host '    git config --global user.name  "你的名字"' -ForegroundColor Yellow
        Write-Host '    git config --global user.email "你的邮箱"' -ForegroundColor Yellow
        Write-Host ''
        Write-Host '  不想公开真实邮箱就用 GitHub 的 noreply 地址（设置 → Emails 里能看到）：'
        Write-Host '    git config --global user.email "1234567+你的用户名@users.noreply.github.com"' -ForegroundColor DarkGray
        Write-Host ''
        Write-Host '  配好后重新运行本脚本即可。'
        exit 1
    }
    Write-Ok "提交身份：$gitName <$gitEmail>"

    # --- 2. 暂存并检查边界 ----------------------------------------------------
    Write-Step '暂存所有改动'
    & git add -A
    if ($LASTEXITCODE -ne 0) { Write-Err2 'git add 失败'; exit 1 }

    if (-not $SkipCheck) {
        Write-Step '检查将要上传的内容（模型 / 日志 / data 里的对话与记忆）'
        & $py (Join-Path $ProjectRoot 'tools\check_repo_hygiene.py')
        if ($LASTEXITCODE -ne 0) {
            Write-Err2 '检查没通过 —— 已经暂存的内容里有不该上传的东西，先按上面的提示处理。'
            Write-Host '  查看详细清单：git diff --cached --name-only' -ForegroundColor DarkGray
            Write-Host '  取消暂存：    git reset' -ForegroundColor DarkGray
            exit 1
        }
    }

    # --- 3. 提交 --------------------------------------------------------------
    $staged = (& git diff --cached --name-only) 2>$null
    if (-not $staged) {
        Write-Warn2 '没有需要提交的改动'
        exit 0
    }
    if (-not $Message) {
        $Message = "更新：$(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    }
    Write-Step "提交：$Message"
    & git commit -q -m $Message
    if ($LASTEXITCODE -ne 0) { Write-Err2 '提交失败（看上面的 git 输出）'; exit 1 }
    Write-Ok "已提交 $(($staged | Measure-Object).Count) 个文件"

    # --- 4. 远端与推送 --------------------------------------------------------
    Write-Host ''
    if ($Remote) {
        $existing = (& git remote) 2>$null
        if ($existing -contains 'origin') {
            & git remote set-url origin $Remote
            Write-Ok "origin 已更新为 $Remote"
        } else {
            & git remote add origin $Remote
            Write-Ok "已添加 origin：$Remote"
        }
        if ($Push) {
            Write-Step '推送到 GitHub（需要你的 PAT 或 SSH 凭据）'
            Write-Host '  国内网络若卡住，先设代理再重试：' -ForegroundColor DarkGray
            Write-Host '    git config --global http.proxy http://127.0.0.1:7890' -ForegroundColor DarkGray
            & git push -u origin main
            if ($LASTEXITCODE -ne 0) {
                Write-Err2 '推送失败 —— 看上面的 git 报错（认证 / 网络 / 远端已有提交）'
                exit 1
            }
            Write-Ok '推送完成'
        }
    } else {
        Write-Host '------------------------------------------------------------' -ForegroundColor White
        Write-Host ' 接下来（只需要做一次）：' -ForegroundColor White
        Write-Host '------------------------------------------------------------' -ForegroundColor White
        Write-Host ' 1) 在 GitHub 网页上新建仓库（New repository）'
        Write-Host '    仓库名随意；**不要**勾选 Add README / .gitignore / license' -ForegroundColor Yellow
        Write-Host '    （我们已经有一份了，勾了会产生一次无关的首次提交）'
        Write-Host ''
        Write-Host ' 2) 关联远端并推送（把 URL 换成你自己的）：'
        Write-Host '      git remote add origin https://github.com/你的用户名/仓库名.git' -ForegroundColor Yellow
        Write-Host '      git push -u origin main' -ForegroundColor Yellow
        Write-Host ''
        Write-Host ' 3) 国内网络推不动时先走代理（你的 Clash 在 7890）：'
        Write-Host '      git config --global http.proxy http://127.0.0.1:7890' -ForegroundColor DarkGray
        Write-Host '      # 用 VPN 的话记得取消：git config --global --unset http.proxy' -ForegroundColor DarkGray
        Write-Host ''
        Write-Host ' 4) 认证：HTTPS 用 Personal Access Token 当密码（GitHub 不再接受登录密码），'
        Write-Host '    或改用 SSH：git remote set-url origin git@github.com:用户名/仓库名.git'
        Write-Host ''
        Write-Host ' 完整说明：docs\上传GitHub.md' -ForegroundColor Green
    }
} finally {
    Pop-Location
}
