# 上传到 GitHub（不含模型）

目标：把**代码、脚本、文档**推上去，而**模型权重（33.6GB）和你的对话数据**留在本机。

一句话版本：

```powershell
cd D:\Harness\ChatBot
powershell -ExecutionPolicy Bypass -File scripts\publish-github.ps1
```

它会：初始化仓库（分支 `main`）→ 暂存 → **强制检查有没有把模型/日志/data 一起暂存** →
提交 → 打印推送命令。下面是要点与完整手动步骤。

---

## 一、上传什么、不传什么

| 路径 | 体积 | 上传吗 | 为什么 |
|---|---|---|---|
| `src/` `tests/` `tools/` `scripts/` `launcher/` `web/` `docs/` `config/*.yaml,json` `*.py` | ~1.4MB | ✅ | 代码与文档本体 |
| `models/` | **33.6GB** | ❌ | GGUF/whisper/IndexTTS2 + 音色样本；都能用 `scripts\download-models.ps1` 重新下 |
| `vendor/` | **5.8GB** | ❌ | GPT-SoVITS 源码/预训练、index-tts 克隆；用 `install-tts.ps1` / `install-gptsovits.ps1` 重新拉 |
| `bin/` | 0.7GB | ❌ | llama.cpp 的 Windows CUDA 构建；`scripts\install.ps1` 会下 |
| `venvs/` `build/` `dist/` | 0.6GB | ❌ | 环境与构建产物（含 82MB 的桌宠 exe） |
| `data/` | 2MB | ❌ **绝对不要** | **你真实的对话与记忆库**（`memory.sqlite3` + 备份）、桌宠设置与形象图 |
| `logs/` | 15MB | ❌ | 运行日志与对话记录（`chatbot.jsonl`） |
| `config/local.yaml` | — | ❌ | 本机覆盖配置（仓库里保留 `local.yaml.example` 模板） |

规则写在 `.gitignore`（每条都带原因），并由两个东西守着，防止以后手滑：

```powershell
python tools\check_repo_hygiene.py     # 问 git：这些路径到底会不会被提交
python -m pytest tests\test_repo_hygiene.py -q   # 4 个用例：私人路径必须被忽略、该上传的不能被忽略
```

`data/` 里的 `memory.sqlite3` 是**你的**聊天记录与长期记忆 —— 推上公开仓库等于公开日记，
而且删不干净（GitHub 的缓存、别人的 fork、已经 clone 的副本都会留）。所以这一步是硬性的。

---

## 二、完整步骤（手动版）

### 0. 先自查

```powershell
cd D:\Harness\ChatBot
git init -b main
git add -A
python tools\check_repo_hygiene.py        # 必须输出 [OK] 干净
```

它同时会告诉你"将上传 N 个文件，合计多少 MB"。正常是 **126 个文件、约 1.4MB**。

### 1. 配置 git 身份（只做一次）

提交必须带身份，而且**邮箱会公开出现在提交记录里**：

```powershell
git config --global user.name  "你的名字"
# 不想公开真实邮箱：设置 → Emails 里能看到 GitHub 给你的 noreply 地址
git config --global user.email "1234567+你的用户名@users.noreply.github.com"
```

### 2. 本地提交

```powershell
git commit -m "初始提交：本地优先的中文语音聊天机器人"
```

（或者直接用 `scripts\publish-github.ps1`，它会替你做上面三步并检查边界。）

### 3. 在 GitHub 上新建仓库

网页 → New repository：

* 填仓库名（例如 `ChatBot`），Public / Private 随你；
* **不要**勾选 `Add a README file`、`.gitignore`、`license` —— 我们本地已经有内容了，
  勾了会在远端先产生一次提交，push 时还要先 pull 合并，平白多一步。

### 4. 关联并推送

```powershell
git remote add origin https://github.com/你的用户名/仓库名.git
git push -u origin main
```

**国内网络**：如果卡在 `Resolving deltas` 或超时，让你机器上的 Clash 帮 git 走代理：

```powershell
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
# 换回 VPN/直连时记得取消：
git config --global --unset http.proxy
git config --global --unset https.proxy
```

**认证**：GitHub 不再接受账号密码。HTTPS 方式要建一个 Personal Access Token
（Settings → Developer settings → Tokens (classic) → 勾 `repo`），push 时用户名填你的
GitHub 用户名、**密码填 token**。想省事就先 `winget install GitHub.cli` 然后 `gh auth login`，
之后 `gh repo create` 能一步建仓库并推上去。

### 5. 确认上传结果

打开仓库页面看一眼：应当只有 100 多个文件、没有 `models/`、没有 `data/`、没有 `logs/`。
再确认一次本地状态：

```powershell
git ls-files | Measure-Object -Line        # 行数 ≈ 126
git ls-files | Select-String "^(data|logs|models|vendor|bin)/"   # 应当没有任何输出
```

---

## 三、之后每次更新

```powershell
git add -A
git commit -m "说明这次改了什么"
git push
```

`README.md` 和 `docs/` 会在仓库首页直接展示，改动它们等于更新说明。

---

## 四、别人克隆下来要怎么跑起来

仓库里没有模型，所以克隆者需要自己准备（这些步骤 README 里也写了）：

```powershell
git clone https://github.com/你的用户名/仓库名.git D:\Harness\ChatBot
cd D:\Harness\ChatBot
powershell -ExecutionPolicy Bypass -File scripts\install.ps1          # 建 conda 环境 + 依赖 + llama.cpp
powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1 -Mirror   # 下对话模型
python tools\download_whisper.py                                      # 下语音识别模型
powershell -ExecutionPolicy Bypass -File scripts\start-all.ps1
```

要注意的两点，最好在仓库说明里讲清楚（README 已经写了"全部在 D 盘"）：

* 文档里的路径固定假设 `D:\Harness\ChatBot` 与 `D:\conda-envs\chatbot` —— 这是本项目的
  设计前提（所有东西都在 D 盘），换盘需要改 `config/config.yaml` 与几个脚本里的默认值；
* 单人用的私人助手项目，**没有做多用户隔离与鉴权**，别把它暴露到公网。

---

## 五、如果已经推了不该推的东西

越早处理越干净：

```powershell
# 1) 从版本库里移除（保留本地文件），并把忽略规则补上
git rm -r --cached data logs
git commit -m "移除私人数据"
git push
```

**但这只是删掉了最新提交里的文件，历史里还在**。敏感内容（token、密码、私人记录）必须
当作已泄漏处理：撤销/轮换那个凭据，然后用 `git filter-repo` 重写历史（或直接删库重建 ——
对这个项目而言往往更快）。所以第一步就把边界钉死，比事后补救划算得多。

---

## 六、可选：加个许可证

仓库没有任何 license 时，默认是"保留所有权利"，别人不能合法复用。想开放就把
`LICENSE` 文件加到根目录（MIT / Apache-2.0 都常见），GitHub 会自动识别并显示在仓库页。
