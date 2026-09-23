# 本地 LLM 聊天机器人

一个完全跑在你自己机器上的中文语音聊天助手：本地大模型对话、跨会话长期记忆、语音输入、
甜美语音朗读、可切换人设。硬件按 **RTX 5080 16GB + 31GB 内存** 调优，所有模型权重都放在
项目目录（D 盘）。

```
你说话 🎤 → Whisper 识别 → 混合检索长期记忆 → 本地大模型生成
                                                    ↓
                        网页流式显示 ← 本地 TTS（可克隆音色）朗读 🔊
                                                    ↓
                          后台异步抽取记忆 → 时序知识图谱 + 向量库
```

---

## 一、三分钟上手

```powershell
cd D:\Harness\ChatBot

# 1) 安装运行环境（conda 环境 chatbot + 依赖 + llama.cpp 推理引擎），约 3~8 分钟
powershell -ExecutionPolicy Bypass -File scripts\install.ps1

# 2) 自检：不需要模型权重，先确认代码链路是通的
powershell -ExecutionPolicy Bypass -File scripts\verify.ps1

# 3) 下载主模型（默认 27B Q4，约 16.5GB；国内加 -Mirror 快很多）
powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1 -Mirror

# 4) 一键启动（模型服务 + 网页界面 + 自动开浏览器）
powershell -ExecutionPolicy Bypass -File scripts\start-all.ps1
```

浏览器会自动打开 <http://127.0.0.1:8077>。首次提问需要等 27B 模型加载完（1~3 分钟）。

想先看界面长什么样、不等下载？用模拟模式：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-all.ps1 -Mock
```

关闭：`scripts\stop.ps1`


### 下载模型：优先 ModelScope，HF 镜像作备选

实测（本机、同一台机器、同一时间）：

| 源 | 同一个 1.6GB `model.bin` | 结论 |
|---|---|---|
| huggingface.co 直连 | 连不上 | 不可用 |
| `hf-mirror.com` | 头 20MB 有 1.5MB/s，整文件跑到 40~850MB 就断，且**不会自动恢复** | 只适合小文件 |
| **ModelScope** | **4.4MB/s，支持 Range 断点续传，一次跑完** | 大文件走这里 |

所以：

```powershell
# 语音识别模型（1.6GB）——一条命令，自动断点续传
python D:\Harness\ChatBot\tools\download_whisper.py

# 任何其它大文件（IndexTTS2 的 gpt.pth 3.5GB、GPT-SoVITS 权重等）
python D:\Harness\ChatBot\tools\download_file.py --url <直链> --out <目标文件> --chunk-mb 24
```

`tools\download_file.py` 三个关键设计（都是踩坑换来的）：

1. **先写 `.part`，校验长度后才改名**——半成品不会被误当成完整模型。之前它直接写
   `model.bin`，另一个下载工具（`hf download`）想删除这个"过期文件"重建时就撞上
   Windows 文件锁，报 `PermissionError: [WinError 32]`。
2. **按 24MB 分块 + 逐块重试**——一次连接断掉只损失一块，不会前功尽弃。
3. **`trust_env=False`**——绕开系统代理，否则连 127.0.0.1 也会被代理拦。

小文件（几十 MB 的 config/tokenizer、ffmpeg.exe、G2PWModel.zip）用 `hf download` 就够了：

```powershell
conda activate chatbot
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"     # 必须：镜像不代理 Xet，不关会 401
hf download <repo> --local-dir <目录>
```

### 运行环境：conda 环境 `chatbot`（推荐）

解释器与全部依赖都放在 **D 盘**：

| 位置 | 内容 |
|---|---|
| `D:\conda-envs\chatbot` | 运行环境本体（Python 3.11） |
| `D:\conda-pkgs` | conda 包缓存（`.condarc` 里的 `pkgs_dirs`） |
| `D:\Harness\ChatBot\models`、`data`、`logs` | 模型权重、数据库、日志 |

```powershell
conda activate chatbot          # 之后所有命令都在这个环境里跑
python -m pytest tests -v       # 跑测试
python tests\smoke_stdlib.py    # 零依赖自检
python tests\ui_check.py        # 真实浏览器检查界面（需要 Chrome 且在跑服务）
```

没有 conda 也能用：`scripts\install.ps1` 会在 `venvs\main` 建 venv。
`scripts\common.ps1` 的 `Get-Python` 解析顺序是
**`CHATBOT_PYTHON` 环境变量 → conda 环境 `chatbot` → `venvs\main` → PATH 上的 python**，
所以脚本永远用同一个解释器，不需要手动指定。想强制指定就设：
`$env:CHATBOT_PYTHON = 'D:\some\python.exe'`。

依赖清单只有一份：`requirements.txt`（核心）+ `requirements-speech.txt`（语音，可用
`install.ps1 -NoSpeech` 跳过）。手工装环境时照着这两个文件装即可：

```powershell
D:\conda-envs\chatbot\python.exe -m pip install -r requirements.txt -r requirements-speech.txt `
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 二、模型选型：为什么是这些

### 2.1 硬约束（实测，不是估计）

| 项目 | 实测值 | 影响 |
|---|---|---|
| 显卡 | RTX 5080 **16GB** GDDR7（驱动 595.97，CUDA 13.2） | 权重 + KV cache 的总预算 |
| 桌面占用 | 已被浏览器/通讯软件占 **3.4GB** | 实际可用 **≈12.5GB** |
| 内存 | 31.1GB（DDR5） | 决定能"溢出"多少层到 CPU |
| CPU | Ryzen 7 9800X3D | 溢出层的计算速度足够可接受 |
| D 盘可用 | 447GB | 27B 级模型 + 语音模型约 22GB，空间充足 |

**关键结论：27B 的 Q4 量化权重（约 16.5GB）装不进 12.5GB 可用显存。** 任何声称"16GB 卡跑满血
27B"的说法都需要靠把层放到内存里——这不是缺陷，而是这个硬件上的物理事实。

### 2.2 三个档位，按你的取舍选

配置在 `config/models.json`，切换只需改一个参数，无需改代码。
**表中每个仓库与文件名都已用 `python tests/verify_models.py` 对 HuggingFace API 实测校验过。**

| 档位 | 模型 | 权重（实测） | GPU 层数 | 上下文 | **实测速度** | 适合 |
|---|---|---|---|---|---|---|
| `fast-9b`（**当前默认**） | Qwen3.5-9B Q4_K_M（官方仓库） | **5.29GB** | 全部在 GPU | 32768 | **~125 tok/s，0.25s/轮** | **日常与语音对话** |
| `quality-27b` | Qwen3.6-27B Q4_K_M（带 MTP 加速头） | **15.66GB** | 44 层在 GPU | 8192 | ~10 tok/s | 质量优先、能等 |

**默认档位是 9B，这是实测选出来的**，不是拍脑袋：

- 9B 权重 5.29GB 能**全量放进显存**，每 token 无需跨 PCIe 搬运 → **125 tok/s**
- 27B 权重 15.66GB 装不进 12.5GB 可用显存，实测速度随"留在 CPU 的权重量"线性下降：
  28 层 5.74 tok/s → 40 层 8.25 → 46 层 10.14 → 全量 OOM。**这是数学后果，换任何推理框架都一样**
- 聊天陪伴场景要的是响应快，所以默认 9B

想切到 27B：改 `config/models.json` 的 `default_tier` 与 `config/config.yaml` 的 `llm.model` 为
`quality-27b`，然后 `scripts\stop.ps1` + `scripts\start-all.ps1`。

### 2.3 两个必须知道的调优项（都已写进脚本）

**① 思考预算 = 0**（`--reasoning-budget 0`）。Qwen3.x 默认开启思考，实测同一问题：

| 设置 | 延迟 | 思考量 |
|---|---|---|
| 默认（不限制） | **14–20 秒** | 4278–6053 字 |
| `--reasoning-budget 256` | 2.4 秒 | ~780 字 |
| **`--reasoning-budget 0`** | **0.3–0.5 秒** | 0 |

关掉思考后回答反而更自然口语（实测："辛苦啦，快找个舒服的地方躺下～"），因为预算不再被
"我该如何扮演这个角色、该不该用表情符号"这类自我盘问吃光。要做复杂推理时把它改成 256。

**② 本地请求必须绕过系统代理**。如果你的机器开着 Clash/V2Ray（系统代理 `127.0.0.1:7890`），
httpx 会把发往 `127.0.0.1:8080` 的请求也交给代理，代理连不上本机端口就回 **502**，
于是界面报"模型服务不可用"而模型其实完全健康。**判据：连一个没人监听的端口如果返回 502
而不是 connection refused，就是代理在拦。** 代码里所有本地客户端都设了 `trust_env=False`
（注意 `proxy=None` **无效**，实测仍是 502），并有守护测试防止回退。

### 2.4 下载完成后必须校验完整性

GGUF 被截断时 `llama-server` 会在加载到一半时报一个很难懂的错。下载脚本只在**完整成功**时
才写 `data/models.manifest.json`，所以下完立刻校验一次：

```powershell
python tests\verify_downloads.py --mirror --repair      # conda activate chatbot 之后
```

它会比对远端 `Content-Length`、检查 GGUF 头部结构与"末尾全零"截断特征。
发现截断时**重新跑同一条下载命令即可续传**，不会从头开始。

### 2.3 辅助模型（记忆检索质量的关键）

| 用途 | 模型 | 体积（实测） | 缺失时的后果 |
|---|---|---|---|
| 记忆向量化 | Qwen3-Embedding-0.6B（1024 维） | 0.6GB | 自动降级为本地哈希嵌入：记忆仍可用，但"换个说法就找不到" |
| 记忆重排 | Qwen3-Reranker-0.6B | 1.12GB | 自动降级为词法重合度重排，前几条准确率下降 |
| 语音识别 | faster-whisper large-v3-turbo (int8) | 1.6GB | 只能用文字输入 |
| 语音合成 | IndexTTS2（可克隆音色） | ~4GB | 自动切在线音色（edge-tts） |

前两个在 `download-models.ps1 -Support` 里一起下载；后两个由各自的运行库首次使用时自动拉取。

**为什么用 llama.cpp 而不是 vLLM**：16GB 显存跑 27B 必须做"部分层卸载"，llama.cpp 的
`--n-gpu-layers` + KV cache 量化（`q8_0`）是这件事上最成熟、可控性最高的方案；vLLM 在 Windows
上仍依赖第三方构建，且官方不支持把权重分层放 CPU/GPU。这里的取舍是**可预测性优先**。

---

## 三、功能详解

### 3.1 长期记忆（本项目的核心）

不是"把聊天记录塞进上下文"，而是一套分层记忆系统：

| 层 | 内容 | 作用 |
|---|---|---|
| 工作记忆 | 最近若干轮对话 | 当前话题连贯性（按 token 预算中间裁剪） |
| 情景记忆 | 原始消息 + FTS5 全文索引 | 「我三个月前说过什么」精确可查 |
| 语义记忆 | 抽取的事实 / 偏好 / 档案 | 「它知道我不喝加糖的咖啡」 |
| 时序知识图谱 | 实体 + 带时效的关系边 | 「谁和谁什么关系、从什么时候开始」 |
| 反思记忆 | 从事实簇提炼的高层洞察 | 「它理解我而不是只记录我」 |

**写入路径**（异步，绝不拖慢回复）：

1. 抽取：用受 JSON Schema 约束的模型调用，抽出事实 / 实体 / 关系 / 档案。
   模型不可用时回退到保守的规则抽取（宁可少记，不可记错）。
2. 消解：把「小明」「我」「用户」归一到同一个图谱节点。
3. 去重：内容哈希去重 + 向量近邻去重。
4. **冲突消解**：同一主体+谓词下的新事实会**关闭旧事实的有效期窗口**并建立取代链。
   这是"从北京搬到上海"不会变成"同时住两个城市"的原因。
5. 落库：写入记忆、向量、关系边，并投影出档案字段。

**检索路径**（三路混合 + 重排 + 时间衰减）：

| 通道 | 解决的失败场景 |
|---|---|
| 稠密向量 | 「我喝咖啡不加糖」vs「美式，不要糖」——换个说法也能召回 |
| BM25 词法 | 人名、数字、专有名词——向量会糊掉，词法不会 |
| 时序图谱 | 问句里出现某个人名时，把他所有关系边都作为候选 |
| 档案直注 | 「你是谁」这类寒暄也要知道你叫什么 |

6. **可解释**：每条命中的记忆都带完整评分拆解（`dense/lexical/graph/importance/recency/rerank`
   各贡献多少）。检索调参不靠猜——界面打开 🧪 就能看到。

**睡眠期固化**（空闲时后台跑）：
补齐缺失向量 → 合并近似重复 → 把久远的情景记忆压缩成摘要 → 从事实簇提炼洞察 →
让从未被召回的记忆缓慢衰减、被反复确认的增强。

**可纠正**：左侧「记忆」面板能看到每一条、能改、能删。记忆系统要长期可信，用户必须能纠错。

**你不需要填任何东西**：记忆全部是聊出来的，面板里没有任何"新增记忆"的输入框，
也不用先做资料设置。聊到关于你自己的事（名字、职业、喜好、在忙什么），几秒后条目自己出现。
这条约定还写进了每一轮的系统提示词（`_MEMORY_CONDUCT`），否则小模型看到"你有长期记忆"
就会反过来查户口——追问姓名、职业、住址，因为记忆是空的，它把空槽位当成了要填的问题。

**想清空重来**（比如之前用测试语料聊过，残留的假事实会被当成真的记住）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stop.ps1   # 先停服务
python tests\reset_memory.py                                # 预演：只统计，不改数据
python tests\reset_memory.py --apply --sessions             # 备份 + 清空记忆与聊天记录
powershell -ExecutionPolicy Bypass -File scripts\start-all.ps1
```

`--apply` 一定会先把 `data\memory.sqlite3` 复制成带时间戳的备份，删错了可以拷回来。

### 3.2 语音

* **输入**：按住空格或 🎤 说话。前端用 Web Audio 直接编码成 16kHz 单声道 WAV，
  所以**没装 ffmpeg 也能用**；服务端在收到 WebM/OGG 等格式时才需要 ffmpeg 转码。
* **输出**：本地优先、在线兜底的路由器。本地 IndexTTS2 支持**零样本音色克隆 + 情感控制**
  （"甜美"来自参考音频的音色，情绪来自 `emotion` 字段）；不可用时自动切到在线音色。
  **注意：IndexTTS2 默认没有安装**（`models/tts/IndexTTS2` 为空），此时路由会落到在线
  edge-tts——那是联网调用微软的服务，不是本地合成。`/api/health?deep=true` 里的
  `tts.backends.indextts.error` 会直接告诉你要做什么。
* **装本地音色（都装进 `chatbot` 这一个环境）**：

  ```powershell
  conda activate chatbot
  python -m pip install torch==2.8.* torchaudio==2.8.* --index-url https://download.pytorch.org/whl/cu128
  python -m pip install -e vendor\index-tts -i https://pypi.tuna.tsinghua.edu.cn/simple
  python tools\check_tts_stack.py          # 装完必须验证（见下）
  ```

  权重 `IndexTeam/IndexTTS-2`（5.9GB，`gpt.pth` 3.5G / `s2mel.pth` 1.2G / 情感模型 1.2G）
  已经在 `models\tts\IndexTTS2`，不需要再下。

  > **一个必须知道的版本冲突**：`indextts` 钉死 `numpy==2.2.6`，而 GPT-SoVITS 的
  > `requirements.txt` 写 `numpy<2.0` —— 两者在 pip 解析层面无法同时满足。
  > 顺序是「先装 GPT-SoVITS 的依赖，再把 numpy 抬回 2.2.6」，然后跑
  > `python tools\check_tts_stack.py` 验证两边都能导入。
  > 如果有模块是按 numpy 1.x ABI 编译的，导入会失败 —— 那时训练只能改用独立环境
  > （`scripts\install-gptsovits.ps1 -EnvName chatbot-tts-train`），脚本会告诉你。
* **参考音色**：IndexTTS2 是克隆模型，没有内置音色，必须有 5~15 秒参考音频。
  `tools\make_reference.py --personas` 已按每个人设各自的音色生成了
  `models\voices\<人设>.wav`；**把 `cloned_sweet.wav` 换成你自己的录音**就能克隆你自己的声音。
* **流式合成**：按块切分（`speech.tts_chunk_chars`，默认 40 字 ≈ 2~4 秒语音），
  **合成一块就发一块**，浏览器立刻开始播，同时后台并发合成后面的块
  （`speech.tts_chunk_concurrency`，默认 2）。前端走 `/api/voice/tts/stream`
  （4 字节长度前缀 + 1 字节 tag 的分帧），失败自动回退到整段的 `/api/voice/tts`。
* **结果缓存**：同一句话 + 同一音色第二次直接命中内存缓存（实测 0.015 秒，对比在线后端
  每次 1.8~13 秒）。开场白、口头禅这类重复文本几乎零延迟。
* **显存共存**：语音与 TTS 模型都是懒加载 + 空闲自动卸载（默认 15/10 分钟），
  用完把显存还给大模型。这是 27B + 语音能在 16GB 上共存的原因。

实测（本机，47 字回答）：
| 路径 | 出声时间 |
|---|---|
| 整段接口（旧行为） | 2.6~4.4s，长回答实测到 **13.4~17.6s** |
| 流式接口（现在） | 首块 2.0~6.4s，且**首块到达即出声** |
| 重复同一句（缓存） | **0.015s** |
| Windows SAPI（本地） | **0.32s**（与文本长度无关，但音色机械） |

在线 edge-tts 的延迟与文本长度**无关**（实测 3 字 5.1s、30 字 1.8s、19 字 12.3s），
纯粹是到微软服务器的网络抖动——所以想稳定快，只能换本地音色。

详见 **[docs/语音配置.md](docs/语音配置.md)**（含音色克隆步骤与推荐音色表）。

### 3.3 人设

人设不只是提示词，而是**提示词 + 采样参数 + 音色 + 情感 + 记忆作用域**的绑定。切换人设时
声音和性格一起变。`config/personas.yaml` 里有 7 个预设人设，**默认是「温柔姐姐」**
（`config.yaml` 的 `default_persona`，改这里就换默认）；自己写的排在列表最前面，
内置的 4 个（小甜、清和、阿芜、先生）只做兜底，不会悄悄变成默认。
在界面里可以新建 / 编辑，也可以直接改 `config/personas.yaml`。

### 3.4 桌宠（可执行文件，随时开关）

`dist\ChatBotPet.exe`（约 82MB，PyInstaller 单文件）：无边框、置顶、可拖动的桌宠，
会听你说话、用当前人设的音色回答；系统托盘里可随时显示 / 隐藏 / 退出。

| 操作 | 效果 |
|---|---|
| 按住麦克风按钮 | 说话（松开即发送；走本地 Whisper 识别） |
| 右键 →「一直听着」 | **免提**：点一次就不用再按按钮，停下来自动断句并发出去 |
| 右键 →「免提灵敏度」 | 灵敏 / 普通 / 抗噪 —— 外放或键盘声大时用「抗噪」 |
| 右键 →「更换形象」 | 内置樱樱（猫耳）、薄荷（兔耳）、露娜（星星魔女），也可导入自己的透明 PNG |
| 右键 →「底座模型」 | 9B / 14B / 27B / 35B 直接切（见 [docs/模型切换.md](docs/模型切换.md)） |
| 左键拖动 | 移动位置 |
| 双击 | 打开完整网页界面 |
| 单击身体 | 打开打字输入框 |
| 右键 | 菜单：打字聊天 / 一直听着 / 免提灵敏度 / 静音朗读 / 更换形象 / 底座模型 / 切换人设 / 重新连接 / 隐藏 / 退出 |
| Ctrl+Space | 按住说话（桌宠有焦点时） |

新增三款二次元 Q 版形象，支持呼吸起伏、轻摆、悬停歪头和点击跳跃飘心；单击仍可打字聊天。
选择按人设保存，托盘图标同步切换。素材和生成提示词见 [桌宠素材说明](src/pet/assets/README.md)。
从源码启动即可使用；已有 `ChatBotPet.exe` 需运行 `scripts/build-pet.ps1` 重新打包。

> **服务没在运行时**，菜单里会**额外**出现一项「本地服务没在运行 → 启动它」；服务健康时
> 它不显示。最初它是个常驻项，还能对着一个健康的服务再启动一遍 —— 那会把网页那边的连接
> 掐断（浏览器里就成了 "Failed to fetch"），所以现在：只在确实连不上时出现，且点之前先探
> 一次 /api/health，健康就直接告诉你不用重启。平时启动服务的正式入口仍是双击
> `启动聊天机器人.exe`。

* 它是**瘦客户端**：模型、记忆、语音都在本地服务里，所以换人设、换模型两边自动同步。
* 同一个人设只能开一只桌宠（再点一次会把它切到前台）；不同人设可以并排站着，自动让位。
* 后端没跑时，右键菜单会多出一项「本地服务没在运行 → 启动它」（等约 1 分钟模型加载）。
* **显存守卫**：whisper、IndexTTS2、对话模型三者同时驻留会超出 16GB，而超了的后果不是报错
  而是**进程被原生代码干掉**（网页表现为 "Failed to fetch"）。现在加载其中一个之前，
  如果空闲显存不够，会先把另一个卸掉；`GET /api/health` 里的 `vram_guard` 能看到当前状态。
  关掉它要自己保证显存：`config/config.yaml` → `speech.vram_guard: false`。
* 从源码运行：`powershell -ExecutionPolicy Bypass -File scripts\start-pet.ps1`
* 重新打包：`powershell -ExecutionPolicy Bypass -File scripts\build-pet.ps1`
  （`-OneDir` 启动更快；`-Console` 出带控制台的调试版）
* 自检（无需人工看）：`dist\ChatBotPet.exe --selftest --report dist\_pet.txt`
  —— 离屏渲染 5 种状态 + 走一遍 人设 / 对话 / 语音 链路，结果写进报告文件。
* 麦克风/免提不对劲时：`python tools\check_hands_free.py --calibrate`
  —— 先安静 2 秒、再正常说一句话，它会测出你的说话音量并**推荐该用哪个灵敏度档**；
  `--seconds 8 --trace` 则打印逐 100ms 的电平曲线与触发值。
  桌宠自己也会察觉：免提开着 20 秒完全没听到接近阈值的音量时，它会提示换档位。
* 崩溃会写到 exe 同目录的 `logs\pet-crash.log`：windowed 程序没有控制台，不写文件就什么都看不到。

### 3.5 训练自己的音色（GPT-SoVITS 微调）

IndexTTS2 是零样本克隆（不用训练）；想让声音**更像本人、更稳**，用 GPT-SoVITS 微调：

```powershell
# 1) 安装（默认装进同一个 conda 环境 chatbot；约 12GB）
powershell -ExecutionPolicy Bypass -File scripts\install-gptsovits.ps1 -Plan
powershell -ExecutionPolicy Bypass -File scripts\install-gptsovits.ps1

# 2) 整理素材（切片 + 静音裁剪 + 归一化 + Whisper 转写 + 生成 dataset.list）
python tools\tts_dataset.py --input D:\voice\raw --out D:\voice\dataset --speaker my_voice

# 3) 训练（先 -DryRun 看计划；默认 v2ProPlus，对普通素材更宽容）
powershell -ExecutionPolicy Bypass -File scripts\train-tts.ps1 -DataDir D:\voice\dataset -Speaker my_voice -DryRun
powershell -ExecutionPolicy Bypass -File scripts\train-tts.ps1 -DataDir D:\voice\dataset -Speaker my_voice

# 4) 起推理服务，人设里 voice.backend 写 gpt_sovits
powershell -ExecutionPolicy Bypass -File scripts\train-tts.ps1 -Serve
```

详见 **[docs/训练音色.md](docs/训练音色.md)**：素材要求、参数含义、显存与时间口径、排错表。

---

## 换底座模型（9B / 14B / 27B / 35B，不用改配置文件）

网页左侧「系统」→ 底座模型，或桌宠右键 →「底座模型」，点一下即切换（30~120 秒）。
切换只重启 8080 上的对话模型，嵌入/重排继续跑，记忆不会降级。

| 档位 | 权重 | 显存 | 速度 | 现状 |
|---|---|---|---|---|
| `fast-9b` | 5.29GB | 全进显存 | 60~90 tok/s | ✅ 已下载 |
| `balanced-14b` | 7.88GB | 全进显存 | 40~70 tok/s | ⬜ 待下载 |
| `quality-27b` | 15.66GB | 44 层上 GPU | 9~10 tok/s | ✅ 已下载 |
| `quality-35b` | 10.02GB | `--n-cpu-moe 8` → 约 9.85GB | 5~15 tok/s | ⬜ 待下载 |

```powershell
# 下载（自己执行，支持断点续传）
powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1 -Tier balanced-14b -Mirror
powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1 -Tier quality-35b -Mirror

# 想知道某个 GGUF 要多少显存、专家要放几层到内存（不用下完整个文件）
python tools\plan_moe_offload.py --repo unsloth/Qwen3.6-35B-A3B-GGUF --file Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf --ctx 8192 --budget 10
```

详见 **[docs/模型切换.md](docs/模型切换.md)**（含 35B 塞进 10GB 的完整换算表与排错）。

---

> 想快速看懂整体结构（进程、模块、扩展点、数据流、接口）：见 **[docs/框架总览.md](docs/框架总览.md)**。
> 想把它传到 GitHub（**不含模型**，也绝不会带上 data/ 里的对话与记忆）：见 **[docs/上传GitHub.md](docs/上传GitHub.md)**。

## 四、目录结构

```
D:\Harness\ChatBot\
├─ config\
│  ├─ config.yaml        主配置（采样、检索权重、语音、服务）
│  ├─ personas.yaml      人设定义
│  ├─ models.json        模型档位表（切换模型只改这里）
│  └─ local.yaml         运行时覆盖（设置界面写入，优先级最高）
├─ src\
│  ├─ core\              配置 / 契约 / 注册表 / 会话 / 编排引擎
│  ├─ llm\               LLM 后端、嵌入、重排
│  ├─ memory\            记忆：存储 / 向量 / 抽取 / 检索 / 固化
│  ├─ speech\            音频处理 / 识别 / 合成
│  ├─ persona\           人设与提示词装配
│  ├─ api\               FastAPI 路由与服务
│  └─ plugs\             插件目录（放进去就自动加载）
├─ web\                  网页界面（原生，无构建步骤）
├─ scripts\              安装 / 下载 / 启动 / 停止 / 自检
├─ tests\                单元与端到端测试
├─ models\               所有模型权重（D 盘）
├─ data\                 数据库、人设覆盖、PID
├─ logs\                 运行日志（含 JSONL 结构化日志）
├─ requirements*.txt     依赖清单（install.ps1 与手工建环境共用同一份）
├─ vendor\               index-tts / GPT-SoVITS 源码（TTS 与训练用）
```

---

## 五、常用操作

```powershell
# 交付验收（编译 + 零依赖自检 + 单元测试 + 配置检查 + 接口冒烟，一次跑完）
powershell -ExecutionPolicy Bypass -File scripts\selftest.ps1

# 环境体检
scripts\verify.ps1                  # 完整自检（依赖、GPU、服务、测试）
scripts\verify.ps1 -Quick           # 只看服务是否在跑
scripts\verify.ps1 -Bench           # 附加一次真实推理测速（显示 tok/s）

# 模型管理
scripts\download-models.ps1 -List                  # 列出档位
scripts\download-models.ps1 -Mirror -Support       # 主模型 + 嵌入 + 重排
python tests\verify_models.py                      # 校验仓库与文件名是否仍然可用
scripts\start-models.ps1 -Tier fast-9b             # 只起模型服务
scripts\start-models.ps1 -NGpuLayers 32            # 手动指定 GPU 层数

# 局域网访问（手机当麦克风和音箱；无鉴权，仅限可信网络）
scripts\start-all.ps1 -Lan

# 测试
python -m pytest tests -v                               # conda activate chatbot 之后
python tests\smoke_stdlib.py                           # 零依赖自检（不需要 GPU/模型）
python tests\ui_check.py                               # 真实浏览器检查界面（需要 Chrome）

# 停止（先让网页服务优雅退出：关闭数据库、合并 WAL；失败才强杀）
scripts\stop.ps1

# 清空长期记忆（自测语料会把假事实记成真的，需要时这样重来）
python tests\reset_memory.py --apply --sessions
```

服务端口：网页 8077 ｜ 对话模型 8080 ｜ 嵌入 8081 ｜ 重排 8082

自检与冒烟都在临时数据目录里跑：`tests\smoke_http.py` 会把 `paths.data_dir` 指到
`%TEMP%\chatbot-smoke-*`，所以跑测试不会往你的真实记忆库里写“我叫小明”。

---

## 六、故障排查

| 现象 | 原因与处理 |
|---|---|
| 网页打不开 | 看 `logs\api.err.log`；确认 `verify.ps1 -Quick` 里网页服务是否正常 |
| 提问报"无法连接模型服务" | 模型服务没起来或还在加载：看 `logs\llama-chat.log`，首次加载要 1~3 分钟 |
| 生成极慢（< 2 tok/s） | GPU 层数不合适：见 `docs/显存调优.md`，或换 `-Tier fast-9b` |
| 报显存不足 / 系统卡顿 | 减小 `--n-gpu-layers`（如 `-NGpuLayers 22`），关掉占显存的软件 |
| 说话没反应 | 浏览器麦克风权限；`http://127.0.0.1` 属于安全上下文，权限弹窗应能出现 |
| 提示"无法解码音频" | 装 ffmpeg：`winget install Gyan.FFmpeg` |
| 没有语音朗读 | `verify.ps1` 看 TTS 组件；本地模型没装时会自动用在线音色（需联网） |
| 记忆好像没生效 | 面板打开 🧪 看召回详情；记忆抽取是异步的，聊完等几秒；模型太小时抽取质量会下降 |
| 觉得"必须自己填记忆" | 不用填：记忆全部自动抽取，面板只用来查看/纠正/删除 |
| 弹窗关不掉 / 一打开页面就盖着"编辑记忆" | 已修（CSS 里 `.modal{display:flex}` 曾盖掉 `[hidden]`）；现在 Esc、点遮罩、取消都能关 |
| 它记错了我的事（比如名字） | 那是测试语料留下的假事实：`python tests\reset_memory.py --apply --sessions` 清空重来 |
| 模型下载中断 | 直接重跑下载脚本，会自动断点续传 |
| 停止后 `data\memory.sqlite3` 被占用 / `-wal` 很大 | 用 `scripts\stop.ps1`（会先请求优雅退出并合并 WAL），不要直接杀进程 |

日志：
* `logs\chatbot.jsonl` — 结构化日志，含每轮 token 数、检索耗时、抽取结果
* `logs\llama-chat.log` — 模型服务的原始输出（看显存分配就是这里）

---

## 七、扩展它

架构是插件式的：任何能力都对应一个 `Protocol`（在 `src/core/types.py`），
实现后注册名字即可在配置里选用。加一个新 TTS 引擎 = 一个文件 + 一个装饰器。

```python
# src/plugs/my_tts.py —— 放进去就自动加载，无需改任何核心代码
from core.registry import KIND_TTS, register
from core.types import SpeechChunk

@register(KIND_TTS, "my_engine")
class MyTTS:
    name = "my_engine"
    supports_cloning = False
    offline = True

    async def synthesize(self, text, voice, *, stream=False):
        yield SpeechChunk(data=my_synth(text), mime="audio/wav", is_final=True)

    async def health(self):
        return {"ok": True, "name": self.name}
```

然后在 `config/config.yaml` 里：`speech: { tts_preference: [my_engine, edge] }`

可插拔的扩展点：LLM 后端、嵌入器、重排器、向量库、记忆存储、STT、TTS、输入输出通道、
记忆抽取器。详见 **[docs/架构.md](docs/架构.md)**。

---

## 八、验证状态（诚实清单）

这份代码在交付前已经**真跑过**，不是"写完就交"：

```
scripts\selftest.ps1
  1) 编译检查          exit=0    44 个 Python 文件全部编译通过
  2) 零依赖自检        exit=0    24 项全过（不需要 GPU、不需要模型权重）
  3) 单元测试          exit=0    60 passed
  4) 配置检查          exit=0    含模型路径契约与模块引用完整性
  5) 模型可用性        exit=0    5 个条目全部解析到真实文件与体积
  6) 接口冒烟          exit=0    真实 HTTP：health/personas/chat/记忆/TTS 全部 200
```

此外还实测过：`install.ps1` 在干净 venv 上端到端安装成功（Python 3.11.16 +
fastapi/httpx/numpy/faster-whisper/edge-tts 全部可导入）、`run_server.py` 从项目根
目录启动成功、`/` 返回含标题的 HTML、`/assets/*` 正常、`/api/chat` 返回内容、
`/api/voice/tts` 返回**真实可播放音频**（edge 4/4 命中）、`chat+speak` 内联音频 62KB。

单元测试覆盖的是**关键承诺**，不是行数：跨会话记忆召回、抽取幂等、冲突消解、
检索 token 预算、嵌入服务故障降级、后端故障转可读错误、语音路由回退、
语音后端排序（音质优先于离线）、API 错误码映射（404/422）。

### 已知限制

1. **27B 档位速度只有 4~8 tok/s**——这是 16GB 显存的物理限制，不是软件问题。
   想要流畅请用 `balanced-14b`（7.88GB 全部进显存，40~70 tok/s）。
2. **无鉴权**：`-Lan` 模式下同一局域网内任何人都能访问。仅限可信网络。
3. **记忆抽取依赖主模型质量**。27B/14B 抽取效果明显好于 9B；模型不可用时降级为规则抽取
   （宁可少记，不可记错）。
4. **14B 档位用的是社区权重**（官方无 14B 稠密模型），介意请选 9B 或 27B。
5. **在线音色需要联网**：默认语音后端是 edge-tts（音质好但需网络），离线时自动回退到
   Windows SAPI（音色机械）。要完全离线的甜美音色请装 IndexTTS2，见 `docs/语音配置.md`。
6. **不做微调**。人设靠提示词 + 语音配置，长期记忆靠记忆系统。
7. **单用户设计**。数据库有 `scope` 字段为多用户预留，但没有账号体系。
8. **未在真实 GPU 上端到端跑过**：开发环境的沙箱不允许启动子进程，因此
   `llama-server` 的实际推理（首次提问、tok/s、显存占用）需要你在本机完成第一步验证。
   显存与速度的预期值基于权重体积与架构推算，`docs/显存调优.md` 给出了自测流程。
