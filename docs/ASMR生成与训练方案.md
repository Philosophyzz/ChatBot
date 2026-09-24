# 桌宠 ASMR：纯环境／触发音

本次按你的选择，专做**无说话、无耳语、无音乐**的纸张沙沙、柔软刷拭、木质轻敲。它是声音生成功能，不作助眠或治疗效果承诺。

首轮已完成 100 步真实 LoRA 训练、重新加载检查点和三类声音的前后对照生成；详细指标、样本与当前音质局限见 [首轮实验结果](ASMR首轮实验结果.md)。

## 已实现的产品入口

启动根目录 `启动聊天机器人.exe`，右键 → **ASMR（无语音）**。

也可以输入：`播放纸张 ASMR 15分钟`、`暂停ASMR`、`继续ASMR`、`停止ASMR`。这些明确命令直接控制播放器，不依赖 LLM 生成回复。按住说话也可启动播放；播放期间按住录音会先结束音频，因此暂停和继续请使用文字或右键。

- 纸张沙沙、柔软刷拭、木质轻敲、随机组合。
- 45 秒试听；5 / 15 / 30 分钟定时；5%～80% 独立音量；暂停、继续、停止。
- 保留双声道、缓慢左右声像移动、片段交叉淡化、整段淡入淡出、峰值限制。当前声像移动不是个体化 HRTF 双耳声学模拟。
- 播放时暂停主动感知与主动说话，停止免提监听。主动打字聊天或按住录音会结束 ASMR；结束后不会自动开启麦克风。
- 不自动播放、不打开网页、不上传录音。日常播放在 CPU 上完成，不与聊天模型竞争显存。

当前桌宠音库使用有来源记录的 CC0 触发音，自动编排不同片段。神经生成音频先进入研究试听区；只有试听确认后才导入桌宠。这让“能稳定播放的版本”和“正在学习的新声音”都有明确的状态。

## 模型选择与架构

采用 **HKUSTAudio/AudioX + 注意力 LoRA**。官方模型支持 44.1kHz 双声道声音生成；权重和源码为 CC-BY-NC-4.0，当前按个人非商业实验使用，不能据此宣称可以直接商业发行。

模型位置：`D:\Models\asmr\AudioX`；文本编码器：`D:\Models\asmr\t5-base`。下载脚本固定版本并验证权重 SHA256。AudioX 源码固定在提交 `3bdfb7081636b9e62224039e37dadaa264dc781f`。

```mermaid
flowchart LR
    A[类别与声音描述] --> B[T5 文本编码]
    B --> C[AudioX 扩散模型 + ASMR LoRA]
    C --> D[44.1kHz 双声道候选音效]
    D --> E[技术检查与试听]
    E --> F[桌宠本地音效库]
    F --> G[随机编排 / 声像 / 淡化 / 定时]
    G --> H[桌宠播放]
```

AudioX 的纯文本路径仍保留训练时学到的空视频、空音频特征。运行时省去不参与纯文本任务的 CLIP 和第二个音频编码器，冻结主模型、声码器与 T5，只训练注意力线性层的 LoRA。使用 PyTorch CUDA 的现有运行环境，不依赖 vLLM 或尚待重启的 WSL。vLLM 负责语言模型，不能直接替代音频扩散推理。

调研过的其他路线：

| 路线 | 本次取舍 |
|---|---|
| Stable Audio 3 Small-SFX | 后续可对比，官方提供 LoRA 工具；当前 Hugging Face 下载需要账号申请并接受条款，本机未登录 |
| Stable Audio Open 的现成 ASMR LoRA | 查到的 text2asmr v1 作者明确报告低频杂音、呼噜样声音和人声残留，未用作成品 |
| 耳语 TTS / DeepASMR | 面向人声，与你本次选择的无说话目标不一致 |
| 从零训练通用音频大模型 | 此阶段以预训练模型适配为主，把录音与听评投入放在声音质量上 |

## 数据：这次的试验与正式训练

这次已下载 8 个 Freesound CC0 公开高质量 MP3 预览，保留作者、原始页面、许可页面证据、下载地址与 SHA256；共整理 53 个切片、约 207 秒，其中 45 个训练切片、8 个验证切片。训练和验证按**原始录音**分开，没有把同一录音的相邻片段随机分到两边。

这批 MP3 数据只用来验证下载、切片、训练、评估、生成和接入流程。重采样到 44.1kHz 不会恢复 MP3 丢失的细节；样本数量也不足以支撑成品音质结论。

正式第一轮建议准备 **3～5 小时干净、获授权的无损原始录音**，每类约 1 小时，再逐步扩到更多材质。这是项目的数据预算建议，不是模型的官方最低要求。

1. **录制**：48kHz/24bit WAV，安静房间；稳定麦克风增益和位置。先录 30 秒环境底噪，再录纸张、软刷、木质轻敲。分别覆盖慢/中速度、稀疏/连续、近/中距离和不同材质；不要把键盘、音乐、说话混进目标录音。
2. **授权记录**：每段保留录制者、日期、素材来源、许可或本人授权说明。可下载、可试听不等于可训练；不抓取来源不明的主播作品。
3. **切片**：优先 4～8 秒，保留事件的起音、衰减和必要间隔；不对摩擦声套用语音 VAD，不大幅降噪，不逐声道独立归一化。
4. **标签**：类别、材质、动作、速度、距离、背景、原始录音 ID、麦克风与录音会话 ID。例如 `soft dry paper crinkling, close microphone, slow irregular movement, quiet background, no speech, no music`。描述听得到的声音，避免给所有样本添加没有依据的“高品质”“双耳”标签。
5. **划分**：按录音会话和物体分组，约 80% / 10% / 10% 训练、验证、最终测试；在测试集中保留未见过的材质或速度组合。保持每类均衡，不能靠重复同一段录音扩大数据量。
6. **质量检查**：排除削波、明显电流声、音乐、人声和突发大响声；自动筛选之后仍要人工抽听。保存原始文件，清洗产物单独存放。

当前数据清单格式为 JSONL，每行包含 `audio / text / category / source_id / split / duration / license / sha256`。真实数据可沿用此格式；`tools/asmr_dataset.py` 提供 CC0 小样本的可重复处理实现。

## 训练方法与预算

第一轮的目标是验证“模型能学到这批触发音的差异”，而非一次训练就得到任意 ASMR 音效。

| 项目 | 小样本流程验证 | 正式首轮建议 |
|---|---|---|
| 音频 | 44.1kHz 双声道，约 4 秒 | 44.1kHz 双声道，4～8 秒 |
| 可训练模块 | 注意力 Q / KV / QKV / 输出投影 LoRA | 同左，先保持范围不变 |
| rank | 4 | 8，验证后再比较 16 |
| batch | 1 | 1，梯度累积 4～8 |
| 学习率 | 1e-4 | 5e-5 起步，比较 1e-4 |
| 步数 | 100 | 1000～3000 的实验预算，依据验证集停止 |
| 精度 | 冻结扩散权重 BF16；LoRA 参数 FP32 | 同左 |
| 优化器 | AdamW、梯度裁剪 1.0 | 同左 |
| 数据缓存 | 先编码音频潜变量和文本条件，缓存到 CPU | 大数据改为磁盘缓存 |

训练目标为 AudioX 原有余弦噪声日程的 v-prediction MSE；不训练聊天 LLM，不改变角色人设。声码器和文本编码器在特征提取后退出 GPU；验证使用固定噪声和时间步，便于比较训练前后。每隔指定步数保存验证指标与适配器，只把验证损失更低的候选另存为 `best.safetensors`。验证改善不等于听感改善。

本机 RTX 5080 16GB 应采用串行占用显存：实验包装器核验并暂停本项目聊天推理进程，完成或发生 Python 异常后恢复原模型；不会按进程名停止其他 GPU 应用。实际耗时、峰值显存以 `report.json` 为准，不预先承诺训练时长。强行关机或杀死包装器时不能保证执行恢复逻辑，之后重新启动根目录启动器即可恢复聊天服务。

## 可运行命令

已存在的环境为 `venvs\asmr`，复用主环境的 Torch 2.8.0+cu128，并把兼容性依赖装在独立环境中。复现安装和数据下载：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install-asmr.ps1
```

小样本训练（会临时暂停聊天模型，并在结束后恢复）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\asmr-lab.ps1 -Action train --steps 100 --rank 4 --eval-every 20
```

生成基础模型对照：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\asmr-lab.ps1 -Action generate --steps 50 --seconds 8 --seed 2026 --prompt "Soft dry paper rustling, close microphone, slow movements, no speech, no music." --output data\asmr\candidates\paper-base.wav
```

使用适配器生成同提示、同种子的对照（路径须指向实际产生的检查点）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\asmr-lab.ps1 -Action generate --steps 50 --seconds 8 --seed 2026 --adapter D:\Models\asmr\adapters\AudioX-ASMR-pilot\best.safetensors --prompt "Soft dry paper rustling, close microphone, slow movements, no speech, no music." --output data\asmr\candidates\paper-lora.wav
```

如果验证没有改善，`best.safetensors` 不会生成；仍可研究 `step-100.safetensors`，但不能把它描述为更好的模型。

试听确认无说话、无音乐且没有明显刺耳伪影后，导入某个候选：

```powershell
D:\conda-envs\chatbot\python.exe tools\asmr_import.py data\asmr\candidates\paper-lora.wav --category paper --reviewed-no-speech
```

权重、适配器和训练数据在 `D:\Models\asmr`。产品音库、试听候选和播放样例在项目 `data\asmr`；日志在 `logs\asmr-*`。神经音效不进入对话历史和长期记忆。

## 验收方法

训练集损失下降只能说明优化器工作。每一版还必须做下面的对照：

- **技术检查**：双声道是否保存、是否有限值、是否削波、起止是否有爆音、拼接是否有突变、长时播放是否正常停止。数字峰值不能代替耳机实际声压测量。
- **内容检查**：每类至少 10 个提示 × 3 个种子，比较基础模型与适配器；记录出现人声、音乐、机械噪声的比例。对“无说话”产品，人声残留必须淘汰。
- **听评**：隐藏版本名称，以相同播放响度比较质感自然度、触发音辨识度、背景干净程度、重复感和舒适度。建议至少 5 名听者，先收集偏好，再决定是否扩大训练。
- **泛化检查**：保留未训练的纸张/木材和新速度描述，检查模型是否只复现训练片段；测试阶段的结果不用于反复调参。
- **回归检查**：切换角色、开启聊天、停止/暂停、声卡切换、后端断开时仍能结束音频；播放 ASMR 时不得触发主动说话。

生产数据足够后，才增加 CLAP 文音一致性、足够样本量的 FAD 等辅助指标。这些分数不能单独证明“让人放松”。

## 后续功能顺序

1. 用户标注喜欢／不喜欢的音效，先调整播放频率，不自动拿聊天或屏幕数据训练。
2. 图形化生成与对照试听，显示基础模型/适配器、种子和生成状态；通过试听再收录。
3. 增加布料、水滴、雨声等类别，每类先补足干净数据，防止少量杂音污染所有类别。
4. 如要更真实的绕耳定位，引入有明确许可的 HRTF 或真实双耳录音，单独评估空间感。
5. 比较 Stable Audio 3 Small-SFX 与 AudioX 的延迟、音质、显存及许可后，再决定是否更换基础模型。

## 来源

- [AudioX 官方代码与模型说明](https://github.com/ZeyueT/AudioX)
- [AudioX 官方权重与许可](https://huggingface.co/HKUSTAudio/AudioX)
- [Stable Audio 3 官方 LoRA 文档](https://github.com/Stability-AI/stable-audio-3/blob/main/docs/workflows/lora.md)
- [text2asmr v1 作者对当前杂音问题的说明](https://huggingface.co/aoxo/text2asmr-stable-audio)
- CC0 素材的逐条作者与原始页面见 `D:\Models\asmr\datasets\cc0-pilot\sources.json`。
