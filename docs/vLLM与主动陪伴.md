# vLLM、Live2D 与主动陪伴

## 本机交付状态（2026-09-24）

- 已安装 WSL 2.7.14 的 Microsoft 签名安装包，启用 VirtualMachinePlatform 和 WSL Windows 组件。Windows 返回需要重启；`wsl --status` 仍报告虚拟化平台尚未生效。没有自动重启电脑。
- 已准备并校验 Ubuntu 24.04 rootfs。vLLM 0.30.0 的独立安装脚本、进程管理器、桌宠入口和 API 后端选择已接入。
- **尚未在本机启动 vLLM，也没有宣称完成 vLLM 性能验证。** 重启后才能安装 Linux Python/CUDA 依赖并做真实生成验证。迁移成功前，原 llama.cpp 服务继续供当前聊天及多模态功能使用。
- vLLM 使用 `D:/Models/hf/MiMo-V2.6-Distill-Qwen-9B-INT4-W4A16-AutoRound`。这是 letechlead 发布的社区 INT4 safetensors 量化版，约 8.98 GB；视觉塔保留 BF16，原 GGUF 文件保留在 `D:/Models/gguf`。
- 原始 MiMo BF16 权重约 18.8 GB，不能直接全量装入 16GB 显卡。此处为 vLLM 单独准备量化快照，不把 GGUF 扩展名改成 safetensors，也不把 llama.cpp 服务改名冒充 vLLM。

## 重启后完成迁移

右键桌宠 → 底座模型 → **重启电脑后：继续安装 vLLM**。也可运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\Harness\ChatBot\scripts\install-vllm.ps1
```

脚本使用独立发行版 `ChatBot-vLLM`（文件在 `venvs/wsl-vllm`），环境为 `/opt/chatbot-vllm`。它会安装固定版本 vLLM、检测 CUDA、检查模型下载完成标记，停止本项目旧推理服务，启动 vLLM 并发出真实 completion 请求。只有验证成功才把 `config/local.yaml` 中的连接持久化为 `backend: vllm`，之后唯一 EXE 启动器自动启动 vLLM。

安装日志：`logs/vllm-install.log`；推理日志：`logs/vllm.log`。状态：`data/vllm-install.json`。首次 Linux 依赖安装仍需联网下载 PyTorch/CUDA 等组件。安装器不会重启电脑，不会卸载其他 WSL 发行版。

vLLM 配置在 `config/vllm.json`，与旧 GGUF 清单分开。默认 8192 上下文、单并发、78% 显存预算、eager 模式、每条请求最多一张图；实际能否达到目标速度和显存余量，需重启后的硬件验证。vLLM 更擅长 GPU 调度与批处理，单用户延迟仍需实测。

## Live2D 已实际启用

默认角色为桃濑日和，另有虹色真央；右键「更换形象」统一切换角色、Live2D 模型、图标、初始记忆和音色。原有樱樱、薄荷、露娜继续提供 PNG 形象。两款官方模型的位置、免费素材条款、署名见 [Live2D资源许可](Live2D资源许可.md)。

- 原生 Qt/OpenGL/Cubism 渲染，已在 RTX 5080 上检查两模型加载与切换。
- 自主眨眼、呼吸、视线及间隔 8～16 秒的轻微头身动作，无需不停调用 LLM。
- 聊天后一次调用生成最多三段关键帧；参数只允许模型真实存在的头部、身体、眼眉和手臂参数，数值有限且钳位到模型范围。失败时继续本地动作。
- 每帧使用 cubic 缓动，结束回中立姿态；音频音量包络单独驱动嘴巴，避免动作计划抢占嘴部。此处是音量口型，不是音素识别。
- 音乐模式按电脑输出音量与瞬时能量变化轻摆；不是完整的节拍/歌曲结构识别。

参考 SoulLink_Live2D 的参数能力描述、范围校验、平滑过渡、嘴部与身体动作分离。其逐帧两阶段规划被改成一次返回少量关键帧，以控制单显卡后台调用量。本项目保留 Qt 桌宠，不要求浏览器窗口常开。

## 感知与场景

右键可选「常态 / 一起看电影 / 一起听音乐 / 勿扰」，以及「主动感知」「允许观察当前窗口画面」。桌宠底部持续显示开关与当前场景，「感知状态 / 隐私说明」显示最近抑制原因及系统声音设备状态。

| 场景 | 观察与反馈 | 最短观察间隔 | 主动发言冷却 |
|---|---|---|---|
| 常态 | 前台窗口缩略图、空闲时间和活动汇总；操作电脑期间保持安静 | 90 秒 | 10 分钟 |
| 一起看电影 | 当前窗口画面及可能可见的字幕、媒体标题/播放状态；播放时仅气泡或表情，避免语音盖过电影 | 120 秒 | 5 分钟 |
| 一起听音乐 | Windows 媒体标题/歌手、电脑输出音量和能量变化；轻摆，偶尔短评 | 120 秒 | 5 分钟 |
| 勿扰 | 暂停屏幕、输入活动和系统音量采集，不主动说话 | — | — |

所有场景每小时最多主动发言 4 次；最近 60 秒与桌宠交互过、正在聊天/录音/朗读、用户可能离开、密码/银行等受保护窗口，先由本机规则抑制，不调用 LLM。每个候选观察再由模型决定说或不说，模型故障、超时或 JSON 无效时安静降级。关闭或切换场景后会丢弃在途旧结果。

屏幕感知仅用于本机回环地址的本地模型。外部 API 聊天仍可用，但感知自动暂停。截图限制为当前前台窗口、最长边 960、JPEG，只保存在内存中，不落盘、不写入日志、不添加对话历史或长期记忆。键鼠只汇总空闲、活动次数和移动距离，不读取按键文字；I/O 是磁盘吞吐统计，不读取文件内容。

电影/音乐模式采集 WASAPI 扬声器输出的音量包络，不开麦克风、不保存录音、当前不做电影对白转写或歌词辨识。未接入系统媒体会话的播放器可能无法提供片名/歌名，此时只依据画面和音量，不编造身份信息。受保护窗口判断主要基于标题/进程名，处理私密内容请用总开关暂停。

情绪属于不确定印象：不能仅凭鼠标移动或敲键速度断言用户愤怒、焦虑等，也不把影视人物表情当作用户情绪。用户明确表达的情绪仍交给原聊天情绪模块。

## 验证与后续

完成了参数越界/NaN/未知参数/嘴部保护、发言冷却、每小时上限、忙碌与勿扰门控、外部 API 不传截图、WSL 未就绪不启动等自动检查；完成两模型真实 GPU 渲染、切换后纹理有效、实际 LLM 动作与看图决策检查。观察前后对话消息数保持一致。

后续优先做：重启后的 vLLM 文本/视觉延迟实测；多屏窗口选择和单应用白名单；按需启用的电影音轨本地转写；用原创 Live2D 骨骼替换示例角色；根据用户主动反馈学习打扰频率。vLLM 验证前不做更激进的 KV/编译性能调参。

来源：[vLLM 官方 GPU 安装文档](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)、[GGUF 支持说明](https://docs.vllm.ai/en/latest/features/quantization/gguf/)、[AutoRound 量化与部署](https://github.com/intel/auto-round)、[MiMo INT4 权重卡](https://huggingface.co/letechlead/MiMo-V2.6-Distill-Qwen-9B-INT4-W4A16-AutoRound)、[SoulLink_Live2D](https://github.com/nanlingyin/SoulLink_Live2D)。
