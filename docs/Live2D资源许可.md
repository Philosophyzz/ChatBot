# Live2D 模型与来源

本机安装两款 Live2D 官方原创示例角色：桃濑日和（Hiyori Momose）和虹色真央（Mao Niziiro）。原始模型存放在 `D:/Models/Live2D/Hiyori` 与 `D:/Models/Live2D/Mao`。
本应用中的陪伴性格和对话是独立创作的演绎，不代表官方角色设定。

This content uses sample data owned and copyrighted by Live2D Inc. The sample data are utilized in accordance with terms and conditions set by Live2D Inc. This content itself is created at the author’s sole discretion.

- 模型源：[Live2D/CubismWebSamples](https://github.com/Live2D/CubismWebSamples/tree/develop/Samples/Resources)
- [Free Material License](https://www.live2d.com/eula/live2d-free-material-license-agreement_en.html)
- [各示例角色条款](https://www.live2d.com/eula/live2d-sample-model-terms_en.html)，日和的角色设计不得修改；本项目保留原设计。
- 模型安装器保存精确 Git revision、每个文件的 Git blob 校验值、许可证和 NOTICE 到 `D:/Models/Live2D`。
- `src/pet/assets/hiyori.png`、`mao.png` 是本机直接渲染的同款模型预览，作为对应角色的立绘降级和菜单图标。
- 渲染器：[live2d-py](https://github.com/EasyLive2D/live2d-py)，Cubism Core 适用 Live2D 自有运行库许可；模型的免费使用不等于公共领域。

动作实现参考 [SoulLink_Live2D](https://github.com/nanlingyin/SoulLink_Live2D)：读取真实参数能力、LLM 输出参数关键帧、验证范围、平滑过渡、语音嘴部与身体动作分离。本项目以 Qt/Cubism 原生窗口重新实现，没有移植其浏览器前端。
