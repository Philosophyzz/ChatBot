# 二次元桌宠素材

右键桌宠 → **更换形象**，直接选择：

| 角色 | 素材 | 风格 |
|---|---|---|
| 樱樱 · 猫耳团子 | [sakura_cat.png](sakura_cat.png) | 樱花粉、猫耳、挥爪 |
| 薄荷 · 软软兔 | [mint_bunny.png](mint_bunny.png) | 薄荷绿、垂耳、软绒披肩 |
| 露娜 · 星星魔女 | [luna_witch.png](luna_witch.png) | 丁香紫、星月帽、小斗篷 |

形象按人设记住，重启仍有效。支持呼吸起伏、左右轻摆、悬停歪头、点击跳跃和飘心，
以及聆听、思考、说话时的状态光晕；单击按头部 / 脸颊 / 身体触摸，长按拥抱，双击打开聊天，滚轮缩放，拖动移动桌宠。
这些是透明 PNG 配合程序动画，并非逐帧眨眼或口型动画。

“换回默认形象”恢复当前二次元角色专属立绘。选择另一款内置形象会同时切换对应角色。
仍可通过“选一张图片…”导入自己的图片。

素材随 `scripts/build-pet.ps1` 打包进 exe；已有 exe 需要重新构建才能包含本次更新。
内置素材不会被“换回默认形象”删除。用户选项只保存 `builtin:<id>`，无需写入机器绝对路径。

## 生成记录

使用内置 **imagegen** 工具生成，2026-09-23。三个独立生成请求，无外部角色参考图。
以下为使用的提示词集合，公共约束加各角色描述。

公共约束：

```text
Use case: stylized-concept. Asset type: transparent desktop pet sprite for a Python Qt app,
square canvas. Create ONE original adorable anime chibi mascot, full body.
Polished Japanese 2D game sprite illustration, clean dark outlines, simple cel shading,
readable silhouette at 160px, lively and sweet. Centered front three-quarter view,
entire ears/hat hands feet visible, character fills 85–88% of canvas height with clear
transparent margin around it. Genuinely TRANSPARENT alpha background, no background color,
no checkerboard, no ground shadow, no text, no watermark, no border, no props outside
character. A single finished character only.
```

各角色描述：

```text
sakura_cat: adorable anime chibi cat-eared girl mascot, huge expressive warm pink eyes,
blush, pastel pink bob hair with soft cat ears, cream and rose cozy outfit with a small
bow, tiny shoes, one paw-like hand raised in a friendly wave; clean dark plum outlines.

mint_bunny: adorable anime chibi bunny-eared girl mascot, huge expressive teal eyes,
blush, mint-green short fluffy hair, two soft bunny ears with one ear gently bent,
cozy cream and mint capelet and modest dress, tiny boots, hands held together in front,
cheerful curious slight head tilt; clean dark teal outlines.

luna_witch: adorable anime chibi little witch mascot, huge expressive periwinkle eyes,
blush, lavender bob hair, small soft navy and lilac witch hat with one gold star charm,
cozy starry navy cape and modest lilac dress, tiny boots, one hand raised greeting,
cheerful sweet smile and playful head tilt; clean dark plum outlines; no broom.
```

PNG 保留工具输出的透明通道；运行时仅在切换形象时缓存为最长边 512 像素，动画直接变换绘制。


## Live2D 示例预览

`hiyori.png` 与 `mao.png` 来自本机直接渲染的 Live2D 官方模型，分别匹配桃濑日和与虹色真央；不是 AI 生成的原创角色。版权与使用条款见 [Live2D资源许可](../../../docs/Live2D资源许可.md)。
