"""给桌宠换形象（命令行入口）。

桌宠托盘菜单里有「更换形象 → 选一张图片…」，这个工具做同一件事，只是从命令行走 ——
批量换、或者在没有鼠标的时候用：

    python tools\\set_pet_skin.py --list
    python tools\\set_pet_skin.py --image D:\\pics\\girl.png --persona gentle_sister
    python tools\\set_pet_skin.py --image girl.png                # 所有人设通用
    python tools\\set_pet_skin.py --clear --persona gentle_sister
    python tools\\set_pet_skin.py --demo                         # 生成一张示例图并装上，先看看效果

图片会被规整成 512×512 的透明底 PNG，存到 data\\pet_skins\\，设置写在
data\\pet_settings.json。桌宠最多 30 秒后自动换上新形象（不用重启）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def main(argv: Optional[List[str]] = None) -> int:
    from pet.settings import settings_path
    from pet.skin import DEFAULT_SIZE, SkinError, clear_skin, install_skin, list_skins, skins_dir

    parser = argparse.ArgumentParser(description="替换桌宠形象")
    parser.add_argument("--image", default=None, help="要用的图片（png/jpg/webp/bmp/gif）")
    parser.add_argument("--persona", default=None, help="只给这个人设换（默认：所有人设通用）")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="规整成多少像素的正方形")
    parser.add_argument("--keep-original", action="store_true", help="不规整，直接原样存下来")
    parser.add_argument("--clear", action="store_true", help="清除自定义形象，回到默认角色")
    parser.add_argument("--list", action="store_true", help="列出当前已经设置的形象")
    parser.add_argument("--demo", action="store_true", help="生成一张示例图并装上（用来试功能）")
    args = parser.parse_args(argv)

    if args.list or (not args.image and not args.clear and not args.demo):
        current = list_skins()
        print(f"设置文件：{settings_path()}")
        print(f"形象目录：{skins_dir()}")
        if not current:
            print("当前没有任何自定义形象 —— 桌宠用的是内置的绘制角色。")
        else:
            print(f"当前有 {len(current)} 个自定义形象：")
            for persona, path in current:
                print(f"  {persona:<20} {path}")
        if not (args.list or args.clear or args.demo):
            print("\n用法示例：python tools\\set_pet_skin.py --image D:\\pics\\girl.png")
        return 0

    if args.clear:
        removed = clear_skin(args.persona)
        target = args.persona or "所有人设"
        print(f"已清除 {target} 的自定义形象。" + (f"（删掉了 {len(removed)} 个文件）" if removed else ""))
        return 0

    if args.demo:
        from pet.skin import placeholder_image

        demo_path = skins_dir() / "_demo_source.png"
        image = placeholder_image(args.size)
        if not image.save(str(demo_path), "PNG"):
            print(f"示例图写不出来：{demo_path}")
            return 1
        try:
            stored = install_skin(demo_path, args.persona, size=args.size)
        except SkinError as exc:
            print(f"装不上：{exc}")
            return 1
        print(f"已装上示例形象：{stored}")
        print("桌宠最多 30 秒后会自动换上它（或者右键 →「更换形象 → 换回默认形象」撤回）。")
        return 0

    try:
        stored = install_skin(
            args.image,
            args.persona,
            size=args.size,
            keep_original=args.keep_original,
        )
    except SkinError as exc:
        print(f"失败：{exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"失败：{type(exc).__name__}: {exc}")
        return 1

    who = args.persona or "所有人设"
    print(f"已给 {who} 换上形象：{stored}")
    if not args.keep_original:
        print(f"（已规整为 {args.size}×{args.size} 透明底 PNG；原始文件没有被改动）")
    print("桌宠最多 30 秒后自动生效 —— 想立刻看到就右键 →「更换形象」再点一下，或重启桌宠。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
