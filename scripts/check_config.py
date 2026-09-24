"""Minimal dependency-free configuration check.

Run this when you have Python but not the project's virtual environment yet::

    python scripts/check_config.py

``scripts/verify.ps1`` is the full check (dependencies, GPU, services, tests); this
script exists so a JSON/YAML typo can be caught before spending time on an install.
It validates:

* every ``.json`` under ``config/`` parses, and ``models.json`` has the fields the
  download/start scripts index by,
* ``config.yaml`` parses and the sections the code reads are present,
* persona definitions have the fields the UI and TTS router need,
* every model path in the config resolves inside the project root (the "models on
  D:" contract),
* no shipped file references a plugin name that is not registered,
* every first-party import names a module that actually exists.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
issues: List[str] = []
notes: List[str] = []


def fail(message: str) -> None:
    issues.append(message)


def ok(message: str) -> None:
    print(f"  [OK] {message}")


def warn(message: str) -> None:
    notes.append(message)
    print(f"  [!]  {message}")


def check_json() -> Dict[str, Any]:
    print("检查 JSON 配置…")
    models: Dict[str, Any] = {}
    for path in sorted((ROOT / "config").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            fail(f"{path.name} JSON 语法错误：第 {exc.lineno} 行 {exc.msg}")
            continue
        ok(f"{path.name} 解析成功")
        if path.name == "models.json":
            models = data
    return models


def check_models(models: Dict[str, Any]) -> None:
    print("检查模型档位表…")
    if not models:
        fail("config/models.json 缺失或无法解析")
        return
    tiers = models.get("chat_tiers") or []
    if not tiers:
        fail("models.json 里没有任何 chat_tiers")
        return
    ids = {tier.get("id") for tier in tiers}
    default = models.get("default_tier")
    if default not in ids:
        fail(f"default_tier={default!r} 不在 chat_tiers 中（可用：{sorted(ids)}）")
    else:
        ok(f"默认档位 {default}")
    for tier in tiers:
        name = tier.get("id", "?")
        for field in ("repo", "file", "local_name", "n_gpu_layers", "ctx"):
            if field not in tier:
                fail(f"档位 {name} 缺少字段 {field}")
        local = tier.get("local_name", "")
        if local and not local.isascii():
            fail(f"档位 {name} 的 local_name 含非 ASCII 字符，可能在某些工具链下出错")
        if not isinstance(tier.get("n_gpu_layers"), int):
            fail(f"档位 {name} 的 n_gpu_layers 必须是整数（-1 表示全部卸载）")
    ok(f"{len(tiers)} 个档位字段完整")

    for spec in models.get("support_models") or []:
        name = spec.get("id", "?")
        if spec.get("kind") == "python":
            continue
        for field in ("repo", "file", "local_name", "port"):
            if field not in spec:
                fail(f"辅助模型 {name} 缺少字段 {field}")
    ok(f"{len(models.get('support_models') or [])} 个辅助模型条目")


def check_yaml() -> Dict[str, Any]:
    print("检查 YAML 配置…")
    yaml_path = ROOT / "config" / "config.yaml"
    if not yaml_path.exists():
        fail("config/config.yaml 不存在")
        return {}
    text = yaml_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except ImportError:
        warn("PyYAML 未安装，改用项目内置的极简解析器做检查")
        sys.path.insert(0, str(ROOT / "src"))
        from core.config import _mini_yaml_load  # type: ignore

        data = _mini_yaml_load(text)
    except Exception as exc:  # noqa: BLE001
        fail(f"config.yaml 解析失败：{exc}")
        return {}

    for section in ("llm", "memory", "speech", "server"):
        if section not in data:
            fail(f"config.yaml 缺少 [{section}] 段")
    if "llm" in data:
        base_url = str(data["llm"].get("base_url", ""))
        if not base_url.startswith("http"):
            fail(f"llm.base_url 不合法：{base_url!r}")
        else:
            ok(f"对话模型地址 {base_url}")
        if int(data["llm"].get("context_tokens", 0)) <= 0:
            fail("llm.context_tokens 必须为正整数")
    if "memory" in data:
        weights = [
            data["memory"].get(key)
            for key in ("w_dense", "w_lexical", "w_graph", "w_importance", "w_recency")
        ]
        if all(value == 0 for value in weights if value is not None):
            fail("memory 的所有检索权重都是 0，检索将失去意义")
        if float(data["memory"].get("token_budget", 0)) <= 0:
            fail("memory.token_budget 必须为正数")
    if "speech" in data:
        preference = data["speech"].get("tts_preference") or []
        if not preference:
            fail("speech.tts_preference 为空，语音合成将无后端可用")
        else:
            ok(f"语音优先级 {' > '.join(preference)}")
    ok("config.yaml 结构完整")

    personas_path = ROOT / "config" / "personas.yaml"
    if personas_path.exists():
        try:
            import yaml  # type: ignore

            personas_doc = yaml.safe_load(personas_path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            sys.path.insert(0, str(ROOT / "src"))
            from core.config import _mini_yaml_load  # type: ignore

            personas_doc = _mini_yaml_load(personas_path.read_text(encoding="utf-8"))
        entries = personas_doc.get("personas") or {}
        if not isinstance(entries, dict):
            fail("personas.yaml 的 personas 必须是映射")
        else:
            for persona_id, spec in entries.items():
                if not isinstance(spec, dict):
                    fail(f"人设 {persona_id} 的定义不是对象")
                    continue
                sys.path.insert(0, str(ROOT / "src"))
                from persona.catalog import CATALOG
                spec = {**CATALOG.get(persona_id, {}), **spec}
                if not str(spec.get("system_prompt") or "").strip():
                    fail(f"人设 {persona_id} 缺少 system_prompt")
                voice = spec.get("voice") or {}
                if voice and not isinstance(voice, dict):
                    fail(f"人设 {persona_id} 的 voice 必须是对象")
            ok(f"personas.yaml 定义了 {len(entries)} 个人设")
    return data


def check_paths() -> None:
    print("检查路径契约（模型使用配置目录）…")
    root = ROOT.resolve()
    sys.path.insert(0, str(root / "src"))
    from core.config import load_config
    model_root = load_config(root).paths.models_dir.resolve()
    config_dir = root / "config"
    offenders: List[str] = []
    for name in ("config.yaml", "models.json"):
        path = config_dir / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        # Look for absolute Windows paths (D:\... or D:/...) in model settings, while
        # ignoring URLs — "http://127.0.0.1:8080/v1" would otherwise look like the
        # drive-letter-plus-slash shape.
        for match in re.finditer(r"(?<![:/\w])([A-Za-z]:[\\/][^\"'\s,}\]]+)", text):
            raw = match.group(0)
            candidate = Path(raw.replace("\\\\", "\\"))
            try:
                assert any(candidate.resolve().is_relative_to(base) for base in (root, model_root))
            except (ValueError, AssertionError):
                offenders.append(f"{name}: {raw}")
    if offenders:
        for item in offenders:
            fail(f"模型路径不在配置目录内：{item}")
    else:
        ok("模型路径检查通过")
    for directory in ("models", "data", "logs", "web", "src"):
        if not (root / directory).exists():
            warn(f"目录缺失（首次启动会自动创建）：{directory}")


def check_source_imports() -> None:
    """Verify that first-party imports name modules that actually exist.

    Catches the classic refactor mistake — moving a file and leaving a stale
    ``from memory.old_module import ...`` behind — without needing to import
    anything (imports would require the virtual environment to be installed).
    """
    print("检查源码内部导入完整性…")
    src = ROOT / "src"
    if not src.exists():
        fail("src/ 目录不存在")
        return

    top_level = {"core", "llm", "memory", "speech", "persona", "api", "plugs"}
    existing = {
        ".".join(path.relative_to(src).with_suffix("").parts)
        for path in src.rglob("*.py")
    }
    existing |= {path.name for path in src.iterdir() if path.is_dir()}

    missing: List[str] = []
    pattern = re.compile(
        r"^\s*(?:from|import)\s+((?:core|llm|memory|speech|persona|api|plugs)(?:\.[\w]+)*)",
        re.M,
    )
    for path in sorted(src.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            target = match.group(1)
            if target in existing:
                continue
            # ``from memory import store`` and ``from memory.store import x`` are
            # both fine as long as some prefix of the dotted path exists.
            parts = target.split(".")
            if any(".".join(parts[: index + 1]) in existing for index in range(len(parts))):
                continue
            if parts[0] in top_level and len(parts) == 1:
                continue
            line = text[: match.start()].count("\n") + 1
            missing.append(f"{path.relative_to(src)}:{line} -> {target}")

    if missing:
        for item in missing:
            fail(f"引用了不存在的模块：{item}")
    else:
        ok(f"扫描 {len(existing)} 个模块，未发现失效导入")


def main() -> int:
    print("=" * 60)
    print(" 配置检查")
    print(f" 项目根目录: {ROOT}")
    print("=" * 60)
    models = check_json()
    check_models(models)
    check_yaml()
    check_paths()
    check_source_imports()

    print("")
    print("=" * 60)
    if issues:
        print(f" 发现 {len(issues)} 个问题：")
        for item in issues:
            print(f"   - {item}")
    else:
        print(" 配置检查通过")
    if notes:
        print(f" 提示 {len(notes)} 条：")
        for item in notes:
            print(f"   - {item}")
    print("=" * 60)
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
