"""Configuration, registry and utility tests — the failure modes that break everything else."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_paths_derive_under_root(tmp_path: Path) -> None:
    from core.config import Paths

    paths = Paths(root=tmp_path)
    assert paths.gguf_dir == tmp_path / "models" / "gguf"
    assert paths.voices_dir == tmp_path / "models" / "voices"
    assert paths.db_path == tmp_path / "data" / "memory.sqlite3"
    paths.ensure()
    assert paths.data_dir.exists() and paths.gguf_dir.exists()


def test_paths_reject_model_outside_root(tmp_path: Path) -> None:
    """The "models must live on D:" contract is enforced, not just documented."""
    from core.config import Paths

    paths = Paths(root=tmp_path)
    with pytest.raises(ValueError):
        paths.assert_on_project_root(tmp_path.parent / "elsewhere.gguf")
    inside = tmp_path / "models" / "gguf" / "x.gguf"
    assert paths.assert_on_project_root(inside) == inside.resolve()


def test_config_yaml_overrides_and_env(tmp_path: Path, monkeypatch) -> None:
    from core.config import load_config

    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "config.yaml").write_text(
        "llm:\n  temperature: 0.25\n  max_tokens: 321\nserver:\n  port: 9999\n",
        encoding="utf-8",
    )
    config = load_config(tmp_path)
    assert config.llm.temperature == 0.25
    assert config.llm.max_tokens == 321
    assert config.server.port == 9999

    monkeypatch.setenv("CHATBOT_LLM__TEMPERATURE", "0.9")
    monkeypatch.setenv("CHATBOT_SERVER__PORT", "1234")
    config = load_config(tmp_path)
    assert config.llm.temperature == 0.9
    assert config.server.port == 1234


def test_config_local_yaml_wins_over_config_yaml(tmp_path: Path) -> None:
    from core.config import load_config

    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "config.yaml").write_text("memory:\n  top_k: 5\n", encoding="utf-8")
    (tmp_path / "config" / "local.yaml").write_text("memory:\n  top_k: 42\n", encoding="utf-8")
    config = load_config(tmp_path)
    assert config.memory.top_k == 42


def test_config_public_view_hides_api_key(tmp_path: Path) -> None:
    from core.config import AppConfig, Paths

    config = AppConfig(paths=Paths(root=tmp_path))
    config.llm.api_key = "super-secret"
    payload = json.dumps(config.to_public(), ensure_ascii=False)
    assert "super-secret" not in payload


def test_shipped_config_files_parse(project_root: Path) -> None:
    """A typo in a shipped config file must fail the test suite, not the user's launch."""
    from core.config import load_config

    config = load_config(project_root)
    assert config.default_persona
    assert config.llm.base_url.startswith("http")
    assert config.memory.dim > 0
    assert config.speech.tts_preference, "TTS 优先级不能为空"

    models = json.loads((project_root / "config" / "models.json").read_text(encoding="utf-8"))
    tier_ids = {tier["id"] for tier in models["chat_tiers"]}
    assert models["default_tier"] in tier_ids
    for tier in models["chat_tiers"]:
        for key in ("repo", "file", "local_name", "n_gpu_layers", "ctx"):
            assert key in tier, f"{tier['id']} 缺少字段 {key}"
        assert tier["local_name"].isascii(), "落盘文件名需为 ASCII，避免中文路径问题"


def test_plugins_are_registered() -> None:
    """Every capability named in the docs must actually be resolvable."""
    import plugs  # noqa: F401  (importing the package loads the built-in plugins)

    from core.registry import (
        KIND_EMBEDDER,
        KIND_LLM,
        KIND_MEMORY_STORE,
        KIND_RERANKER,
        KIND_RETRIEVER,
        KIND_STT,
        KIND_TTS,
        KIND_VECTOR_STORE,
        registry,
    )

    for kind, expected in (
        (KIND_LLM, {"mock", "llamacpp", "ollama", "openai", "vllm"}),
        (KIND_EMBEDDER, {"hash", "llamacpp"}),
        (KIND_RERANKER, {"heuristic", "none"}),
        (KIND_VECTOR_STORE, {"sqlite", "memory"}),
        (KIND_MEMORY_STORE, {"sqlite"}),
        (KIND_RETRIEVER, {"hybrid"}),
        (KIND_STT, {"mock"}),
    ):
        missing = expected - set(registry.names(kind))
        assert not missing, f"{kind} 缺少实现: {missing}"


def test_llama_cpp_assets_include_cudart_runtime() -> None:
    """The installer must fetch BOTH the binaries and the CUDA runtime package.

    ggml-org publishes two families of Windows assets:

      `llama-b11067-bin-win-cuda-13.4-x64.zip`   binaries (contains llama-server.exe)
      `cudart-llama-bin-win-cuda-13.4-x64.zip`   CUDA runtime DLLs (cudart64_*.dll,
                                                   cublas64_*.dll, cublasLt64_*.dll)

    Neither alone is enough. Installing only the first produces a llama-server that
    **starts fine and silently runs on the CPU**: `--list-devices` prints `(none)`
    and every token is computed on the CPU. This project lost hours to exactly that --
    a 27B model ran at 3.6 tok/s and `--n-gpu-layers` appeared to have no effect,
    because the CUDA backend could never load without the runtime DLLs.

    The test pins three things: both patterns exist, the runtime pattern is tied to the
    binary release tag, and arm64 assets are excluded everywhere.
    """
    import re

    src = Path(__file__).resolve().parents[1] / "scripts" / "install.ps1"
    text = src.read_text(encoding='utf-8')

    assert '$plan = @(' in text, 'install.ps1 应有资产选择表'
    cuda_rows = re.findall(r"@\{\s*Name\s*=\s*'CUDA[^}]*\}", text)
    assert cuda_rows, '未找到 CUDA 资产行'
    for row in cuda_rows:
        assert re.search(r"Bin\s*=\s*'([^']+)'", row), f'CUDA 行缺少 Bin 模式：{row}'
        cudart_match = re.search(r"Cudart\s*=\s*'([^']+)'", row)
        assert cudart_match and cudart_match.group(1) != '$null', (
            f'CUDA 行必须同时指定 cudart 运行时包，否则会被静默降级到 CPU：{row}'
        )

    all_bin = re.findall(r"Bin\s*=\s*'([^']+)'", text)
    all_cudart = [m for m in re.findall(r"Cudart\s*=\s*'([^']+)'", text) if m != '$null']
    assert all_bin and all_cudart

    binary_patterns = [re.compile(p) for p in all_bin]
    runtime_patterns = [re.compile(p) for p in all_cudart]

    for good in (
        'llama-b11067-bin-win-cuda-13.4-x64.zip',
        'llama-b11067-bin-win-cuda-12.4-x64.zip',
        'llama-b11067-bin-win-vulkan-x64.zip',
        'llama-b11067-bin-win-cpu-x64.zip',
    ):
        assert any(rx.match(good) for rx in binary_patterns), f'应匹配但没匹配：{good}'
    for bad in (
        'llama-b11067-bin-win-cuda-13.4-arm64.zip',
        'llama-b11067-bin-win-cpu-arm64.zip',
        'llama-b11067-bin-win-opencl-adreno-arm64.zip',
    ):
        assert not any(rx.match(bad) for rx in binary_patterns), f'不应匹配 arm64：{bad}'

    assert any(rx.match('cudart-llama-bin-win-cuda-13.4-x64.zip') for rx in runtime_patterns)
    assert any(rx.match('cudart-llama-bin-win-cuda-12.4-x64.zip') for rx in runtime_patterns)
    assert not any(rx.match('cudart-llama-bin-win-cuda-13.4-arm64.zip') for rx in runtime_patterns)

    assert '-Tag $candidate.Tag' in text, 'cudart 包必须锁定到与主包同一个 release'
    assert '--list-devices' in text, '安装后必须检查 GPU 是否被识别（否则静默降级到 CPU）'
    assert 'CUDA|Vulkan|ROCm|Metal' in text, 'GPU 检查应覆盖各后端而非只认 CUDA'


def test_powershell_scripts_exist_with_required_contract() -> None:
    """Every script the docs tell the user to run must exist and be BOM-prefixed.

    Windows PowerShell 5.1 decodes a BOM-less ``.ps1`` using the ANSI code page, which
    silently corrupts Chinese text — the project hit this twice, once producing
    garbled output and once producing a file the parser rejected. A missing BOM is
    therefore a build error, not a style nit.
    """
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    expected = {
        "common.ps1",
        "fix-encoding.ps1",
        "install.ps1",
        "install-tts.ps1",
        "install-gptsovits.ps1",
        "download-models.ps1",
        "start-models.ps1",
        "start-all.ps1",
        "start-pet.ps1",
        "build-pet.ps1",
        "train-tts.ps1",
        "stop.ps1",
        "verify.ps1",
        "selftest.ps1",
    }
    present = {path.name for path in scripts.glob("*.ps1")}
    missing = expected - present
    assert not missing, f"缺少脚本：{sorted(missing)}"

    for name in sorted(expected):
        data = (scripts / name).read_bytes()
        assert data.startswith(b"\xef\xbb\xbf"), (
            f"{name} 缺少 UTF-8 BOM：PowerShell 5.1 会按 ANSI 解码，中文会损坏"
        )
        assert len(data) > 200, f"{name} 内容异常短"

    # The Python helpers the scripts shell out to must exist too.
    tests_dir = root / "tests"
    for helper in ("verify_env.py", "check_module.py", "check_imports.py", "verify_models.py"):
        assert (tests_dir / helper).exists(), f"缺少脚本依赖：tests/{helper}"


def test_scripts_resolve_the_interpreter_instead_of_hardcoding_it(project_root) -> None:
    """A hardcoded interpreter path silently runs the *other* Python.

    The project moved from ``venvs\\main`` to the conda environment ``chatbot``. Any
    script still pointing at the old path would keep working — against an environment
    where the dependencies were never installed — and report green results from stale
    code. Resolution order must therefore live in exactly one place (``Get-Python``).
    """
    scripts = project_root / "scripts"
    for name in ("selftest.ps1", "start-all.ps1", "start-models.ps1", "verify.ps1", "install.ps1"):
        text = (scripts / name).read_text(encoding="utf-8-sig")
        assert "venvs\\main\\Scripts\\python.exe" not in text, (
            f"{name} 里写死了 venvs\\main 的解释器路径；应调用 Get-Python"
        )

    common = (scripts / "common.ps1").read_text(encoding="utf-8-sig")
    assert "function Get-CondaPython" in common
    assert "CHATBOT_PYTHON" in common, "需要能用环境变量强制指定解释器"
    # conda 环境必须排在 venvs\main 之前：现在是 conda 环境 chatbot 在跑。
    conda_at = common.index("Get-CondaPython -Name 'chatbot'")
    venv_at = common.index("venvs\\main\\Scripts\\python.exe")
    assert conda_at < venv_at, "conda 环境 chatbot 应优先于 venvs\\main"
    assert common.index("$env:CHATBOT_PYTHON") < conda_at, "显式指定必须最优先"


def test_dependencies_have_one_source_of_truth(project_root) -> None:
    """requirements files feed both install.ps1 and manual (conda) setup.

    The list used to be inlined in ``install.ps1``, so anyone creating the environment by
    hand (``conda create -n chatbot``) had no way to know what to install, and the two
    copies drifted.
    """
    core = (project_root / "requirements.txt").read_text(encoding="utf-8")
    speech = (project_root / "requirements-speech.txt").read_text(encoding="utf-8")
    for package in ("fastapi", "uvicorn", "httpx", "pydantic", "PyYAML", "websockets", "pytest"):
        assert package.lower() in core.lower(), f"requirements.txt 缺少 {package}"
    for package in ("faster-whisper", "edge-tts", "soundfile"):
        assert package in speech, f"requirements-speech.txt 缺少 {package}"

    install = (project_root / "scripts" / "install.ps1").read_text(encoding="utf-8-sig")
    assert "requirements.txt" in install and "requirements-speech.txt" in install
    assert "'fastapi>=0.110'" not in install, "install.ps1 不应再内联依赖清单"


def test_desktop_pet_and_training_tooling_ship_together(project_root) -> None:
    """The pet exe and the fine-tuning pipeline depend on files that are easy to drop.

    Both features are "run this one command" for the user, which means a missing helper
    file is only discovered halfway through a 12GB install — or when the pet fails to
    start. This pins the entry points and the module wiring.
    """
    for relative in (
        "run_pet.py",
        "src/pet/window.py",
        "src/pet/client.py",
        "src/pet/audio.py",
        "tools/make_pet_icon.py",
        "tools/tts_dataset.py",
        "tools/check_gptsovits.py",
        "tools/download_gptsovits_weights.py",
        "tools/download_file.py",
        "tools/download_whisper.py",
        "src/plugs/gpt_sovits_tts.py",
        "docs/训练音色.md",
    ):
        assert (project_root / relative).exists(), f"缺少 {relative}"

    # The GPT-SoVITS backend must be re-exported from speech.tts (build_tts imports it
    # from there) and registered under the name personas use.
    plugin = (project_root / "src" / "plugs" / "gpt_sovits_tts.py").read_text(encoding="utf-8")
    assert '@register(KIND_TTS, "gpt_sovits")' in plugin
    tts = (project_root / "src" / "speech" / "tts.py").read_text(encoding="utf-8")
    assert "GPTSoVITSTTSBackend" in tts

    # The dataset tool must write the format GPT-SoVITS actually consumes.
    dataset = (project_root / "tools" / "tts_dataset.py").read_text(encoding="utf-8")
    assert '"|"' in dataset or "|{speaker}|{language}|" in dataset
    assert "dataset.list" in dataset


def test_python_files_must_not_have_a_bom() -> None:
    """The mirror image of the PowerShell rule: Python source must be BOM-free.

    ``compile()`` rejects a leading U+FEFF with "invalid non-printable character", so a
    BOM added to a ``.py`` file breaks the whole build. This was hit for real while
    fixing the PowerShell encoding issue: the BOM fix was applied with a byte-level
    ``WriteAllText(..., UTF8Encoding($true))`` and swept up a ``.py`` file too.
    """
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for base in ("src", "tests", "scripts"):
        for path in sorted((root / base).rglob("*.py")):
            if path.read_bytes().startswith(b"\xef\xbb\xbf"):
                offenders.append(str(path.relative_to(root)))
    assert not offenders, f"这些 Python 文件带 BOM，会导致 SyntaxError：{offenders}"


def test_python_sources_all_compile() -> None:
    """Catch a syntax error in any shipped source without importing it.

    Imports would need the whole dependency set; ``compile()`` needs nothing, so this
    runs everywhere and names the exact file and line when something is broken.
    """
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    checked = 0
    for base in ("src", "tests", "scripts"):
        for path in sorted((root / base).rglob("*.py")):
            checked += 1
            try:
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            except SyntaxError as exc:
                failures.append(f"{path.relative_to(root)}:{exc.lineno}: {exc.msg}")
    assert checked > 20, f"只扫描到 {checked} 个文件，路径可能不对"
    assert not failures, "存在语法错误：\n" + "\n".join(failures)


def test_powershell_scripts_have_no_encoding_corruption(project_root) -> None:
    """Detect the Chinese-text corruption that an editor round-trip causes.

    The project broke its own PowerShell scripts three times by reading them with
    ``Get-Content`` (ANSI code page on Windows PowerShell 5.1) and writing them back as
    UTF-8. The damage is silent: the file still parses, but a Chinese string becomes
    mojibake or loses a character, so the user sees garbled output.

    Detection is deliberately based on facts that are unambiguous:

    * a U+FFFD replacement character,
    * the file not being valid UTF-8 at all,
    * a run of CJK Extension-A characters (U+3400–U+4DBF), which is the signature of
      UTF-8 bytes reinterpreted as GBK — ordinary Chinese text never uses that block,
    * a required Chinese phrase being absent (catches a character being dropped).

    Repeated-character matching is **not** used: ``项目目录``, ``超时时`` and
    ``一一对应`` are all legitimate, and flagging them produced nothing but false alarms.
    """
    scripts = project_root / "scripts"
    required_phrases = {
        "start-all.ps1": ["项目目录", "已启动"],
        "start-models.ps1": ["启动对话模型"],
        "install.ps1": ["环境安装完成", "磁盘空间"],
        "verify.ps1": ["环境与链路自检"],
        "selftest.ps1": ["全量验证"],
        "download-models.ps1": ["磁盘空间检查"],
        "stop.ps1": ["停止网页服务"],
        "common.ps1": ["路径与公共函数"],
    }

    problems: list[str] = []
    for path in sorted(scripts.glob("*.ps1")):
        raw = path.read_bytes()
        body = raw[3:] if raw.startswith(b"\xef\xbb\xbf") else raw

        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            problems.append(f"{path.name}: 不是合法 UTF-8（{exc}）")
            continue

        if "\ufffd" in text:
            problems.append(f"{path.name}: 含 U+FFFD 替换字符")
        extension_a = sum(1 for ch in text if "\u3400" <= ch <= "\u4dbf")
        if extension_a > 3:
            problems.append(f"{path.name}: 疑似 GBK 误码（{extension_a} 个扩展区字符）")

        for phrase in required_phrases.get(path.name, []):
            if phrase not in text:
                problems.append(f"{path.name}: 缺少中文串 {phrase!r}（可能有字符被吞）")

    assert not problems, "PowerShell 脚本存在编码损坏：\n" + "\n".join(problems)


def test_local_http_clients_bypass_the_system_proxy() -> None:
    """Every client that talks to a local service must set ``trust_env=False``.

    Windows boxes running Clash/V2Ray set a system-wide proxy (measured here:
    ``127.0.0.1:7890``). httpx reads it via ``trust_env`` and therefore routes
    requests for ``127.0.0.1:8080`` through the proxy. The proxy cannot reach a
    localhost-only port, so it answers **502** — and the application reports
    "model server broken" / "embedding service unavailable" while the model server is
    perfectly healthy. This cost hours: the give-away was that a *dead* port also
    returned 502 instead of a connection error.

    ``proxy=None`` does **not** fix it (measured: still 502). Only
    ``trust_env=False`` yields the real ``ConnectError``.
    """
    root = Path(__file__).resolve().parents[1] / "src"
    clients = {
        "llm/openai_compat.py": "chat completions",
        "llm/embed.py": "embeddings and reranking",
        "speech/tts.py": "OpenAI-compatible TTS",
    }
    problems: list[str] = []
    for relative, purpose in clients.items():
        text = (root / relative).read_text(encoding="utf-8")
        if "httpx.AsyncClient(" not in text and "httpx.Client(" not in text:
            continue
        if "trust_env=False" not in text:
            problems.append(f"{relative}（{purpose}）缺少 trust_env=False")
    assert not problems, (
        "这些客户端会走系统代理，导致本机服务被代理拦截并返回 502：\n" + "\n".join(problems)
    )


def test_no_inline_python_in_powershell_scripts() -> None:
    """Regression: inline ``python -c`` with non-ASCII breaks on Windows.

    The command line is passed through the ANSI code page, so a Chinese string inside
    an inline snippet reaches Python corrupted and raises SyntaxError. Every Python
    invocation carrying non-ASCII must therefore be a real file.

    The danger is precisely *non-ASCII content inside the inline snippet*: the command
    line is passed through the ANSI code page, so Chinese arrives corrupted and Python
    raises SyntaxError. This is what broke ``install.ps1`` (a Chinese here-string) and
    ``selftest.ps1`` (a Chinese one-liner); both were real failures.

    A short **pure-ASCII** one-liner is safe and is therefore still allowed, e.g.
    ``python -c "import sys; print(sys.version_info >= (3, 10))"``. The check rejects:

    * a here-string after ``-c`` (always multi-line, always risky), and
    * any inline snippet whose text contains non-ASCII characters.
    """
    import re

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    here_string = re.compile(r"-c\s+@['\"]")
    inline_code = re.compile(r"""-c\s+(?P<q>['"])(?P<body>.*?)(?P=q)""")
    is_ascii = lambda text: all(ord(ch) < 128 for ch in text)

    offenders: list[str] = []
    for path in sorted(scripts.glob("*.ps1")):
        text = path.read_text(encoding="utf-8")
        for match in here_string.finditer(text):
            line_no = text[: match.start()].count("\n") + 1
            offenders.append(f"{path.name}:{line_no} (here-string)")
        for match in inline_code.finditer(text):
            body = match.group("body")
            if is_ascii(body):
                continue  # short ASCII probe: safe, and sometimes the clearest form
            line_no = text[: match.start()].count("\n") + 1
            offenders.append(f"{path.name}:{line_no} (内联代码含非 ASCII 字符)")

    assert not offenders, (
        "PowerShell 内联 Python 片段含非 ASCII 内容（命令行按 ANSI 代码页传递，中文会被"
        "破坏成 SyntaxError）：" + ", ".join(offenders) + "；请把逻辑放到 tests/*.py 里。"
    )


def test_registry_kind_constants_are_complete() -> None:
    """Guard the mistake that silently breaks whole subsystems.

    Decorating with a kind constant that does not exist in ``core.registry`` is an
    ImportError at runtime — every module that imports the decorated file dies. This
    test statically scans the source for ``@register(KIND_...)`` usages and checks
    each name is defined and exported.
    """
    import importlib
    import re

    # NOTE: ``import core.registry as x`` binds the *Registry instance*, because the
    # package ``__init__`` re-exports ``registry`` and that attribute shadows the
    # submodule name. ``importlib.import_module`` always returns the module object.
    registry_module = importlib.import_module("core.registry")
    assert type(registry_module).__name__ == "module", type(registry_module).__name__

    src = Path(__file__).resolve().parents[1] / "src"
    pattern = re.compile(r"@register\(\s*(KIND_[A-Z_]+)")
    used: set = set()
    for path in src.rglob("*.py"):
        used.update(pattern.findall(path.read_text(encoding="utf-8")))

    assert used, "未扫描到任何 @register 调用，检查正则是否失效"
    missing = sorted(name for name in used if not hasattr(registry_module, name))
    assert not missing, f"core/registry.py 缺少常量定义: {missing}"

    not_exported = sorted(name for name in used if name not in registry_module.__all__)
    assert not not_exported, f"常量未加入 __all__: {not_exported}"


def test_registry_rejects_unknown_plugin() -> None:
    import plugs  # noqa: F401

    from core.registry import KIND_LLM, registry

    with pytest.raises(KeyError):
        registry.create(KIND_LLM, "does-not-exist")


def test_token_estimator_is_conservative() -> None:
    """The estimator must round *up* in the ways that keep prompts safe.

    Two invariants matter, and each is checked against a sentence whose composition
    is stated explicitly so the expectation cannot drift from the fixture:

    * CJK costs at least ~1 token per character. Measured reality for the sample
      below is 16 han chars + "token" + 4 punctuation = 21 tokens; a Chinese LLM
      tokenizer emits roughly one token per han character plus a few for the rest.
    * Latin costs at least ~1.2 tokens per word (real tokenizers average ~1.3).
    """
    from core.utils import estimate_tokens, truncate_to_tokens

    chinese = "这是一段中文测试文本，用于估算 token 数量。"
    han_chars = sum(1 for ch in chinese if "\u4e00" <= ch <= "\u9fff")
    assert han_chars == 16, f"fixture changed: expected 16 han chars, found {han_chars}"
    estimated = estimate_tokens(chinese)
    assert estimated >= han_chars + 1, f"undercounts CJK: {estimated} for {han_chars} han chars"

    latin = "This is an English sentence used to estimate token counts."
    word_count = len(latin.split())
    assert estimate_tokens(latin) >= word_count * 1.2, estimate_tokens(latin)

    assert estimate_tokens("") == 0
    assert estimate_tokens("a") >= 1
    # Truncation must enforce the budget it was given.
    truncated = truncate_to_tokens(chinese * 20, 20)
    assert estimate_tokens(truncated) <= 20
    assert len(truncated) < len(chinese) * 20
    # A string already inside budget must come back untouched.
    assert truncate_to_tokens(chinese, estimate_tokens(chinese)) == chinese


def test_speakable_text_strips_markdown() -> None:
    from core.utils import speakable_text

    messy = "你好 **世界**，看这段代码：\n```python\nprint(1)\n```\n还有 [链接](http://x.com) 和 `行内代码`。"
    cleaned = speakable_text(messy)
    for token in ("```", "**", "print(1)", "http://x.com", "`"):
        assert token not in cleaned


def test_split_sentences_merges_short_fragments() -> None:
    from core.utils import split_sentences

    parts = split_sentences("你好。今天天气不错！要不要出去走走？")
    assert 1 <= len(parts) <= 3
    assert "".join(parts).replace(" ", "") == "你好。今天天气不错！要不要出去走走？"


def test_http_smoke_writes_to_a_scratch_database(project_root) -> None:
    """The self-test must not pollute the user's real long-term memory.

    ``smoke_http.py`` chats "我叫小明，我喜欢不加糖的美式咖啡" to exercise the extraction
    path. Run against the real ``data/`` it plants those sentences permanently, and the
    assistant afterwards introduces itself to its owner as 小明 — a self-test that
    corrupts the product. The data directory must therefore be redirected to a temp
    directory, and cleaned up afterwards.
    """
    source = (project_root / "tests" / "smoke_http.py").read_text(encoding="utf-8")
    assert "replace(config.paths, data_dir=" in source, "冒烟脚本必须把数据目录改到临时目录"
    assert "scratch.cleanup()" in source, "临时目录要清理"
    assert "tempfile.TemporaryDirectory" in source
