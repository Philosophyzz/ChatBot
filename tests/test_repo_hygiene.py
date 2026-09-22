"""仓库卫生：把"代码"和"用户私人数据"之间的那条线钉死。

这个项目里 models/ 有 33.6GB、vendor/ 5.8GB，而 data/ 是**用户真实的对话与记忆库**。
一旦推上 GitHub，撤回也撤不干净（缓存、fork、clone 都会留）。所以边界不能靠记性：
这里直接问 git 本人 —— 这些路径到底会不会被忽略。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: 绝不允许进入版本库的路径（相对仓库根）。前四个是私人数据，其余是可重新生成的大件。
MUST_BE_IGNORED = (
    "data/memory.sqlite3",
    "data/pet_settings.json",
    "logs/chatbot.jsonl",
    "logs/api.err.log",
    "config/local.yaml",
    "models/gguf/Qwen3.5-9B-Q4_K_M.gguf",
    "models/whisper/large-v3-turbo/model.bin",
    "vendor/GPT-SoVITS/GPT_SoVITS/s2_train.py",
    "bin/llama.cpp/llama-server.exe",
    "venvs/main/Scripts/python.exe",
    "build/ChatBotPet/Analysis-00.toc",
    "dist/ChatBotPet.exe",
    "__pycache__/x.cpython-311.pyc",
    ".agent-teams/state.json",
)

#: 必须被上传的东西（少了它们别人克隆下来跑不起来）。
MUST_BE_TRACKED = (
    "README.md",
    "requirements.txt",
    "config/config.yaml",
    "config/models.json",
    "config/personas.yaml",
    "config/local.yaml.example",
    "scripts/start-all.ps1",
    "scripts/download-models.ps1",
    "src/core/app.py",
    "web/index.html",
    "tools/check_repo_hygiene.py",
)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 - fixed command
        ["git", *args], cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace"
    )


def _require_git_repo() -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("不在 git 仓库里（压缩包解出来的源码），跳过仓库卫生检查")
    if _git("rev-parse", "--is-inside-work-tree").returncode != 0:
        pytest.skip("git 工作区不可用")


def test_sensitive_paths_are_ignored_by_git() -> None:
    """`git check-ignore` 是权威答案：问 git 会不会把这些路径纳入版本库。"""
    _require_git_repo()
    leaked = [path for path in MUST_BE_IGNORED if _git("check-ignore", "-q", path).returncode != 0]
    assert not leaked, (
        "这些路径没有被 .gitignore 覆盖，一旦 git add -A 就会把私人数据推上 GitHub：\n  "
        + "\n  ".join(leaked)
    )


def test_gitignore_exists_and_lists_the_big_ones() -> None:
    path = ROOT / ".gitignore"
    assert path.exists(), "没有 .gitignore 就等于把 models/ 和 data/ 一起上传"
    lines = {line.strip() for line in path.read_text(encoding="utf-8").splitlines()}
    for rule in ("models/", "vendor/", "bin/", "data/", "logs/", "venvs/", "build/", "dist/", "config/local.yaml"):
        assert rule in lines, f".gitignore 缺少关键规则：{rule}"


def test_needed_files_are_not_ignored() -> None:
    """反向检查：别把该上传的文件也忽略掉了（那样别人克隆下来跑不起来）。"""
    _require_git_repo()
    ignored = [path for path in MUST_BE_TRACKED if _git("check-ignore", "-q", path).returncode == 0]
    assert not ignored, "这些文件必须上传，但被忽略了：\n  " + "\n  ".join(ignored)


def test_power_script_encoding_contract_is_documented_for_git() -> None:
    """`.ps1` 必须保持 BOM+CRLF、`.py` 必须不带 BOM —— .gitattributes 要写明，别让 git 猜。"""
    path = ROOT / ".gitattributes"
    assert path.exists(), "缺少 .gitattributes：换行/BOM 约定没有声明，克隆后可能出现乱码脚本"
    text = path.read_text(encoding="utf-8")
    assert "*.ps1" in text and "eol=crlf" in text
    assert "*.py" in text and "eol=lf" in text
