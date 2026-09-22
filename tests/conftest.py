"""Test suite shared fixtures.

Import strategy: the application lives under ``src/`` (flat modules like ``core``,
``memory``, ``llm``), so ``src`` is put on ``sys.path`` here rather than requiring an
installable package. That keeps ``python -m pytest`` working straight from a clone.

Async tests deliberately avoid ``pytest-asyncio``: each test drives its coroutine
with ``asyncio.run`` via the ``run`` fixture. One less plugin to install, and the
failure output stays readable.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

T = TypeVar("T")


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture
def run():
    """Drive an async coroutine to completion inside a sync test."""

    def _run(coro: Awaitable[T]) -> T:
        return asyncio.run(coro)  # type: ignore[arg-type]

    return _run


@pytest.fixture
def temp_config(tmp_path: Path):
    """A fully isolated AppConfig rooted in a temp directory.

    Every test that touches the database or model cache uses this, so tests never
    read or write the user's real ``data/`` directory.
    """
    from core.config import AppConfig, Paths

    config = AppConfig(paths=Paths(root=tmp_path))
    config.paths.ensure()
    config.llm.backend = "mock"
    config.llm.mock = True
    config.memory.embedder = "hash"
    config.memory.fallback_embedder = "hash"
    config.memory.reranker = "heuristic"
    config.memory.extract_async = False
    config.memory.consolidate_enabled = False
    config.memory.dim = 256
    config.speech.stt_backend = "mock"
    config.speech.tts_backend = "router"
    config.speech.tts_preference = ["tone"]
    config.debug = True
    return config


@pytest.fixture
def mock_llm():
    from llm.mock import MockLLMBackend

    return MockLLMBackend(delay_ms=0)


@pytest.fixture
def memory_stack(temp_config, mock_llm, run):
    """A ready-to-use (store, vectors, retriever, extractor, engine) tuple."""
    from memory.db import Database
    from memory.engine import MemoryEngine
    from memory.store import MemoryStore
    from memory.vector import SqliteVectorStore
    from llm.embed import HashingEmbedder

    async def _build():
        db = Database(temp_config.paths.db_path)
        db.init()
        store = MemoryStore(db)
        embedder = HashingEmbedder(dim=temp_config.memory.dim)
        vectors = SqliteVectorStore(db)
        engine = await MemoryEngine.create(
            temp_config, llm=mock_llm, db=db, embedder=embedder, vectors=vectors
        )
        return engine

    engine = run(_build())
    yield engine
    run(engine.close())
