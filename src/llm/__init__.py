"""LLM backends: one OpenAI-compatible client plus a deterministic mock."""

from llm.openai_compat import OpenAICompatBackend
from llm.mock import MockLLMBackend

__all__ = ["MockLLMBackend", "OpenAICompatBackend"]
