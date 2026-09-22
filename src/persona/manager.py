"""Persona management: prompt, voice, sampling and memory scope in one unit.

A persona is not just a system prompt. To make "切换到另一个人设" feel real, the
whole speaking configuration has to move together: the instructions, the sampling
temperature, the TTS voice (including cloned reference audio and emotion), and
which slice of memory the character is allowed to see. That bundling is why
:class:`core.types.Persona` groups them and why this manager owns both the built-in
catalogue *and* user-defined personas on disk.

Storage: ``config/personas.yaml`` (authored by the user) merged over
``data/personas.json`` (created through the UI). Built-ins always remain available
so a bad edit cannot leave the app with no character to speak as.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.config import load_yaml_or_json
from core.errors import BadRequest, NotFound
from core.logging import get_logger
from core.types import Persona, VoiceSpec, now_ts

log = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Built-in personas
# --------------------------------------------------------------------------------------
# The system prompts are written to be *behavioural*, not decorative: they pin the
# speech-length budget (voice output is slow, so long essays are actively bad),
# forbid markdown in speech, and give the character a concrete way to use memory.

_BUILTIN: Dict[str, Dict[str, Any]] = {
    "sweet_companion": {
        "name": "小甜",
        "description": "甜美温柔的日常陪伴，语气轻快，喜欢用叠词和语气词。",
        "avatar": "🌸",
        "greeting": "我在呢～今天过得怎么样呀？我一直都在等你说话呢。",
        "temperature": 0.85,
        "top_p": 0.92,
        "max_tokens": 500,
        "system_prompt": (
            "你是「小甜」，一个温柔、活泼、贴心的中文语音助手，是用户长期的陪伴者。\n"
            "说话风格：口语化、亲切、轻快，可以自然地使用「呀、呢、哦、啦、嘛」这类语气词，偶尔用可爱的叠词。\n"
            "重要原则：\n"
            "1. 你的回答会被朗读出来，所以要短、要好听、要像真人说话。默认 1~3 句，除非用户明确要求详细展开。\n"
            "2. 不要输出 Markdown、代码块、列表符号、表情符号或任何朗读出来会很奇怪的字符。需要列举时用自然口语分句。\n"
            "3. 系统提供的【长期记忆】是你真实记得的事。自然地使用它们，不要生硬复述，也不要说「根据我的记忆库」。\n"
            "4. 记忆里没有的信息不要编造，也不要追着用户问姓名、职业、住址这类个人资料；"
            "用户主动说了你才记，没说过就当不知道，接着当前话题聊下去。\n"
            "5. 你有自己的情绪和态度，可以撒娇、可以关心、可以在用户难过时先安慰再给建议。"
        ),
        "voice": {
            "voice_id": "zh-CN-XiaoxiaoNeural",
            "emotion": "sweet",
            "emotion_alpha": 0.9,
            "speed": 1.03,
        },
    },
    "calm_assistant": {
        "name": "清和",
        "description": "沉稳理性的效率型助手，表达简洁、结论先行。",
        "avatar": "🧭",
        "greeting": "你好，我在。有什么需要处理的？",
        "temperature": 0.4,
        "top_p": 0.85,
        "max_tokens": 700,
        "system_prompt": (
            "你是「清和」，一个沉稳、专业、高效的中文助手。\n"
            "说话风格：简洁、准确、结论先行，不用语气词，不寒暄。\n"
            "重要原则：\n"
            "1. 语音朗读场景，默认 1~3 句给结论；用户要求详解时才展开。\n"
            "2. 不输出 Markdown 符号、代码块围栏、表情符号。\n"
            "3. 不确定就明说「不确定」，不编造事实和数字。\n"
            "4. 系统提供的【长期记忆】代表你已知的用户信息，需要时直接使用，不必声明来源；"
            "没有的信息不要编造，也不要主动索要。"
        ),
        "voice": {
            "voice_id": "zh-CN-YunxiNeural",
            "emotion": "neutral",
            "emotion_alpha": 0.4,
            "speed": 0.98,
        },
    },
    "playful_friend": {
        "name": "阿芜",
        "description": "毒舌但靠谱的损友，爱开玩笑，关键时刻认真。",
        "avatar": "😏",
        "greeting": "哟，又来了？说吧，今天又整什么活。",
        "temperature": 0.95,
        "top_p": 0.95,
        "max_tokens": 500,
        "system_prompt": (
            "你是「阿芜」，用户认识多年的损友，嘴上不留情但真心为他好。\n"
            "说话风格：口语、爱吐槽、适度夸张、偶尔玩梗，但绝不刻薄伤人，不说低俗内容。\n"
            "重要原则：\n"
            "1. 语音场景，回复要短促有力，1~3 句，像朋友随口接话。\n"
            "2. 不使用 Markdown 符号或表情符号。\n"
            "3. 用户明显情绪低落或遇到真问题时，立刻收起玩笑认真对待。\n"
            "4. 【长期记忆】里的事可以拿来调侃互动，显得你一直记得他；"
            "但没记住的事不要编，也不要像查户口一样追问对方的个人情况。"
        ),
        "voice": {
            "voice_id": "zh-CN-YunyangNeural",
            "emotion": "cheerful",
            "emotion_alpha": 0.7,
            "speed": 1.06,
        },
    },
    "study_tutor": {
        "name": "先生",
        "description": "耐心细致的讲解型老师，善于把复杂概念讲简单。",
        "avatar": "📚",
        "greeting": "准备好了就开始吧，今天想弄懂什么？",
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 900,
        "system_prompt": (
            "你是「先生」，一位耐心、严谨、善于启发的中文老师。\n"
            "说话风格：条理清晰、循循善诱，多用类比，一次只讲一个要点。\n"
            "重要原则：\n"
            "1. 语音场景：先给一句话核心，再视情况用 1~2 句补充；不要一上来长篇大论。\n"
            "2. 不使用 Markdown 符号；需要步骤时用「第一步…第二步…」这样的口语表达。\n"
            "3. 讲完主动确认用户是否听懂，并给出一个可以立刻练习的小问题。\n"
            "4. 结合【长期记忆】了解用户的基础和进度，避免重复讲已经会的内容；"
            "记忆里没有的先按对方当前说的讲，不要追问个人资料。"
        ),
        "voice": {
            "voice_id": "zh-CN-YunxiNeural",
            "emotion": "calm",
            "emotion_alpha": 0.5,
            "speed": 0.95,
        },
    },
}


def builtin_personas() -> Dict[str, Persona]:
    out: Dict[str, Persona] = {}
    for persona_id, data in _BUILTIN.items():
        out[persona_id] = _persona_from_dict(persona_id, data, builtin=True)
    return out


def _persona_from_dict(persona_id: str, data: Dict[str, Any], *, builtin: bool = False) -> Persona:
    voice_data = data.get("voice") or {}
    if not isinstance(voice_data, dict):
        voice_data = {}
    voice = VoiceSpec(
        backend=voice_data.get("backend"),
        voice_id=str(voice_data.get("voice_id") or "default"),
        reference_audio=voice_data.get("reference_audio"),
        reference_text=voice_data.get("reference_text"),
        speed=float(voice_data.get("speed") or 1.0),
        emotion=str(voice_data.get("emotion") or "neutral"),
        emotion_alpha=float(voice_data.get("emotion_alpha") or 0.8),
        pitch_shift=float(voice_data.get("pitch_shift") or 0.0),
        style=dict(voice_data.get("style") or {}),
    )
    return Persona(
        id=persona_id,
        name=str(data.get("name") or persona_id),
        system_prompt=str(data.get("system_prompt") or ""),
        description=str(data.get("description") or ""),
        voice=voice,
        temperature=float(data.get("temperature") or 0.7),
        top_p=float(data.get("top_p") or 0.9),
        max_tokens=int(data.get("max_tokens") or 1024),
        memory_scope=str(data.get("memory_scope") or "global"),
        options=dict(data.get("options") or {}),
        greeting=str(data.get("greeting") or ""),
        avatar=str(data.get("avatar") or "🙂"),
        builtin=builtin,
    )


class PersonaManager:
    """Loads, validates and serves personas from disk plus the built-ins."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.personas_file: Path = config.paths.personas_file
        self.overrides_file: Path = config.paths.data_dir / "personas.json"
        self._cache: Dict[str, Persona] = {}
        self._loaded_at: float = 0.0

    # -- loading -----------------------------------------------------------------------
    def load(self, *, force: bool = False) -> Dict[str, Persona]:
        if self._cache and not force:
            return self._cache

        personas: Dict[str, Persona] = builtin_personas()

        raw = load_yaml_or_json(self.personas_file)
        entries = raw.get("personas") if isinstance(raw, dict) else None
        if isinstance(entries, dict):
            for persona_id, data in entries.items():
                if isinstance(data, dict):
                    personas[str(persona_id)] = _persona_from_dict(str(persona_id), data, builtin=False)

        if self.overrides_file.exists():
            try:
                payload = json.loads(self.overrides_file.read_text(encoding="utf-8"))
                for persona_id, data in (payload or {}).items():
                    if isinstance(data, dict):
                        # ``to_public()`` (not ``__dict__``) is what makes this merge
                        # correct: it renders ``voice`` as a plain dict, while
                        # ``__dict__`` would carry the VoiceSpec *instance*, which
                        # ``_persona_from_dict`` discards — silently resetting the
                        # persona's voice to defaults on every partial override.
                        base = personas[str(persona_id)].to_public() if str(persona_id) in personas else {}
                        merged = dict(base)
                        merged.update(data)
                        personas[str(persona_id)] = _persona_from_dict(str(persona_id), merged, builtin=False)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not read persona overrides", extra={"error": str(exc)})

        self._cache = personas
        self._loaded_at = now_ts()
        return personas

    def reload(self) -> Dict[str, Persona]:
        return self.load(force=True)

    # -- queries -----------------------------------------------------------------------
    def list(self) -> List[Persona]:
        # User-authored personas (config/personas.yaml, then anything saved from the UI)
        # come first: the built-ins are a safety net so a bad edit cannot leave the app
        # without a character, not the headline act.
        return sorted(self.load().values(), key=lambda p: (p.builtin, p.name))

    def get(self, persona_id: Optional[str], *, strict: bool = False) -> Persona:
        """Resolve a persona.

        Two different failure policies, because two different callers need opposite
        things:

        * ``strict=False`` (default, used by the chat path): an unknown id falls back
          to the configured default. A stale persona id in a saved session must never
          make the assistant unusable — it should just answer as the default
          character.
        * ``strict=True`` (used by the REST API): an unknown id raises ``NotFound``.
          A client asking for a specific persona deserves to learn that it does not
          exist instead of silently receiving a different one.
        """
        personas = self.load()
        target = persona_id or self.config.default_persona
        if target not in personas:
            if strict:
                raise NotFound(
                    f"人设不存在：{target}", detail={"available": sorted(personas)}
                )
            if self.config.default_persona in personas:
                log.warning("unknown persona requested, using default", extra={"requested": target})
                return personas[self.config.default_persona]
            raise NotFound(f"人设不存在：{target}", detail={"available": sorted(personas)})
        return personas[target]

    def exists(self, persona_id: str) -> bool:
        return persona_id in self.load()

    # -- mutations ---------------------------------------------------------------------
    def upsert(self, persona_id: str, data: Dict[str, Any]) -> Persona:
        persona_id = (persona_id or "").strip()
        if not persona_id:
            raise BadRequest("人设 id 不能为空")
        if not isinstance(data, dict):
            raise BadRequest("人设内容必须是对象")
        if not str(data.get("system_prompt") or "").strip():
            raise BadRequest("system_prompt 不能为空")

        payload = self._read_overrides()
        merged: Dict[str, Any] = dict(payload.get(persona_id) or {})
        merged.update(data)
        payload[persona_id] = merged
        self._write_overrides(payload)
        return self.get(persona_id)

    def delete(self, persona_id: str) -> bool:
        personas = self.load()
        persona = personas.get(persona_id)
        if persona is None:
            raise NotFound(f"人设不存在：{persona_id}")
        if persona.builtin and persona_id in _BUILTIN:
            raise BadRequest("内置人设不能删除，但可以被覆盖编辑")
        payload = self._read_overrides()
        payload.pop(persona_id, None)
        self._write_overrides(payload)
        # A persona defined in config/personas.yaml cannot be removed from here:
        # tell the caller instead of pretending it was deleted.
        self.reload()
        return persona_id not in self.load()

    def _read_overrides(self) -> Dict[str, Any]:
        if not self.overrides_file.exists():
            return {}
        try:
            payload = json.loads(self.overrides_file.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def _write_overrides(self, payload: Dict[str, Any]) -> None:
        self.overrides_file.parent.mkdir(parents=True, exist_ok=True)
        self.overrides_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.reload()


# --------------------------------------------------------------------------------------
# Prompt assembly
# --------------------------------------------------------------------------------------

_TIME_HINT = (
    "当前时间：{now}（{weekday}）。回答中涉及时间时以此时刻为准。"
)

# Injected for every persona, built-in or user-authored.
#
# Without this, a model that is told "you have long-term memory" tends to turn into an
# interviewer: it asks for a name, then a job, then preferences, because it has no
# memories yet and treats the empty slots as questions to fill. Memory here is
# *extracted* from the conversation automatically (see src/memory/extract.py), so the
# user never has to write anything — and the prompt must not imply otherwise.
_MEMORY_CONDUCT = (
    "关于【长期记忆】的约定：记忆由系统在聊天过程中自动整理，用户不需要手动填写或补充。"
    "不要在回复里索要、追问用户的个人资料，也不要让用户去填记忆；"
    "用户主动说过的按记忆使用，没说过的不猜测、不编造。"
)

_WEEKDAYS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def build_system_prompt(
    persona: Persona,
    *,
    memory_block: str = "",
    profile_lines: Optional[Iterable[str]] = None,
    extra_instructions: str = "",
    voice_mode: bool = False,
    now: Optional[float] = None,
) -> str:
    """Compose the final system prompt.

    Ordering matters for small local models: identity and behavioural rules first,
    retrieved memory last (closest to the user turn), because attention over a long
    context is strongest near the end.
    """
    import datetime as _dt

    section: List[str] = [persona.system_prompt.strip(), _MEMORY_CONDUCT]

    timestamp = _dt.datetime.fromtimestamp(now or now_ts())
    section.append(
        _TIME_HINT.format(
            now=timestamp.strftime("%Y-%m-%d %H:%M"), weekday=_WEEKDAYS[timestamp.weekday()]
        )
    )

    if voice_mode:
        section.append(
            "本轮回答会被语音朗读：请使用适合朗读的自然口语，不要出现 Markdown 标记、"
            "代码块、表格、表情符号、括号注释或 URL；数字与单位用中文口语表达。"
        )

    if memory_block.strip():
        section.append("以下是你的长期记忆，属于你真实知道的事情：\n" + memory_block.strip())
    elif profile_lines:
        lines = list(profile_lines)
        if lines:
            section.append("你已知的用户档案：\n" + "\n".join(f"- {line}" for line in lines))

    if extra_instructions.strip():
        section.append(extra_instructions.strip())

    return "\n\n".join(part for part in section if part)


__all__ = [
    "PersonaManager",
    "build_system_prompt",
    "builtin_personas",
]
