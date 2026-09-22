"""Memory extraction: turning raw dialogue into structured, durable memory.

This is the "write path" of the memory system and the piece that most determines
whether the assistant feels like it actually *knows* you. It runs asynchronously
so it never adds latency to a reply, and it is idempotent per message so a retry
or a restart cannot double-insert.

Pipeline for one (user, assistant) turn:

1. **Extract** — a constrained JSON-schema completion pulls out atomic facts,
   entities (+aliases), typed relations, and profile attributes. A deterministic
   rule-based fallback takes over when the model is unavailable or returns junk,
   so memory never silently stops working.
2. **Resolve** — entity mentions are mapped to graph nodes (creating aliases), so
   "小明", "我" and "用户" do not become three unrelated people.
3. **Deduplicate** — exact duplicates collapse by content hash; near duplicates
   are detected by embedding similarity against same-subject memories.
4. **Reconcile** — the interesting part: a new fact that contradicts an existing
   one (same subject+predicate, low object similarity) *closes the old one's
   validity window* and links the chain via ``supersedes``/``superseded_by``.
   This is what keeps long-term memory from accumulating mutually exclusive
   claims — the classic failure mode of naive "append everything" designs.
5. **Persist** — memories, vectors, relations and the profile projection are all
   written, and the turn is logged so it is never processed twice.

Extraction prompts are intentionally explicit about the JSON contract and about
what *not* to store, because a small local model follows a short, concrete
instruction far better than a long one.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.logging import get_logger
from core.registry import KIND_EXTRACTOR, register
from core.types import LLMRequest, MemoryItem, MemoryKind, Message, Role, now_ts
from core.utils import truncate_to_tokens
from llm.embed import cosine
from memory.store import MemoryStore, content_hash

log = get_logger(__name__)

# --------------------------------------------------------------------------------------
# Schema (kept as one constant so the prompt and the validator cannot drift apart)
# --------------------------------------------------------------------------------------

EXTRACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["fact", "preference", "profile", "episode", "relation"],
                    },
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "importance": {"type": "number"},
                    "confidence": {"type": "number"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "valid_from": {"type": ["number", "null"]},
                },
                "required": ["content", "kind"],
            },
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "summary": {"type": "string"},
                },
                "required": ["name"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["subject", "predicate", "object"],
            },
        },
        "profile": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["key", "value"],
            },
        },
    },
    "required": ["facts"],
}

_SYSTEM_PROMPT = """你是一个长期记忆抽取器。阅读一段对话，抽取**值得长期记住**的信息，输出 JSON。

只抽取满足以下任一条件的内容：
1. 关于用户的稳定事实（姓名、职业、所在地、家庭、健康、目标、计划）。
2. 用户的偏好、习惯、厌恶（食物、音乐、作息、交流方式）。
3. 用户与其他人/事物的关系。
4. 用户明确希望你记住的事，或反复出现的重要话题。

**不要**抽取：寒暄、助手自己的话、临时情绪、常识、一次性问题、无法验证的猜测。

输出规范：
- content：一句独立、可独立理解的陈述句（不要代词，把"他/它/那个"还原成具体名称）。
- kind：fact | preference | profile | relation | episode
- subject / predicate / object：当内容是三元组关系时填写，否则可为空字符串。
- importance：0~1，越长期有用越高（姓名、职业 0.9；当天安排 0.3）。
- confidence：0~1，你对这条信息正确性的把握。
- 一条事实一条记录，不要合并多件事。

只输出 JSON，不要解释，不要 Markdown 代码块。"""

_USER_TEMPLATE = """最近的对话上下文（越靠后越新）：
{context}

请抽取本轮（最后一条用户消息 + 助手回复）中值得长期记忆的信息。
输出 JSON，结构为：
{{"facts":[{{"content":"","kind":"fact","subject":"","predicate":"","object":"","importance":0.5,"confidence":0.8,"tags":[]}}],
"entities":[{{"name":"","type":"person","aliases":[],"summary":""}}],
"relations":[{{"subject":"","predicate":"","object":"","confidence":0.8}}],
"profile":[{{"key":"","value":"","confidence":0.8}}]}}"""


# --------------------------------------------------------------------------------------
# JSON recovery
# --------------------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json_loose(text: str) -> Optional[Dict[str, Any]]:
    """Recover a JSON object from a model response that may be wrapped or chatty.

    Local models routinely add prose, code fences, or a trailing comma. Throwing
    away the whole extraction because of a stray backtick would be wasteful, so we
    try progressively more forgiving strategies.
    """
    if not text:
        return None
    candidates: List[str] = []
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return None


# --------------------------------------------------------------------------------------
# Heuristic fallback extractor
# --------------------------------------------------------------------------------------

_PROFILE_RULES: Tuple[Tuple[str, re.Pattern, str], ...] = (
    ("名字", re.compile(r"(?:我(?:的名字)?(?:叫|是)|叫我)\s*([\u4e00-\u9fff A-Za-z0-9]{1,12})"), "名字"),
    ("职业", re.compile(r"我(?:是|在|做)\s*([\u4e00-\u9fff A-Za-z0-9]{2,16}?)(?:工程师|老师|医生|学生|设计师|程序员|经理|律师|会计|老板)"), "职业"),
    ("所在城市", re.compile(r"我(?:住在|在|来自)\s*([\u4e00-\u9fff]{2,10})(?:市|省|区|县)?"), "所在城市"),
    ("年龄", re.compile(r"我\s*(\d{1,2})\s*岁"), "年龄"),
)

_PREFERENCE_RULES: Tuple[re.Pattern, ...] = (
    re.compile(r"我(?:很|最|特别|超)?(?:喜欢|爱|偏爱)\s*([^。！？!?\n，,]{1,24})"),
    re.compile(r"我(?:不喜欢|讨厌|受不了|讨厌吃)\s*([^。！？!?\n，,]{1,24})"),
    re.compile(r"我(?:习惯|通常|一般会)\s*([^。！？!?\n，,]{1,24})"),
)

_REMEMBER_RULES: Tuple[re.Pattern, ...] = (
    re.compile(r"记住[：:，,]?\s*([^。！？!?\n]{2,80})"),
    re.compile(r"帮我记(?:一下|住)[：:，,]?\s*([^。！？!?\n]{2,80})"),
)


def heuristic_extract(user_text: str) -> Dict[str, Any]:
    """Rule-based extraction used when the model cannot be used.

    Deliberately conservative: it only recognises explicit self-statements, so a
    degraded mode produces *less* memory rather than wrong memory.
    """
    text = (user_text or "").strip()
    facts: List[Dict[str, Any]] = []
    profile: List[Dict[str, Any]] = []

    for key, pattern, _label in _PROFILE_RULES:
        match = pattern.search(text)
        if match:
            value = match.group(1).strip()
            if value:
                profile.append({"key": key, "value": value, "confidence": 0.8})
                facts.append(
                    {
                        "content": f"用户的{key}是{value}",
                        "kind": "profile",
                        "subject": "用户",
                        "predicate": key,
                        "object": value,
                        "importance": 0.85,
                        "confidence": 0.8,
                        "tags": ["profile"],
                    }
                )

    for pattern in _PREFERENCE_RULES:
        for match in pattern.finditer(text):
            value = match.group(1).strip(" ，,。.")
            if len(value) < 2:
                continue
            negative = "不喜欢" in match.group(0) or "讨厌" in match.group(0) or "受不了" in match.group(0)
            verb = "不喜欢" if negative else "喜欢"
            facts.append(
                {
                    "content": f"用户{verb}{value}",
                    "kind": "preference",
                    "subject": "用户",
                    "predicate": verb,
                    "object": value,
                    "importance": 0.6,
                    "confidence": 0.7,
                    "tags": ["preference"],
                }
            )

    for pattern in _REMEMBER_RULES:
        for match in pattern.finditer(text):
            value = match.group(1).strip(" ，,。.")
            if len(value) < 2:
                continue
            facts.append(
                {
                    "content": value,
                    "kind": "episode",
                    "importance": 0.8,
                    "confidence": 0.9,
                    "tags": ["explicit"],
                }
            )

    return {"facts": facts[:8], "entities": [{"name": "用户", "type": "person", "aliases": ["我"]}], "relations": [], "profile": profile}


# --------------------------------------------------------------------------------------
# Extraction result
# --------------------------------------------------------------------------------------


@dataclass
class ExtractionOutcome:
    created: int = 0
    superseded: int = 0
    deduplicated: int = 0
    entities: int = 0
    relations: int = 0
    profile_updates: int = 0
    method: str = "llm"
    memory_ids: List[str] = field(default_factory=list)
    error: str = ""

    def to_public(self) -> Dict[str, Any]:
        return {
            "created": self.created,
            "superseded": self.superseded,
            "deduplicated": self.deduplicated,
            "entities": self.entities,
            "relations": self.relations,
            "profile_updates": self.profile_updates,
            "method": self.method,
            "memory_ids": list(self.memory_ids),
            "error": self.error,
        }


# --------------------------------------------------------------------------------------
# The extractor
# --------------------------------------------------------------------------------------

#: Predicates that hold exactly one value at a time ("functional slots"). Only these may
#: supersede: a person has one name, one city, one job — a new value makes the old one
#: false. Anything else (preferences above all) can hold many values at once, and
#: treating them as functional destroys valid memories. Measured failure: the extractor
#: superseded "用户喜欢喝不加糖的美式咖啡" with "用户喜欢喝热的美式咖啡" — both are
#: true, and the first was silently retired from recall.
_FUNCTIONAL_PREDICATES = frozenset(
    {
        "姓名", "名字", "name", "昵称", "nickname", "年龄", "age", "性别", "gender",
        "生日", "birthday", "所在地", "所在城市", "城市", "居住地", "居住城市", "location",
        "city", "职业", "occupation", "job", "profession", "工作", "公司", "company",
        "employer", "当前项目", "current_project", "当前状态", "status", "时区", "timezone",
        "手机", "电话", "phone", "邮箱", "email",
    }
)

#: Kind names that may take part in supersession.
_FUNCTIONAL_KINDS = frozenset({"profile", "fact"})


def _is_functional_slot(predicate: str, kind: str) -> bool:
    """True when a (predicate, kind) pair can only hold one value."""
    key = (predicate or "").strip()
    if key and key.lower() in {p.lower() for p in _FUNCTIONAL_PREDICATES}:
        return True
    # A structured profile attribute is a slot by definition even if the predicate is
    # not in the table (the model may phrase it 名字/姓名/name inconsistently).
    return (kind or "").strip().lower() == "profile" and bool(key)


@register(KIND_EXTRACTOR, "llm")
class MemoryExtractor:
    """Extracts and reconciles memories for one turn at a time."""

    name = "llm"

    def __init__(
        self,
        store: MemoryStore,
        llm: Any,
        embedder: Any = None,
        vectors: Any = None,
        *,
        config: Any = None,
        scope: str = "global",
    ) -> None:
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.vectors = vectors
        self.config = config
        self.scope = scope
        cfg = getattr(config, "memory", None)
        self.contradiction_threshold = float(getattr(cfg, "contradiction_threshold", 0.86))
        self.max_items = int(getattr(cfg, "max_items_per_turn", 8))
        self.extraction_cfg = getattr(config, "extraction", None)
        self.min_chars = int(getattr(self.extraction_cfg, "min_chars", 6))

    # -- public API --------------------------------------------------------------------
    async def process_turn(
        self,
        *,
        conversation_id: str,
        user_message: Message,
        assistant_message: Optional[Message],
        persona_id: Optional[str] = None,
        scope: str = "",
        force: bool = False,
    ) -> ExtractionOutcome:
        """Extract, reconcile and persist memory for a single conversation turn."""
        scope = scope or self.scope
        if not force and self.store.turn_processed(user_message.id):
            return ExtractionOutcome(method="skipped")

        outcome = ExtractionOutcome()
        user_text = (user_message.content or "").strip()
        if len(user_text) < self.min_chars:
            await self.store.mark_turn(
                conversation_id=conversation_id,
                message_id=user_message.id,
                assistant_id=getattr(assistant_message, "id", None),
                status="skipped",
                error="utterance too short",
            )
            return ExtractionOutcome(method="skipped")

        history = await self.store.recent_dialogue(conversation_id, turns=getattr(self.extraction_cfg, "context_turns", 6))
        payload: Dict[str, Any]
        try:
            payload = await self._extract_with_llm(user_text, assistant_message, history)
            if not payload.get("facts") and not payload.get("profile"):
                # A well-formed but empty response is legitimate ("nothing to
                # remember"); do not silently substitute heuristics for it.
                outcome.method = "llm-empty"
        except Exception as exc:  # noqa: BLE001 - memory must degrade, never break chat
            log.warning("llm extraction failed, using heuristics", extra={"error": str(exc)})
            payload = heuristic_extract(user_text)
            outcome.method = "heuristic"
            outcome.error = str(exc)[:200]

        try:
            await self._persist(
                payload,
                conversation_id=conversation_id,
                user_message=user_message,
                assistant_message=assistant_message,
                persona_id=persona_id,
                scope=scope,
                outcome=outcome,
            )
            await self.store.mark_turn(
                conversation_id=conversation_id,
                message_id=user_message.id,
                assistant_id=getattr(assistant_message, "id", None),
                items_created=outcome.created,
                status="done",
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("persisting extracted memory failed")
            outcome.error = str(exc)[:200]
            await self.store.mark_turn(
                conversation_id=conversation_id,
                message_id=user_message.id,
                status="failed",
                error=str(exc),
            )
        return outcome

    # -- extraction --------------------------------------------------------------------
    async def _extract_with_llm(
        self,
        user_text: str,
        assistant_message: Optional[Message],
        history: Sequence[Tuple[str, str]],
    ) -> Dict[str, Any]:
        context_lines: List[str] = []
        for user_turn, assistant_turn in list(history)[-6:]:
            context_lines.append(f"用户：{truncate_to_tokens(user_turn, 120)}")
            context_lines.append(f"助手：{truncate_to_tokens(assistant_turn, 120)}")
        context_lines.append(f"用户：{truncate_to_tokens(user_text, 400)}")
        if assistant_message is not None and assistant_message.content:
            context_lines.append(f"助手：{truncate_to_tokens(assistant_message.content, 200)}")
        context = "\n".join(context_lines) or f"用户：{user_text}"

        request = LLMRequest(
            messages=[
                Message(role=Role.SYSTEM, content=_SYSTEM_PROMPT),
                Message(role=Role.USER, content=_USER_TEMPLATE.format(context=context)),
            ],
            temperature=float(getattr(self.extraction_cfg, "temperature", 0.1)),
            top_p=0.9,
            max_tokens=int(getattr(self.extraction_cfg, "max_tokens", 900)),
            json_schema=EXTRACTION_SCHEMA,
            extra={"cache_prompt": False},
        )
        completion = await self.llm.complete(request)
        parsed = parse_json_loose(completion.text)
        if parsed is None:
            raise ValueError(f"extractor returned non-JSON output: {completion.text[:160]!r}")
        return self._validate(parsed)

    @staticmethod
    def _validate(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce model output into the expected shape; drop unusable entries."""
        out: Dict[str, Any] = {"facts": [], "entities": [], "relations": [], "profile": []}
        valid_kinds = {k.value for k in MemoryKind}

        for raw in payload.get("facts") or []:
            if not isinstance(raw, dict):
                continue
            content = str(raw.get("content") or "").strip()
            if len(content) < 2:
                continue
            kind = str(raw.get("kind") or "fact").strip().lower()
            if kind not in valid_kinds:
                kind = MemoryKind.FACT.value
            entry = {
                "content": content,
                "kind": kind,
                "subject": _opt_str(raw.get("subject")),
                "predicate": _opt_str(raw.get("predicate")),
                "object": _opt_str(raw.get("object")),
                "importance": _unit(raw.get("importance"), 0.5),
                "confidence": _unit(raw.get("confidence"), 0.7),
                "tags": [str(t)[:24] for t in (raw.get("tags") or []) if str(t).strip()][:6],
                "valid_from": raw.get("valid_from") if isinstance(raw.get("valid_from"), (int, float)) else None,
            }
            out["facts"].append(entry)

        for raw in payload.get("entities") or []:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            if not name:
                continue
            out["entities"].append(
                {
                    "name": name[:64],
                    "type": str(raw.get("type") or "concept").strip()[:32] or "concept",
                    "aliases": [str(a).strip()[:64] for a in (raw.get("aliases") or []) if str(a).strip()][:8],
                    "summary": str(raw.get("summary") or "").strip()[:280],
                }
            )

        for raw in payload.get("relations") or []:
            if not isinstance(raw, dict):
                continue
            subject = str(raw.get("subject") or "").strip()
            predicate = str(raw.get("predicate") or "").strip()
            obj = str(raw.get("object") or "").strip()
            if not (subject and predicate and obj):
                continue
            out["relations"].append(
                {
                    "subject": subject[:64],
                    "predicate": predicate[:48],
                    "object": obj[:64],
                    "confidence": _unit(raw.get("confidence"), 0.7),
                }
            )

        for raw in payload.get("profile") or []:
            if not isinstance(raw, dict):
                continue
            key = str(raw.get("key") or "").strip()
            value = str(raw.get("value") or "").strip()
            if not (key and value):
                continue
            out["profile"].append({"key": key[:48], "value": value[:200], "confidence": _unit(raw.get("confidence"), 0.75)})

        return out

    # -- persistence & reconciliation ---------------------------------------------------
    async def _persist(
        self,
        payload: Dict[str, Any],
        *,
        conversation_id: str,
        user_message: Message,
        assistant_message: Optional[Message],
        persona_id: Optional[str],
        scope: str,
        outcome: ExtractionOutcome,
    ) -> None:
        facts = payload.get("facts") or []
        entities = payload.get("entities") or []
        relations = payload.get("relations") or []
        profile = payload.get("profile") or []
        source_ids = [user_message.id]
        if assistant_message is not None:
            source_ids.append(assistant_message.id)
        ts = now_ts()

        # 1) Entities first: facts reference them, and the resolvers need them to exist.
        #
        # Self-references are canonicalised to a single name so that "我" / "用户" / the
        # user's actual given name all land on ONE graph node. Without this the graph
        # fragmented in practice: 用户 (36 mentions) and 小明 (37) existed as separate
        # people, splitting every relationship between them.
        name_to_id: Dict[str, str] = {}
        for entity in entities[:24]:
            raw_name = entity["name"]
            aliases = list(entity.get("aliases") or [])
            if is_self_reference(raw_name):
                aliases.append(raw_name)
                raw_name = "用户"
                entity["name"] = raw_name
            elif any(is_self_reference(a) for a in aliases):
                # A name the model marked as an alias of "我" is the user's own name:
                # register it as an alias rather than as a new person.
                aliases.append(raw_name)
                raw_name = "用户"
                entity["name"] = raw_name
            try:
                # Resolve through existing aliases BEFORE creating anything, so a second
                # turn that calls the user by name does not mint a duplicate node.
                resolved = await self.store.resolve_entities([raw_name, *aliases], scope=scope)
                existing_id = next((resolved[key] for key in [raw_name, *aliases] if key in resolved), None)
                entity_id = existing_id or await self.store.upsert_entity(
                    raw_name,
                    type=entity.get("type", "concept"),
                    aliases=aliases,
                    summary=entity.get("summary", ""),
                    scope=scope,
                )
                if existing_id:
                    # Still register the new aliases on the node we matched.
                    await self.store.upsert_entity(
                        raw_name, type=entity.get("type", "concept"), aliases=aliases, scope=scope
                    )
                name_to_id[entity["name"]] = entity_id
                outcome.entities += 1
            except Exception as exc:  # noqa: BLE001
                log.debug("entity upsert failed", extra={"error": str(exc), "name": entity.get("name")})

        # 2) Ensure every subject/object mentioned by a fact exists as a node.
        mentioned = {
            value
            for fact in facts
            for value in (fact.get("subject"), fact.get("object"))
            if value and value not in name_to_id
        }
        mentioned.update({r["subject"] for r in relations} | {r["object"] for r in relations})
        for name in list(mentioned)[:24]:
            if name in name_to_id:
                continue
            canonical = "用户" if is_self_reference(name) else name
            if canonical in name_to_id:
                # Two referring expressions for the same node in one turn.
                name_to_id[name] = name_to_id[canonical]
                continue
            try:
                resolved = await self.store.resolve_entities([canonical], scope=scope)
                existing_id = resolved.get(canonical)
                entity_id = existing_id or await self.store.upsert_entity(
                    canonical,
                    type="person" if canonical == "用户" else "concept",
                    aliases=[name] if name != canonical else (),
                    scope=scope,
                )
                name_to_id[name] = entity_id
                name_to_id[canonical] = entity_id
                outcome.entities += 1
            except Exception:  # noqa: BLE001
                continue

        # Rewrite fact subjects so later dedupe compares canonical subjects.
        for fact in facts:
            if fact.get("subject") and is_self_reference(fact["subject"]):
                fact["subject"] = "用户"

        # 3) Facts: dedupe, then reconcile against existing claims.
        vectors_for_new: List[Tuple[str, List[float]]] = []
        for fact in facts[: self.max_items]:
            content = fact["content"]
            kind = fact["kind"]
            # Retrieve near-duplicates once per fact; this is the single most
            # expensive part of the write path, so it stays bounded.
            similar = await self._find_similar(fact, scope=scope)
            duplicate = None
            contradiction = None
            for hit in similar:
                if self._is_duplicate(fact, hit):
                    duplicate = hit
                    break
                if self._contradicts(fact, hit) and (contradiction is None or hit["similarity"] > contradiction["similarity"]):
                    contradiction = hit

            if duplicate is not None:
                outcome.deduplicated += 1
                await self.store.touch_memories([duplicate["item"].id])
                continue

            memory_id = await self.store.add_memory(
                content=content,
                kind=kind,
                subject=fact.get("subject"),
                predicate=fact.get("predicate"),
                object=fact.get("object"),
                importance=fact["importance"],
                confidence=fact["confidence"],
                valid_from=fact.get("valid_from") or ts,
                session_id=conversation_id,
                persona_id=persona_id,
                scope=scope,
                tags=fact.get("tags") or [],
                sources=source_ids,
                supersedes=contradiction["item"].id if contradiction else None,
                dedupe=False,  # already deduplicated above with embeddings
            )
            if memory_id:
                outcome.created += 1
                outcome.memory_ids.append(memory_id)
                if contradiction is not None:
                    outcome.superseded += 1
                    log.info(
                        "memory superseded",
                        extra={
                            "old": contradiction["item"].content[:60],
                            "new": content[:60],
                            "similarity": round(contradiction["similarity"], 3),
                        },
                    )
                if self.embedder is not None:
                    try:
                        vector = await self.embedder.embed_one(content)
                        if vector:
                            vectors_for_new.append((memory_id, vector))
                            if self.vectors is not None:
                                await self.vectors.upsert([memory_id], [vector], [{"kind": kind, "scope": scope}])
                    except Exception as exc:  # noqa: BLE001
                        log.debug("embedding new memory failed", extra={"error": str(exc)})

        # 4) Relations.
        for relation in relations[:16]:
            subject_id = name_to_id.get(relation["subject"])
            object_id = name_to_id.get(relation["object"])
            if not (subject_id and object_id):
                continue
            try:
                await self.store.add_relation(
                    subject_id,
                    relation["predicate"],
                    object_id,
                    confidence=relation["confidence"],
                    source_memory_id=outcome.memory_ids[0] if outcome.memory_ids else None,
                    scope=scope,
                )
                outcome.relations += 1
            except Exception as exc:  # noqa: BLE001
                log.debug("relation insert failed", extra={"error": str(exc)})

        # 5) Profile projection: key/value attributes are cheap to always inject.
        #    Keys are canonicalised first — a language model emits 名字 / 姓名 / name for
        #    the same slot, which produced three competing rows for one attribute.
        for entry in profile[:12]:
            try:
                canonical_key = normalize_profile_key(entry["key"])
                if not canonical_key:
                    continue
                await self.store.upsert_profile(
                    canonical_key,
                    entry["value"],
                    confidence=entry["confidence"],
                    scope=scope,
                    source_memory_id=outcome.memory_ids[0] if outcome.memory_ids else None,
                )
                outcome.profile_updates += 1
            except Exception as exc:  # noqa: BLE001
                log.debug("profile upsert failed", extra={"error": str(exc)})

    def _is_duplicate(self, fact: Dict[str, Any], hit: Dict[str, Any]) -> bool:
        """Decide whether a new claim merely restates an existing one.

        Rules, cheapest first:

        * identical normalised text is always a duplicate;
        * same subject **and** same predicate **and** same object is a restatement
          regardless of wording — this is the case that produced four rows for one
          coffee preference ("用户喜欢喝不加糖的美式咖啡" / "用户最喜欢喝…" /
          "用户喜欢喝美式咖啡，不加糖" / "小明喜欢喝…");
        * otherwise fall back to embedding similarity, but at a threshold that actually
          catches paraphrase (0.90, not 0.955 — measured: the four variants above scored
          well above 0.90 yet below 0.955, so the old cut never fired).
        """
        item: MemoryItem = hit["item"]
        similarity = float(hit.get("similarity") or 0.0)
        if content_hash(item.content) == content_hash(fact.get("content", "")):
            return True

        def norm(value: Any) -> str:
            return str(value or "").strip().lower()

        new_subject, old_subject = norm(fact.get("subject")), norm(item.subject)
        new_predicate, old_predicate = norm(fact.get("predicate")), norm(item.predicate)
        new_object, old_object = norm(fact.get("object")), norm(item.object)
        if new_predicate and new_predicate == old_predicate:
            if new_object and new_object == old_object:
                # Same slot, same filler -> same claim, whatever the surface wording.
                if not new_subject or not old_subject or new_subject == old_subject:
                    return True
        return similarity >= 0.90

    async def _find_similar(self, fact: Dict[str, Any], *, scope: str, limit: int = 6) -> List[Dict[str, Any]]:
        """Find existing memories that may duplicate or contradict this claim.

        Scoping the comparison to the same subject (or high lexical overlap when
        no subject exists) keeps the candidate set tiny while still catching the
        case that matters: two conflicting statements about the same thing.
        """
        out: List[Dict[str, Any]] = []
        subject = fact.get("subject")
        candidates: List[MemoryItem] = []
        if subject:
            candidates = self.store.fetch_active_candidates(scope=scope, limit=200)
            candidates = [c for c in candidates if (c.subject or "") == subject][:40]
        if not candidates:
            hits = self.store.lexical_search_items(fact["content"], scope=scope, limit=limit * 2)
            by_id = await self.store.get_memories([mid for mid, _ in hits])
            candidates = [by_id[mid] for mid, _ in hits if mid in by_id]
        if not candidates:
            return []

        if self.embedder is None:
            # Without an embedder fall back to lexical similarity: still enough to
            # catch the exact-duplicate case, which is the common one.
            from llm.embed import lexical_overlap

            for item in candidates[:limit]:
                out.append({"item": item, "similarity": lexical_overlap(fact["content"], item.content)})
            out.sort(key=lambda entry: entry["similarity"], reverse=True)
            return [entry for entry in out if entry["similarity"] > 0.5]

        try:
            vectors = await self.embedder.embed([fact["content"], *[c.content for c in candidates]])
        except Exception as exc:  # noqa: BLE001
            log.debug("similarity embedding failed", extra={"error": str(exc)})
            return []
        if not vectors:
            return []
        query_vec, *candidate_vecs = vectors
        for item, vec in zip(candidates, candidate_vecs):
            out.append({"item": item, "similarity": cosine(query_vec, vec)})
        out.sort(key=lambda entry: entry["similarity"], reverse=True)
        return out[: max(limit, 4)]

    def _contradicts(self, fact: Dict[str, Any], hit: Dict[str, Any]) -> bool:
        """Decide whether a new claim invalidates an existing one.

        Supersession **retires** the old memory, so a wrong decision silently removes
        something true from recall. The rules are therefore conservative:

        * the two items must be of compatible kinds;
        * the subjects must match (or one be unspecified) — an embedding score alone is
          far too coarse to decide truth;
        * the predicate must be identical, non-empty, and a **functional slot**: one
          that holds a single value at a time (name, city, job). A person has one name,
          so a new name invalidates the old one.
        * **preferences are never functional.** "喜欢不加糖的美式咖啡" and "喜欢热的美式
          咖啡" are both true; the earlier implementation superseded the first with the
          second and quietly dropped a valid memory. Preferences accumulate, and
          consolidation merges genuine duplicates instead.
        * unstructured statements (no predicate) are never contradictions, for the same
          reason.
        """
        item: MemoryItem = hit["item"]
        similarity = float(hit.get("similarity") or 0.0)
        if similarity < 0.62:
            return False

        old_kind = item.kind.value if isinstance(item.kind, MemoryKind) else str(item.kind)
        new_kind = str(fact.get("kind", "fact"))
        if old_kind not in _FUNCTIONAL_KINDS or new_kind not in _FUNCTIONAL_KINDS:
            return False

        old_subject = (item.subject or "").strip()
        new_subject = (fact.get("subject") or "").strip()
        if old_subject and new_subject and old_subject != new_subject:
            return False

        old_predicate = (item.predicate or "").strip()
        new_predicate = (fact.get("predicate") or "").strip()
        if not (old_predicate and new_predicate and old_predicate.lower() == new_predicate.lower()):
            return False
        if not _is_functional_slot(new_predicate, new_kind):
            return False

        old_object = (item.object or "").strip()
        new_object = (fact.get("object") or "").strip()
        if not (old_object and new_object):
            return False
        # Same functional slot, different filler -> the old value is no longer true.
        return old_object != new_object

    async def rescore_all(self, *, scope: str = "") -> int:
        """Recompute embeddings for memories that lack a vector.

        Called by consolidation (and by the maintenance API) so a memory written
        while the embedding server was down is not permanently unsearchable.
        """
        if self.embedder is None or self.vectors is None:
            return 0
        stored = {mid for mid, _blob, _dim in await self.vectors.iter_all()}
        pending = [item for item in self.store.fetch_active_candidates(scope=scope, limit=2000) if item.id not in stored]
        if not pending:
            return 0
        fixed = 0
        batch = 16
        for start in range(0, len(pending), batch):
            chunk = pending[start : start + batch]
            try:
                vectors = await self.embedder.embed([item.content for item in chunk])
            except Exception as exc:  # noqa: BLE001
                log.warning("rescore embedding batch failed", extra={"error": str(exc)})
                break
            ids = [item.id for item in chunk]
            await self.vectors.upsert(ids, vectors, [{"kind": item.kind.value} for item in chunk])
            fixed += len(ids)
        return fixed


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text[:200] if text else None


# --------------------------------------------------------------------------------------
# Canonicalisation helpers
# --------------------------------------------------------------------------------------

#: Profile keys that mean the same attribute. The extractor is a language model: over a
#: few turns it happily emits ``名字``, ``姓名`` and ``name`` for one slot, which showed
#: up in practice as three competing profile rows ("名字=小明", "name=小明", "姓名=张伟").
#: Mapping synonyms onto one canonical key is what makes the profile a stable set of
#: attributes instead of an ever-growing pile.
_PROFILE_KEY_ALIASES: Dict[str, str] = {
    # 姓名
    "name": "姓名", "名字": "姓名", "full_name": "姓名", "昵称": "昵称", "nickname": "昵称",
    # 基本属性
    "age": "年龄", "gender": "性别", "sex": "性别", "生日": "生日", "birthday": "生日",
    "location": "所在地", "city": "所在地", "城市": "所在地", "所在城市": "所在地",
    "居住地": "所在地", "address": "所在地", "居住城市": "所在地",
    "occupation": "职业", "job": "职业", "profession": "职业", "工作": "职业",
    "company": "公司", "employer": "公司",
    # 偏好与状态
    "drink": "饮品偏好", "favorite_drink": "饮品偏好", "喜欢的饮品": "饮品偏好",
    "food": "饮食偏好", "喜欢的食物": "饮食偏好", "饮食": "饮食偏好",
    "hobby": "兴趣爱好", "hobbies": "兴趣爱好", "兴趣": "兴趣爱好", "爱好": "兴趣爱好",
    "music": "音乐偏好", "音乐": "音乐偏好",
    "pet": "宠物", "pets": "宠物", "养的宠物": "宠物",
    "project": "当前项目", "current_project": "当前项目", "项目": "当前项目",
    "status": "当前状态", "当前状态": "当前状态",
    "schedule": "作息习惯", "作息": "作息习惯", "生活习惯": "作息习惯",
    "environment": "环境偏好", "环境偏好": "环境偏好",
    "language": "常用语言", "language_preference": "常用语言",
    "goal": "目标", "goals": "目标",
}

#: Referring expressions that all denote the user. Whoever the assistant is talking to,
#: these must resolve to ONE graph node; otherwise "用户"(36 mentions) and "小明"(37)
#: become two separate people and every relationship splits in half.
_SELF_REFERENCES = frozenset(
    {"用户", "我", "我的", "自己", "本人", "咱", "俺", "你", "对方", "the user", "user"}
)


def normalize_profile_key(key: str) -> str:
    """Map a model-emitted profile key onto its canonical form.

    Two passes, because a language model invents keys faster than any fixed table can
    enumerate them:

    1. **Exact/normalised lookup** for the common cases (``name``→``姓名``).
    2. **Semantic fold by keyword** on the residual, after stripping noise words like
       ``favorite_`` / ``最喜欢`` / ``相关的``. Without this a single attribute still
       fragmented into ``coffee_preference`` / ``最喜欢的饮料`` / ``饮品偏好`` /
       ``食物偏好`` — four rows all describing what the user drinks.

    Matching is keyword-based rather than edit-distance based on purpose: it is
    predictable, has no threshold to tune, and can be reviewed by reading the table.
    """
    cleaned = (key or "").strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    exact = _PROFILE_KEY_ALIASES.get(lowered) or _PROFILE_KEY_ALIASES.get(cleaned)
    if exact:
        return exact[:48]

    # Strip decoration words that carry no attribute meaning.
    residue = lowered.replace("_", "").replace("-", "").replace(" ", "")
    for noise in (
        "favorite", "favourite", "preferred", "preference", "like", "likes", "loves",
        "main", "primary", "current", "the", "my", "user",
        "最喜欢的", "最爱的", "喜欢的", "爱好的", "偏好的", "相关的", "主要的", "常见的",
        "用户的", "个人的", "习惯", "偏好", "信息",
    ):
        residue = residue.replace(noise, "")
    if not residue:
        return cleaned[:48]

    # Compound attributes must be resolved before the coarse groups: "猫的名字" contains
    # 名字 and would otherwise be folded into the *user's* 姓名, merging a pet's name into
    # the person's. These patterns name the owner explicitly, so they win.
    for pattern, owner in (
        (("猫", "cat"), "猫"),
        (("狗", "dog"), "狗"),
        (("宠物", "pet"), "宠物"),
        (("老婆", "妻子", "wife"), "妻子"),
        (("老公", "丈夫", "husband"), "丈夫"),
        (("孩子", "儿子", "女儿", "child", "son", "daughter"), "孩子"),
        (("同事", "colleague"), "同事"),
    ):
        if any(token in residue for token in pattern):
            for attr, keywords in (
                ("姓名", ("名字", "name", "称呼")),
                ("品种", ("品种", "breed", "种类")),
                ("年龄", ("年龄", "age", "岁")),
                ("生日", ("生日", "birthday")),
            ):
                if any(keyword in residue for keyword in keywords):
                    return f"{owner}{attr}"[:48]
            return f"{owner}相关"[:48]

    for canonical, keywords in _PROFILE_KEYWORD_GROUPS:
        if any(keyword in residue for keyword in keywords):
            return canonical
    return cleaned[:48]


#: Coarse semantic groups applied to the residual key. Order matters: the first group
#: whose keyword appears wins, so narrower groups must come first.
_PROFILE_KEYWORD_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    # 饮品要在"食物"之前，否则"最喜欢喝的饮料"会被食物组截走。
    ("饮品偏好", ("咖啡", "茶", "饮料", "饮品", "drink", "coffee", "tea", "beverage")),
    ("饮食偏好", ("食物", "food", "吃", "口味", "饮食", "cuisine", "meal")),
    ("姓名", ("姓名", "名字", "称呼", "name")),
    ("昵称", ("昵称", "nickname", "外号")),
    ("宠物", ("宠物", "猫", "狗", "pet", "cat", "dog")),
    ("所在地", ("城市", "地点", "所在地", "居住", "location", "city", "address")),
    ("职业", ("职业", "工作", "职位", "occupation", "job", "profession")),
    ("公司", ("公司", "单位", "company", "employer")),
    ("年龄", ("年龄", "岁数", "age")),
    ("生日", ("生日", "birthday")),
    ("兴趣爱好", ("兴趣", "爱好", "hobby", "hobbies", "interest")),
    ("音乐偏好", ("音乐", "music", "歌", "song")),
    ("作息习惯", ("作息", "睡眠", "睡觉", "schedule", "sleep", "routine")),
    ("环境偏好", ("环境", "噪音", "安静", "environment", "noise")),
    ("当前项目", ("项目", "project", "product")),
    ("当前状态", ("状态", "status", "最近", "近况")),
    ("常用语言", ("语言", "language")),
    ("目标", ("目标", "计划", "goal", "plan")),
)


def is_self_reference(name: str) -> bool:
    """True when a subject/entity name refers to the user themself."""
    return (name or "").strip().lower() in {item.lower() for item in _SELF_REFERENCES}


def _unit(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, number))


__all__ = [
    "EXTRACTION_SCHEMA",
    "ExtractionOutcome",
    "MemoryExtractor",
    "heuristic_extract",
    "parse_json_loose",
]
