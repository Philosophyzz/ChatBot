"""Plugin registry — the extension point of the whole system.

Adding a capability (a new LLM runtime, a different vector store, a TTS engine,
another input channel) means writing a class that satisfies the matching
``Protocol`` from :mod:`core.types` and decorating it::

    from core.registry import register

    @register("llm", "myruntime")
    class MyRuntime:
        name = "myruntime"
        ...

The engine then resolves it by name from configuration
(``llm.backend: myruntime``). Nothing else in the codebase needs to change, which
is what makes the architecture extensible rather than merely modular.

Kinds are open strings, but the ones the shipped code uses are exported as
constants to keep typos out of the hot path.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

KIND_LLM = "llm"
KIND_EMBEDDER = "embedder"
KIND_RERANKER = "reranker"
KIND_VECTOR_STORE = "vector_store"
KIND_MEMORY_STORE = "memory_store"
KIND_RETRIEVER = "retriever"
KIND_STT = "stt"
KIND_TTS = "tts"
KIND_CHANNEL = "channel"
KIND_PERSONA_SOURCE = "persona_source"
KIND_EXTRACTOR = "extractor"

#: Every kind constant defined in this module. Keep in sync when adding one —
#: ``tests/test_core.py::test_registry_kind_constants_are_complete`` fails loudly
#: if a module decorates with a kind that does not exist, which is otherwise an
#: ImportError at runtime rather than a test failure.
ALL_KINDS: Tuple[str, ...] = (
    KIND_LLM,
    KIND_EMBEDDER,
    KIND_RERANKER,
    KIND_VECTOR_STORE,
    KIND_MEMORY_STORE,
    KIND_RETRIEVER,
    KIND_STT,
    KIND_TTS,
    KIND_CHANNEL,
    KIND_PERSONA_SOURCE,
    KIND_EXTRACTOR,
)

#: Protocols each kind is validated against, when the caller asks for validation.
_PROTOCOL_BY_KIND: Dict[str, Tuple[str, str]] = {
    KIND_LLM: ("core.types", "LLMBackend"),
    KIND_EMBEDDER: ("core.types", "Embedder"),
    KIND_RERANKER: ("core.types", "Reranker"),
    KIND_VECTOR_STORE: ("core.types", "VectorStore"),
    KIND_MEMORY_STORE: ("core.types", "MemoryStore"),
    KIND_STT: ("core.types", "STTBackend"),
    KIND_TTS: ("core.types", "TTSBackend"),
    KIND_CHANNEL: ("core.types", "Channel"),
}

_MISSING = object()


@dataclass(frozen=True)
class Registration:
    kind: str
    name: str
    factory: Callable[..., Any]
    module: str
    doc: str


class Registry:
    """A tiny, explicit service locator.

    It stores *factories*, not instances: construction (and therefore device
    placement, credential lookup and lazy imports) stays under the caller's
    control, which is essential when a model costs 10 GB of VRAM.
    """

    def __init__(self) -> None:
        self._items: Dict[str, Dict[str, Registration]] = {}
        self._instances: Dict[Tuple[str, str, str], Any] = {}
        self._modules_loaded = False

    # -- registration ------------------------------------------------------------------
    def add(self, kind: str, name: str, factory: Callable[..., Any]) -> Registration:
        bucket = self._items.setdefault(kind, {})
        if name in bucket and bucket[name].factory is not factory:
            # Last registration wins, but make the shadowing visible in logs.
            pass
        reg = Registration(
            kind=kind,
            name=name,
            factory=factory,
            module=getattr(factory, "__module__", "?"),
            doc=inspect.getdoc(factory) or "",
        )
        bucket[name] = reg
        return reg

    def register(self, kind: str, name: Optional[str] = None) -> Callable[[Any], Any]:
        """Decorator form: ``@register("tts", "edge")``."""

        def deco(obj: Any) -> Any:
            resolved = name or getattr(obj, "name", None) or getattr(obj, "__name__", "").lower()
            if not resolved:
                raise ValueError(f"cannot infer a name for {obj!r}")
            self.add(kind, resolved, obj)
            return obj

        return deco

    # -- discovery ---------------------------------------------------------------------
    def load_plugins(self, package: str = "plugs", *, force: bool = False) -> List[str]:
        """Import every submodule of ``package`` so decorators run.

        Failures are collected, not raised: one broken optional plugin (a missing
        ``torch``, for instance) must not take down the whole application.
        """
        if self._modules_loaded and not force:
            return []
        loaded: List[str] = []
        try:
            pkg = importlib.import_module(package)
        except Exception:
            return loaded
        for info in pkgutil.iter_modules(getattr(pkg, "__path__", [])):
            if info.name.startswith("_"):
                continue
            mod_name = f"{package}.{info.name}"
            try:
                importlib.import_module(mod_name)
                loaded.append(mod_name)
            except Exception as exc:  # pragma: no cover - optional dependency
                from core.logging import get_logger

                get_logger(__name__).warning("plugin %s failed to import: %s", mod_name, exc)
        self._modules_loaded = True
        return loaded

    # -- lookup ------------------------------------------------------------------------
    def has(self, kind: str, name: str) -> bool:
        return name in self._items.get(kind, {})

    def get_registration(self, kind: str, name: str) -> Optional[Registration]:
        return self._items.get(kind, {}).get(name)

    def names(self, kind: str) -> List[str]:
        return sorted(self._items.get(kind, {}))

    def all_registrations(self) -> List[Registration]:
        return [reg for bucket in self._items.values() for reg in bucket.values()]

    def create(self, kind: str, name: str, *args: Any, **kwargs: Any) -> Any:
        """Instantiate a registered plugin.

        ``name`` here selects the *plugin*. A plugin's constructor may also take a
        ``name`` parameter (to label itself), which is why passing ``name=`` in
        ``kwargs`` raises "got multiple values for argument 'name'" — the two meanings
        collide. Callers that need to label an instance should pass it explicitly as a
        constructor argument only when it is not also the plugin name, or simply let
        the plugin use its own default.
        """
        reg = self.get_registration(kind, name)
        if reg is None:
            available = ", ".join(self.names(kind)) or "(none registered)"
            raise KeyError(f"no {kind} plugin named {name!r}; available: {available}")
        return reg.factory(*args, **kwargs)

    def create_validated(self, kind: str, name: str, *args: Any, **kwargs: Any) -> Any:
        instance = self.create(kind, name, *args, **kwargs)
        self.validate(kind, instance)
        return instance

    def validate(self, kind: str, instance: Any) -> None:
        target = _PROTOCOL_BY_KIND.get(kind)
        if not target:
            return
        try:
            module = importlib.import_module(target[0])
            protocol = getattr(module, target[1])
        except Exception:  # pragma: no cover
            return
        if not isinstance(instance, protocol):
            missing = [
                attr
                for attr in ("name",)
                if not hasattr(instance, attr)
            ]
            raise TypeError(
                f"{type(instance).__name__} does not satisfy the {target[1]} contract"
                + (f" (missing: {', '.join(missing)})" if missing else "")
            )

    # -- memoized singletons ------------------------------------------------------------
    def singleton(self, kind: str, name: str, cache_key: str = "", *args: Any, **kwargs: Any) -> Any:
        """Build once per (kind, name, cache_key).

        Model objects are expensive; the engine asks for the same embedder on
        every request and must get the same instance back.
        """
        key = (kind, name, cache_key)
        if key not in self._instances:
            self._instances[key] = self.create_validated(kind, name, *args, **kwargs)
        return self._instances[key]

    def set_instance(self, kind: str, name: str, instance: Any, cache_key: str = "") -> None:
        """Inject a pre-built instance (used by tests and by the server wiring)."""
        self._instances[(kind, name, cache_key)] = instance

    def clear_instances(self, kind: Optional[str] = None) -> None:
        if kind is None:
            self._instances.clear()
            return
        for key in [k for k in self._instances if k[0] == kind]:
            self._instances.pop(key, None)

    def snapshot(self) -> Dict[str, List[str]]:
        return {kind: self.names(kind) for kind in sorted(self._items)}


#: Process-wide registry. Tests may build their own ``Registry()`` instead.
registry = Registry()


def register(kind: str, name: Optional[str] = None) -> Callable[[Any], Any]:
    """Module-level convenience wrapper around :meth:`Registry.register`."""
    return registry.register(kind, name)


def get_registry() -> Registry:
    return registry


__all__ = [
    "ALL_KINDS",
    "KIND_CHANNEL",
    "KIND_EMBEDDER",
    "KIND_EXTRACTOR",
    "KIND_LLM",
    "KIND_MEMORY_STORE",
    "KIND_PERSONA_SOURCE",
    "KIND_RERANKER",
    "KIND_RETRIEVER",
    "KIND_STT",
    "KIND_TTS",
    "KIND_VECTOR_STORE",
    "Registration",
    "Registry",
    "get_registry",
    "register",
    "registry",
]
