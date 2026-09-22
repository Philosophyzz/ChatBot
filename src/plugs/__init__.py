"""Optional extension point: drop-in plugins.

Any module placed in this package is imported automatically at startup
(``registry.load_plugins("plugs")``), so a new capability can be added without
touching the core. The shipped modules here only re-export the built-in
implementations so that ``import plugs`` guarantees every built-in plugin's
``@register`` decorator has run — the registry then reflects reality rather than
whatever happened to be imported.

Writing your own plugin
-----------------------
Create ``src/plugs/my_thing.py``::

    from core.registry import KIND_TTS, register

    @register(KIND_TTS, "my_engine")
    class MyEngine:
        name = "my_engine"
        supports_cloning = False
        offline = True

        async def synthesize(self, text, voice, *, stream=False):
            yield SpeechChunk(data=b"...", mime="audio/wav", is_final=True)

        async def health(self):
            return {"ok": True, "name": self.name}

Then select it in configuration::

    speech:
      tts_preference: [my_engine, edge]

Rules that keep this safe:

* An import error in a plugin is logged and skipped — never fatal.
* A plugin must satisfy the matching ``Protocol`` from :mod:`core.types`; the
  registry checks it when the instance is created.
* Wrap any heavy optional import (``torch``, ``av``, a vendor SDK) inside the
  method that needs it, so the app still starts on a machine without it.
"""

from __future__ import annotations

# Importing these modules is what registers the built-in implementations.
from llm import embed as _embed  # noqa: F401
from llm import mock as _mock  # noqa: F401
from llm import openai_compat as _openai_compat  # noqa: F401
from memory import store as _store  # noqa: F401
from memory import vector as _vector  # noqa: F401
from speech import stt as _stt  # noqa: F401
from speech import tts as _tts  # noqa: F401

__all__: list = []
