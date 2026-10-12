"""Language model backends.

offline     deterministic template writer. No network, no key, fully reproducible. This is
            the default and what the evaluation numbers in the README are produced with.
anthropic   Claude through the Anthropic Messages API. Used for narrative synthesis and
            free text questions. Enabled with WEIR_LLM=anthropic (or llm.backend in the
            config) plus ANTHROPIC_API_KEY. Every sentence it writes is still passed through
            the verifier, which drops any claim that cites missing evidence or states a
            number that is not in the computed fact registry.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

LOG = logging.getLogger("weir_whisper.llm")


@dataclass
class LLMResult:
    text: str
    backend: str
    model: str | None = None
    usage: dict | None = None


class OfflineBackend:
    name = "offline"
    model = None

    def available(self) -> bool:
        return True

    def complete(self, system: str, prompt: str) -> LLMResult | None:
        return None  # the writer falls back to its templates


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model: str, max_tokens: int = 3000, temperature: float = 0.0):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def available(self) -> bool:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        import importlib.util
        return importlib.util.find_spec("anthropic") is not None

    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system: str, prompt: str) -> LLMResult | None:
        try:
            msg = self.client().messages.create(
                model=self.model, max_tokens=self.max_tokens, temperature=self.temperature, system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("anthropic call failed, falling back to offline writer: %s", exc)
            return None
        text = "".join(getattr(b, "text", "") for b in msg.content)
        usage = {"input_tokens": getattr(msg.usage, "input_tokens", None), "output_tokens": getattr(msg.usage, "output_tokens", None)}
        return LLMResult(text=text, backend=self.name, model=self.model, usage=usage)


def make_backend(settings) -> OfflineBackend | AnthropicBackend:
    cfg = settings["llm"]
    choice = os.environ.get("WEIR_LLM", cfg.get("backend", "offline"))
    if choice == "anthropic":
        be = AnthropicBackend(cfg["model"], int(cfg["max_tokens"]), float(cfg["temperature"]))
        if be.available():
            return be
        LOG.warning("anthropic backend requested but ANTHROPIC_API_KEY or the anthropic package is missing; using offline writer")
    return OfflineBackend()


def parse_json_block(text: str) -> dict | None:
    """Extract the first JSON object from a model response."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
