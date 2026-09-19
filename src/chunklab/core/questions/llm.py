from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from chunklab.core.embedders.base import MissingApiKeyError


@runtime_checkable
class LLM(Protocol):
    def complete(self, prompt: str) -> str: ...


@dataclass
class FakeLLM:
    reply: str = "What is described in this passage?"
    prompts: list[str] = field(default_factory=list)

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


@dataclass
class OpenAIChat:
    model: str = "gpt-4o-mini"

    def complete(self, prompt: str) -> str:
        if not os.environ.get("OPENAI_API_KEY"):
            raise MissingApiKeyError("OPENAI_API_KEY is not set")
        from openai import OpenAI  # lazy import

        resp = OpenAI().chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content or ""


@dataclass
class GeminiChat:
    model: str = "gemini-2.5-flash"

    def complete(self, prompt: str) -> str:
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise MissingApiKeyError("GEMINI_API_KEY is not set")
        from google import genai  # lazy import

        resp = genai.Client().models.generate_content(model=self.model, contents=prompt)
        return resp.text or ""


_PROVIDERS = {
    "fake": lambda model: FakeLLM(),
    "openai": lambda model: OpenAIChat(model) if model else OpenAIChat(),
    "gemini": lambda model: GeminiChat(model) if model else GeminiChat(),
}


def build_llm(spec: str) -> LLM:
    provider, _, model = spec.partition(":")
    try:
        factory = _PROVIDERS[provider]
    except KeyError:
        raise KeyError(
            f"unknown llm provider '{provider}' in '{spec}'; available: {sorted(_PROVIDERS)}"
        ) from None
    return factory(model)
