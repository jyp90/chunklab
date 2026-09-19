from __future__ import annotations

from chunklab.core.embedders.base import Embedder, FakeEmbedder, MissingApiKeyError
from chunklab.core.embedders.cache import CachedEmbedder, EmbeddingCache
from chunklab.core.embedders.gemini import GeminiEmbedder
from chunklab.core.embedders.local import LocalEmbedder
from chunklab.core.embedders.openai import OpenAIEmbedder

_PROVIDERS = {
    "fake": lambda model: FakeEmbedder(),
    "openai": lambda model: OpenAIEmbedder(model) if model else OpenAIEmbedder(),
    "gemini": lambda model: GeminiEmbedder(model) if model else GeminiEmbedder(),
    "local": lambda model: LocalEmbedder(model) if model else LocalEmbedder(),
}


def build_embedder(spec: str) -> Embedder:
    provider, _, model = spec.partition(":")
    try:
        factory = _PROVIDERS[provider]
    except KeyError:
        raise KeyError(
            f"unknown embedder provider '{provider}' in '{spec}'; available: {sorted(_PROVIDERS)}"
        ) from None
    return factory(model)


__all__ = [
    "CachedEmbedder",
    "Embedder",
    "EmbeddingCache",
    "FakeEmbedder",
    "GeminiEmbedder",
    "LocalEmbedder",
    "MissingApiKeyError",
    "OpenAIEmbedder",
    "build_embedder",
]
