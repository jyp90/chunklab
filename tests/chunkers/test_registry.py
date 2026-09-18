import pytest

from chunklab.core.chunkers import CHUNKERS, build_chunker
from chunklab.core.chunkers.base import Chunker


def test_registry_has_three_v1_chunkers():
    assert set(CHUNKERS) == {"recursive", "sentence_window", "markdown"}


def test_build_chunker_passes_params():
    c = build_chunker("recursive", chunk_size=64, overlap=8)
    assert isinstance(c, Chunker)
    assert c.name == "recursive"
    assert c.chunk_size == 64  # type: ignore[attr-defined]


def test_build_unknown_chunker():
    with pytest.raises(KeyError, match="unknown chunker"):
        build_chunker("semantic")
