import os

import numpy as np
import pytest

from chunklab.core.embedders import build_embedder

live = pytest.mark.skipif(
    os.environ.get("CHUNKLAB_LIVE_TESTS") != "1", reason="set CHUNKLAB_LIVE_TESTS=1"
)


@live
@pytest.mark.parametrize("spec", ["openai", "gemini"])
def test_live_embedding_shape(spec: str):
    v = build_embedder(spec).embed(["refund policy", "shipping times"])
    assert v.shape[0] == 2
    assert v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-3)
