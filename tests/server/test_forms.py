import pytest

from chunklab.server.forms import config_from_form, config_to_yaml, parse_grid


def test_parse_grid():
    assert parse_grid("256, 512 ,1024") == [256, 512, 1024]
    assert parse_grid("") == []
    with pytest.raises(ValueError, match="integer"):
        parse_grid("256, big")


def test_config_from_form_full():
    cfg = config_from_form(
        {
            "chunker_recursive": "on",
            "recursive_chunk_size": "256,512",
            "recursive_overlap": "0,32",
            "chunker_markdown": "on",
            "markdown_chunk_size": "800",
            "embedders": ["fake", "openai:text-embedding-3-small"],
            "embedder_custom": "local:all-MiniLM-L6-v2",
            "top_k": "3,5",
            "hybrid": "both",
            "hit_threshold": "0.6",
        }
    )
    assert [c.name for c in cfg.chunkers] == ["recursive", "markdown"]
    assert cfg.chunkers[0].params == {"chunk_size": [256, 512], "overlap": [0, 32]}
    assert cfg.embedders == ["fake", "openai:text-embedding-3-small", "local:all-MiniLM-L6-v2"]
    assert cfg.retrieval.top_k == [3, 5] and cfg.retrieval.hybrid == [False, True]
    assert (
        cfg.hit_threshold == 0.6
        and cfg.documents == ["docs/*"]
        and cfg.cache_path == ".chunklab-cache.db"
    )


def test_config_from_form_defaults_and_errors():
    cfg = config_from_form(
        {
            "chunker_sentence_window": "on",
            "sentence_window_window": "1",
            "embedders": "fake",
            "top_k": "5",
            "hybrid": "dense",
        }
    )
    assert cfg.retrieval.hybrid == [False] and cfg.hit_threshold == 0.5
    with pytest.raises(ValueError, match="chunker"):
        config_from_form({"embedders": "fake", "top_k": "5", "hybrid": "dense"})
    with pytest.raises(ValueError, match="embedder"):
        config_from_form({"chunker_markdown": "on", "top_k": "5", "hybrid": "dense"})


def test_config_to_yaml_roundtrip():
    cfg = config_from_form(
        {"chunker_markdown": "on", "embedders": "fake", "top_k": "5", "hybrid": "hybrid"}
    )
    text = config_to_yaml(cfg)
    assert (
        "name: markdown" in text
        and "hybrid:\n- true" in text
        or "hybrid: [true]" in text
        or "- true" in text
    )
