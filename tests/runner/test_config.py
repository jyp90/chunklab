from pathlib import Path

import pytest

from chunklab.core.runner.config import ChunkerGrid, Combo, ExperimentConfig, expand_matrix

YAML = """
documents: ["docs/*.md"]
questions: questions.json
chunkers:
  - name: recursive
    params: {chunk_size: [256, 512], overlap: [0, 50]}
  - name: sentence_window
    params: {window: [1]}
embedders: ["fake", "openai:text-embedding-3-small"]
retrieval:
  top_k: [5]
  hybrid: [false, true]
hit_threshold: 0.6
"""


def test_from_yaml_parses(tmp_path: Path):
    p = tmp_path / "exp.yaml"
    p.write_text(YAML)
    cfg = ExperimentConfig.from_yaml(p)
    assert cfg.documents == ["docs/*.md"]
    assert cfg.chunkers[0].params == {"chunk_size": [256, 512], "overlap": [0, 50]}
    assert cfg.hit_threshold == 0.6
    assert cfg.cache_path == "~/.chunklab/cache.db"


def test_expand_matrix_count_and_order(tmp_path: Path):
    p = tmp_path / "exp.yaml"
    p.write_text(YAML)
    combos = expand_matrix(ExperimentConfig.from_yaml(p))
    # chunker variants: recursive 2*2=4 + sentence_window 1 = 5; embedders 2; top_k 1; hybrid 2
    assert len(combos) == 5 * 2 * 1 * 2
    assert combos[0] == Combo("recursive", (("chunk_size", 256), ("overlap", 0)), "fake", 5, False)
    assert len({c.id for c in combos}) == len(combos)


def test_combo_id_and_params_dict():
    c = Combo("recursive", (("chunk_size", 256), ("overlap", 0)), "fake", 5, True)
    assert c.id == "recursive(chunk_size=256,overlap=0)|fake|k=5|hybrid=True"
    assert c.params_dict() == {"chunk_size": 256, "overlap": 0}


def test_chunker_without_params_yields_one_variant():
    cfg = ExperimentConfig(
        documents=["x.md"],
        questions="q.json",
        chunkers=[{"name": "markdown"}],
        embedders=["fake"],
    )
    combos = expand_matrix(cfg)
    assert len(combos) == 1
    assert combos[0].id == "markdown()|fake|k=5|hybrid=False"


def test_resolve_documents_globs_relative_to_base(tmp_path: Path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "b.md").write_text("b")
    (tmp_path / "docs" / "a.md").write_text("a")
    cfg = ExperimentConfig(
        documents=["docs/*.md", "docs/a.md"], questions="q", chunkers=[], embedders=[]
    )
    paths = cfg.resolve_documents(tmp_path)
    assert [p.name for p in paths] == ["a.md", "b.md"]


def test_invalid_yaml_missing_required(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("documents: []\n")
    with pytest.raises(ValueError):
        ExperimentConfig.from_yaml(p)


def test_resolve_documents_accepts_absolute_patterns(tmp_path: Path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("a")
    (tmp_path / "docs" / "b.md").write_text("b")
    cfg = ExperimentConfig(
        documents=[str(tmp_path / "docs" / "*.md")], questions="q", chunkers=[], embedders=[]
    )
    paths = cfg.resolve_documents(tmp_path / "elsewhere")
    assert [p.name for p in paths] == ["a.md", "b.md"]


def test_resolve_documents_supports_recursive_globs(tmp_path: Path):
    nested = tmp_path / "docs" / "deep"
    nested.mkdir(parents=True)
    (nested / "c.md").write_text("c")
    (tmp_path / "docs" / "a.md").write_text("a")
    cfg = ExperimentConfig(documents=["docs/**/*.md"], questions="q", chunkers=[], embedders=[])
    assert [p.name for p in cfg.resolve_documents(tmp_path)] == ["a.md", "c.md"]


def test_chunker_grid_rejects_non_scalar_param_values():
    with pytest.raises(ValueError, match="chunk_size"):
        ChunkerGrid(name="recursive", params={"chunk_size": [[10, 20]]})


def test_chunker_grid_accepts_scalar_param_values():
    grid = ChunkerGrid(name="recursive", params={"chunk_size": [10], "sep": ["x"], "on": [True]})
    assert grid.params["chunk_size"] == [10]
