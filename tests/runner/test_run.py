import json
from pathlib import Path

import pytest

from chunklab.core.embedders.base import FakeEmbedder
from chunklab.core.models import Question, Span
from chunklab.core.questions.io import save_questions
from chunklab.core.runner.config import ExperimentConfig
from chunklab.core.runner.run import RunResult, run_experiment, validate_questions
from chunklab.core.text import load_document


@pytest.fixture
def workspace(tmp_path: Path, sample_doc_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sample.md").write_text(sample_doc_path.read_text())
    doc = load_document(docs / "sample.md")
    refund = doc.text.index("Customers may request a refund")
    refund_end = doc.text.index("Digital goods")
    lost = doc.text.index("If a package is marked delivered")
    lost_end = doc.text.index("lost package report.") + len("lost package report.")
    save_questions(
        [
            Question(
                "q-refund", "how many days for a refund?", (Span("sample", refund, refund_end),)
            ),
            Question(
                "q-lost",
                "package marked delivered but not received",
                (Span("sample", lost, lost_end),),
            ),
        ],
        tmp_path / "questions.json",
    )
    (tmp_path / "exp.yaml").write_text(
        f"""
documents: ["docs/*.md"]
questions: questions.json
chunkers:
  - name: recursive
    params: {{chunk_size: [150, 400], overlap: [0]}}
  - name: markdown
embedders: ["fake"]
retrieval:
  top_k: [2]
  hybrid: [false, true]
cache_path: "{tmp_path / "cache.db"}"
"""
    )
    return tmp_path


def test_run_experiment_produces_all_combos(workspace: Path):
    cfg = ExperimentConfig.from_yaml(workspace / "exp.yaml")
    result = run_experiment(cfg, workspace)
    assert len(result.combos) == 3 * 1 * 1 * 2
    assert all(c.error is None for c in result.combos)
    for c in result.combos:
        assert set(c.metrics) == {"hit@2", "mrr", "ndcg", "precision", "iou"}
        assert c.n_chunks > 0
        assert len(c.per_question) == 2
        assert len(c.per_question[0]["retrieved"]) == 2


def test_run_experiment_finds_refund_span_with_fake_embedder(workspace: Path):
    cfg = ExperimentConfig.from_yaml(workspace / "exp.yaml")
    result = run_experiment(cfg, workspace)
    md_dense = next(c for c in result.combos if c.chunker == "markdown" and not c.hybrid)
    refund = next(q for q in md_dense.per_question if q["id"] == "q-refund")
    assert refund["hit"] is True


def test_run_result_json_roundtrip(workspace: Path):
    cfg = ExperimentConfig.from_yaml(workspace / "exp.yaml")
    result = run_experiment(cfg, workspace)
    out = workspace / "result.json"
    result.to_json(out)
    data = json.loads(out.read_text())
    assert data["run_id"] == result.run_id
    assert data["combos"][0]["combo_id"] == result.combos[0].combo_id
    loaded = RunResult.from_json(out)
    assert loaded.combos[0].metrics == result.combos[0].metrics


def test_partial_failure_is_isolated(workspace: Path):
    cfg = ExperimentConfig.from_yaml(workspace / "exp.yaml")
    cfg = cfg.model_copy(update={"embedders": ["fake", "boom"]})

    class Boom:
        name = "boom"

        def embed(self, texts):
            raise RuntimeError("provider down")

    def factory(spec: str):
        return Boom() if spec == "boom" else FakeEmbedder()

    result = run_experiment(cfg, workspace, embedder_factory=factory)
    failed = [c for c in result.combos if c.error]
    ok = [c for c in result.combos if not c.error]
    assert len(failed) == 6 and len(ok) == 6
    assert "RuntimeError: provider down" in failed[0].error
    assert failed[0].metrics == {}


def test_progress_callback_called_per_combo(workspace: Path):
    cfg = ExperimentConfig.from_yaml(workspace / "exp.yaml")
    seen: list[str] = []
    run_experiment(cfg, workspace, progress=seen.append)
    assert len(seen) == 6


def test_validate_questions_rejects_unknown_doc_and_out_of_range(sample_doc_path: Path):
    doc = load_document(sample_doc_path)
    with pytest.raises(ValueError, match="unknown doc_id"):
        validate_questions([Question("q", "t", (Span("nope", 0, 5),))], [doc])
    with pytest.raises(ValueError, match="out of range"):
        validate_questions([Question("q", "t", (Span("sample", 0, 10_000),))], [doc])


def test_validate_questions_rejects_question_without_spans(sample_doc_path: Path):
    doc = load_document(sample_doc_path)
    with pytest.raises(ValueError, match="question 'q-empty' has no spans"):
        validate_questions([Question("q-empty", "t", ())], [doc])


def test_validate_questions_rejects_duplicate_ids(sample_doc_path: Path):
    doc = load_document(sample_doc_path)
    q = Question("dup", "t", (Span("sample", 0, 5),))
    with pytest.raises(ValueError, match="duplicate question id 'dup'"):
        validate_questions([q, q], [doc])


def test_relative_cache_path_resolves_against_config_dir(workspace: Path, tmp_path, monkeypatch):
    exp = workspace / "exp.yaml"
    exp.write_text(
        "\n".join(
            line for line in exp.read_text().splitlines() if not line.startswith("cache_path:")
        )
        + '\ncache_path: "c.db"\n'
    )
    elsewhere = tmp_path.parent / "cwd-elsewhere"
    elsewhere.mkdir(exist_ok=True)
    monkeypatch.chdir(elsewhere)
    cfg = ExperimentConfig.from_yaml(exp)
    run_experiment(cfg, workspace)
    assert (workspace / "c.db").exists()
    assert not (elsewhere / "c.db").exists()


def test_to_json_creates_parent_dirs(workspace: Path):
    cfg = ExperimentConfig.from_yaml(workspace / "exp.yaml")
    result = run_experiment(cfg, workspace)
    out = workspace / "new" / "dir" / "r.json"
    result.to_json(out)
    assert json.loads(out.read_text())["run_id"] == result.run_id


def test_run_experiment_warns_and_skips_bad_document(workspace: Path):
    (workspace / "docs" / "bad.pdf").write_bytes(b"not a pdf")
    cfg = ExperimentConfig.from_yaml(workspace / "exp.yaml")
    cfg = cfg.model_copy(update={"documents": ["docs/*"]})
    warnings: list[str] = []
    result = run_experiment(cfg, workspace, warn=warnings.append)
    assert len(warnings) == 1
    assert "bad.pdf" in warnings[0]
    assert result.combos and all(c.error is None for c in result.combos)
