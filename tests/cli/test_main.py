import json
from pathlib import Path

from typer.testing import CliRunner

from chunklab.cli.main import app
from chunklab.core.models import Question, Span
from chunklab.core.questions.io import load_questions, save_questions
from chunklab.core.text import load_document

# Click 8.2+ dropped CliRunner(mix_stderr=...); stdout/stderr are combined into
# `result.output` by default. Older Click needs mix_stderr=True for the same effect.
try:
    runner = CliRunner(mix_stderr=True)
except TypeError:
    runner = CliRunner()


def _workspace(tmp_path: Path, sample_doc_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sample.md").write_text(sample_doc_path.read_text())
    doc = load_document(docs / "sample.md")
    s = doc.text.index("Customers may request a refund")
    e = doc.text.index("Digital goods")
    save_questions(
        [Question("q1", "how many days for a refund 30 days", (Span("sample", s, e),))],
        tmp_path / "questions.json",
    )
    (tmp_path / "exp.yaml").write_text(
        f"""
documents: ["docs/*.md"]
questions: questions.json
chunkers:
  - name: markdown
embedders: ["fake"]
retrieval: {{top_k: [3], hybrid: [false]}}
cache_path: "{tmp_path / "cache.db"}"
"""
    )
    return tmp_path


def test_version():
    r = runner.invoke(app, ["version"])
    assert r.exit_code == 0
    assert "0.1.0" in r.stdout


def test_run_writes_result_and_prints_table(tmp_path: Path, sample_doc_path: Path):
    ws = _workspace(tmp_path, sample_doc_path)
    r = runner.invoke(app, ["run", str(ws / "exp.yaml")])
    assert r.exit_code == 0, r.output
    assert "hit@3" in r.stdout
    data = json.loads((ws / "result.json").read_text())
    assert len(data["combos"]) == 1


def test_run_fail_below_exits_1(tmp_path: Path, sample_doc_path: Path):
    ws = _workspace(tmp_path, sample_doc_path)
    r = runner.invoke(app, ["run", str(ws / "exp.yaml"), "--fail-below", "hit@3=1.5"])
    assert r.exit_code == 1
    assert "hit@3" in r.output


def test_run_baseline_regression_exits_1(tmp_path: Path, sample_doc_path: Path):
    ws = _workspace(tmp_path, sample_doc_path)
    r1 = runner.invoke(app, ["run", str(ws / "exp.yaml"), "--out", str(ws / "base.json")])
    assert r1.exit_code == 0, r1.output
    base = json.loads((ws / "base.json").read_text())
    for c in base["combos"]:
        c["metrics"] = {k: 1.0 for k in c["metrics"]}
    (ws / "base.json").write_text(json.dumps(base))
    r2 = runner.invoke(
        app,
        ["run", str(ws / "exp.yaml"), "--baseline", str(ws / "base.json"), "--max-drop", "0.0"],
    )
    # at least precision/iou cannot be exactly 1.0 for a whole markdown section vs a sub-span
    assert r2.exit_code == 1
    assert "dropped" in r2.output


def test_generate_questions_with_fake_llm(tmp_path: Path, sample_doc_path: Path):
    out = tmp_path / "q.json"
    r = runner.invoke(
        app,
        [
            "generate-questions",
            str(sample_doc_path),
            "--out",
            str(out),
            "--llm",
            "fake",
            "--per-doc",
            "2",
            "--min-len",
            "50",
        ],
    )
    assert r.exit_code == 0, r.output
    qs = load_questions(out)
    assert len(qs) == 2
    assert all(q.spans[0].doc_id == "sample" for q in qs)
