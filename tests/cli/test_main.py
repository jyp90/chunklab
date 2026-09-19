import json
from pathlib import Path

from typer.testing import CliRunner

from chunklab.cli.main import app
from chunklab.core.models import Question, Span
from chunklab.core.questions.io import load_questions, save_questions
from chunklab.core.text import load_document

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
    assert "embeddings: " in r.stdout and "served from cache" in r.stdout
    data = json.loads((ws / "result.json").read_text())
    assert len(data["combos"]) == 1


def test_run_fail_below_exits_1(tmp_path: Path, sample_doc_path: Path):
    ws = _workspace(tmp_path, sample_doc_path)
    r = runner.invoke(app, ["run", str(ws / "exp.yaml"), "--fail-below", "hit@3=1.5"])
    assert r.exit_code == 1
    assert "FAILED" in r.output and "hit@3 " in r.output and "< 1.500" in r.output


def test_run_fail_below_unknown_metric_exits_2(tmp_path: Path, sample_doc_path: Path):
    ws = _workspace(tmp_path, sample_doc_path)
    r = runner.invoke(app, ["run", str(ws / "exp.yaml"), "--fail-below", "hit@9=0.5"])
    assert r.exit_code == 2
    assert "hit@9" in r.output


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


def test_run_notes_combos_missing_from_baseline(tmp_path: Path, sample_doc_path: Path):
    ws = _workspace(tmp_path, sample_doc_path)
    r1 = runner.invoke(app, ["run", str(ws / "exp.yaml"), "--out", str(ws / "base.json")])
    assert r1.exit_code == 0, r1.output
    exp = ws / "exp.yaml"
    exp.write_text(
        exp.read_text().replace("  - name: markdown", "  - name: markdown\n  - name: recursive")
    )
    r2 = runner.invoke(app, ["run", str(ws / "exp.yaml"), "--baseline", str(ws / "base.json")])
    assert r2.exit_code == 0, r2.output
    assert "note: 1 combo(s) not in baseline, skipped: recursive(" in r2.output


def test_run_missing_questions_file_exits_2(tmp_path: Path, sample_doc_path: Path):
    ws = _workspace(tmp_path, sample_doc_path)
    (ws / "questions.json").unlink()
    r = runner.invoke(app, ["run", str(ws / "exp.yaml")])
    assert r.exit_code == 2, r.output
    assert "error: questions file not found:" in r.output
    assert "questions.json" in r.output
    assert "Traceback" not in r.output


def test_help_lists_command_descriptions():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0, r.output
    assert "Run the chunking" in r.output
    assert "Sample passages from documents" in r.output
    assert "Print the chunklab version." in r.output


def test_run_bad_fail_below_expression_exits_2(tmp_path: Path, sample_doc_path: Path):
    ws = _workspace(tmp_path, sample_doc_path)
    r = runner.invoke(app, ["run", str(ws / "exp.yaml"), "--fail-below", "hit@3"])
    assert r.exit_code == 2, r.output
    assert "METRIC=VALUE" in r.output
    assert "Traceback" not in r.output


def test_run_unknown_embedder_exits_2(tmp_path: Path, sample_doc_path: Path):
    ws = _workspace(tmp_path, sample_doc_path)
    exp = ws / "exp.yaml"
    exp.write_text(exp.read_text().replace('embedders: ["fake"]', 'embedders: ["cohere"]'))
    r = runner.invoke(app, ["run", str(exp)])
    assert r.exit_code == 2, r.output
    assert "unknown embedder provider 'cohere'" in r.output
    assert "Traceback" not in r.output
    assert '"unknown' not in r.output  # KeyError quoting stripped


def test_run_out_overwriting_questions_file_exits_2(tmp_path: Path, sample_doc_path: Path):
    ws = _workspace(tmp_path, sample_doc_path)
    r = runner.invoke(app, ["run", str(ws / "exp.yaml"), "--out", str(ws / "questions.json")])
    assert r.exit_code == 2, r.output
    assert "error: --out must not overwrite the config or questions file" in r.output
    assert "Traceback" not in r.output
    # the questions file must survive untouched
    assert json.loads((ws / "questions.json").read_text())


def test_run_out_overwriting_config_exits_2(tmp_path: Path, sample_doc_path: Path):
    ws = _workspace(tmp_path, sample_doc_path)
    r = runner.invoke(app, ["run", str(ws / "exp.yaml"), "--out", str(ws / "exp.yaml")])
    assert r.exit_code == 2, r.output
    assert "error: --out must not overwrite the config or questions file" in r.output
    assert "Traceback" not in r.output


def test_run_invalid_yaml_exits_2(tmp_path: Path):
    exp = tmp_path / "exp.yaml"
    exp.write_text("documents: [docs/*.md\nchunkers: :\n")
    r = runner.invoke(app, ["run", str(exp)])
    assert r.exit_code == 2, r.output
    assert "error:" in r.output
    assert "Traceback" not in r.output


def test_run_non_scalar_param_reports_single_clean_line(tmp_path: Path, sample_doc_path: Path):
    ws = _workspace(tmp_path, sample_doc_path)
    exp = ws / "exp.yaml"
    exp.write_text(
        exp.read_text().replace(
            "  - name: markdown",
            "  - name: recursive\n    params: {chunk_size: [[10, 20]]}",
        )
    )
    r = runner.invoke(app, ["run", str(exp)])
    assert r.exit_code == 2, r.output
    assert "Traceback" not in r.output
    error_lines = [line for line in r.output.splitlines() if line.startswith("error:")]
    assert error_lines == [
        "error: chunkers.0.params: chunker 'recursive' param 'chunk_size' "
        "has non-scalar value [10, 20]"
    ]
    assert "errors.pydantic.dev" not in r.output


def test_generate_questions_unknown_llm_exits_2(tmp_path: Path, sample_doc_path: Path):
    r = runner.invoke(
        app,
        [
            "generate-questions",
            str(sample_doc_path),
            "--out",
            str(tmp_path / "q.json"),
            "--llm",
            "anthropic",
        ],
    )
    assert r.exit_code == 2, r.output
    assert "unknown llm provider 'anthropic'" in r.output
    assert "Traceback" not in r.output


def test_generate_questions_missing_api_key_exits_2(
    tmp_path: Path, sample_doc_path: Path, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = runner.invoke(
        app,
        [
            "generate-questions",
            str(sample_doc_path),
            "--out",
            str(tmp_path / "q.json"),
            "--min-len",
            "50",
        ],
    )
    assert r.exit_code == 2, r.output
    assert "error: OPENAI_API_KEY is not set" in r.output
    assert "Traceback" not in r.output


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


def test_generate_questions_exits_2_when_nothing_generated(tmp_path: Path):
    src = tmp_path / "notes.txt"
    src.write_text("short one\n\nshort two\n")
    out = tmp_path / "q.json"
    r = runner.invoke(
        app, ["generate-questions", str(src), "--out", str(out), "--llm", "fake", "--per-doc", "2"]
    )
    assert r.exit_code == 2, r.output
    assert "warning: notes: only 0 paragraph(s) >= 200 chars (requested 2)" in r.output
    assert "error: no questions generated" in r.output
    assert not out.exists()
