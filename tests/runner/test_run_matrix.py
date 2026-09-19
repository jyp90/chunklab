from pathlib import Path

from chunklab.core.models import Question, Span
from chunklab.core.runner import ExperimentConfig
from chunklab.core.runner.run import run_matrix
from chunklab.core.text import load_document

SAMPLE = Path(__file__).resolve().parents[1] / "fixtures" / "sample.md"


def test_run_matrix_with_in_memory_docs(tmp_path: Path):
    doc = load_document(SAMPLE)
    s = doc.text.index("Customers may request")
    e = doc.text.index("Digital goods")
    qs = [Question("q", "refund within 30 days", (Span(doc.id, s, e),))]
    cfg = ExperimentConfig(
        documents=["unused"],
        questions="unused",
        chunkers=[{"name": "markdown"}],
        embedders=["fake"],
    )
    seen = []
    res = run_matrix(cfg, [doc], qs, tmp_path / "c.db", progress=seen.append)
    assert len(res.combos) == 1 and res.combos[0].error is None
    assert seen == [res.combos[0].combo_id]
    assert (tmp_path / "c.db").exists()
