import time
from pathlib import Path

from chunklab.core.models import Question, Span
from chunklab.core.runner import ExperimentConfig
from chunklab.core.text import load_document
from chunklab.server.runs import RunManager
from chunklab.server.store import Store

SAMPLE = Path(__file__).resolve().parents[1] / "fixtures" / "sample.md"


def _wait(mgr, rid, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = mgr.status(rid)
        if st.state != "running":
            return st
        time.sleep(0.02)
    raise AssertionError("run did not finish")


def test_run_manager_completes_and_persists(tmp_path: Path):
    doc = load_document(SAMPLE)
    s = doc.text.index("Customers")
    e = doc.text.index("Digital")
    qs = [Question("q", "refund 30 days", (Span(doc.id, s, e),))]
    cfg = ExperimentConfig(
        documents=["docs/*"],
        questions="questions.json",
        chunkers=[{"name": "markdown"}, {"name": "recursive", "params": {"chunk_size": [128]}}],
        embedders=["fake"],
    )
    with Store(tmp_path / "c.db") as store:
        mgr = RunManager(store, tmp_path / "cache.db")
        rid = mgr.start(cfg, [doc], qs)
        st = _wait(mgr, rid)
        assert st.state == "done" and st.done == st.total == 2
        row = store.get_run(rid)
        assert row.status == "done" and row.result.run_id == rid and len(row.result.combos) == 2
        assert "name: markdown" in row.config_yaml


def test_run_manager_records_error(tmp_path: Path):
    doc = load_document(SAMPLE)
    qs = [Question("q", "x", (Span(doc.id, 0, 5),))]
    cfg = ExperimentConfig(
        documents=["docs/*"],
        questions="q.json",
        chunkers=[{"name": "markdown"}],
        embedders=["fake"],
    )

    def boom(spec):
        raise RuntimeError("factory down")

    with Store(tmp_path / "c.db") as store:
        mgr = RunManager(store, tmp_path / "cache.db", embedder_factory=boom)
        rid = mgr.start(cfg, [doc], qs)
        st = _wait(mgr, rid)
        assert st.state == "error" and "factory down" in st.error
        assert store.get_run(rid).status == "error"


def test_run_manager_validates_questions_up_front(tmp_path: Path):
    import pytest

    doc = load_document(SAMPLE)
    cfg = ExperimentConfig(
        documents=["docs/*"],
        questions="q.json",
        chunkers=[{"name": "markdown"}],
        embedders=["fake"],
    )
    with Store(tmp_path / "c.db") as store:
        mgr = RunManager(store, tmp_path / "cache.db")
        with pytest.raises(ValueError, match="unknown doc_id"):
            mgr.start(cfg, [doc], [Question("q", "x", (Span("ghost", 0, 5),))])
        assert mgr.status("nope") is None
