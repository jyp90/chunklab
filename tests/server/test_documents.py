from pathlib import Path

from fastapi.testclient import TestClient

from chunklab.server.routes_documents import safe_stem

SAMPLE = Path(__file__).resolve().parents[1] / "fixtures" / "sample.md"


def test_safe_stem_strips_paths_and_odd_chars():
    assert safe_stem("../../etc/passwd.md") == "passwd"
    assert safe_stem("my report (final).pdf") == "my_report__final_"
    assert safe_stem(".md") == "document"


def test_upload_lists_and_views_document(client: TestClient, workspace: Path):
    with SAMPLE.open("rb") as f:
        r = client.post("/documents", files=[("files", ("sample.md", f, "text/markdown"))])
    assert r.status_code == 200
    assert "sample" in r.text  # doc_list partial
    assert (workspace / "docs" / "sample.md").exists()

    r = client.get("/documents/sample/view")
    assert r.status_code == 200
    assert '<pre id="doc" data-doc-id="sample">' in r.text
    assert "# Refund Policy" in r.text
    assert "&lt;" not in r.text  # sample has no html; sanity that escaping does not double up

    r = client.get("/")
    assert "sample" in r.text


def test_upload_escapes_html_in_text(client: TestClient, workspace: Path):
    r = client.post(
        "/documents",
        files=[("files", ("evil.md", b"<script>alert(1)</script> text\n", "text/markdown"))],
    )
    assert r.status_code == 200
    r = client.get("/documents/evil/view")
    assert "&lt;script&gt;" in r.text and "<script>alert" not in r.text


def test_upload_reports_skipped_files(client: TestClient):
    r = client.post(
        "/documents",
        files=[
            ("files", ("empty.md", b"", "text/markdown")),
            ("files", ("bad.docx", b"zzz", "application/octet-stream")),
            ("files", ("ok.txt", b"Some text.\n", "text/plain")),
        ],
    )
    assert r.status_code == 200
    assert "empty.md" in r.text and "bad.docx" in r.text
    assert "ok" in r.text
    assert client.get("/documents/ok/view").status_code == 200
    assert client.get("/documents/empty/view").status_code == 404


def test_delete_document(client: TestClient):
    client.post("/documents", files=[("files", ("a.md", b"alpha beta\n", "text/markdown"))])
    r = client.delete("/documents/a")
    assert r.status_code == 200
    assert "alpha" not in r.text
    assert client.get("/documents/a/view").status_code == 404


def test_delete_with_glob_metachar_deletes_nothing(client: TestClient, workspace: Path):
    client.post("/documents", files=[("files", ("a.md", b"alpha\n", "text/markdown"))])
    client.post("/documents", files=[("files", ("b.md", b"bravo\n", "text/markdown"))])
    r = client.delete("/documents/%2A")
    assert r.status_code == 404
    assert (workspace / "docs" / "a.md").exists()
    assert (workspace / "docs" / "b.md").exists()
    listing = client.get("/")
    assert "a" in listing.text and "b" in listing.text


def test_delete_removes_only_its_own_file(client: TestClient, workspace: Path):
    client.post("/documents", files=[("files", ("a.md", b"alpha\n", "text/markdown"))])
    client.post("/documents", files=[("files", ("b.md", b"bravo\n", "text/markdown"))])
    r = client.delete("/documents/a")
    assert r.status_code == 200
    assert not (workspace / "docs" / "a.md").exists()
    assert (workspace / "docs" / "b.md").exists()


def test_view_rejects_bad_id(client: TestClient):
    r = client.get("/documents/..%2Fx/view")
    assert r.status_code == 404
