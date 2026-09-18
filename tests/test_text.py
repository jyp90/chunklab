from pathlib import Path

import pytest

from chunklab.core.text import (
    MAX_DOCUMENT_BYTES,
    DocumentTooLargeError,
    EmptyDocumentError,
    UnsupportedFormatError,
    load_document,
    load_documents,
    normalize,
)


def test_normalize_line_endings_and_blank_lines():
    raw = "a\r\nb\r\n\r\n\r\n\r\nc  \n"
    assert normalize(raw) == "a\nb\n\nc\n"


def test_normalize_strips_trailing_whitespace_per_line():
    assert normalize("x   \ny\t\n") == "x\ny\n"


def test_normalize_ensures_single_trailing_newline():
    assert normalize("abc") == "abc\n"
    assert normalize("abc\n\n\n") == "abc\n"


def test_load_markdown(sample_doc_path: Path):
    doc = load_document(sample_doc_path)
    assert doc.id == "sample"
    assert doc.text.startswith("# Refund Policy\n")
    assert doc.source == str(sample_doc_path.resolve())


def test_load_txt(tmp_path: Path):
    p = tmp_path / "notes.txt"
    p.write_text("hello\r\nworld", encoding="utf-8")
    doc = load_document(p)
    assert doc.id == "notes"
    assert doc.text == "hello\nworld\n"


def test_load_pdf(tmp_path: Path, capsys):
    import pymupdf

    p = tmp_path / "doc.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Refunds within 30 days.")
    pdf.save(p)
    pdf.close()
    capsys.readouterr()

    doc = load_document(p)
    assert "Refunds within 30 days." in doc.text
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_unsupported_format(tmp_path: Path):
    p = tmp_path / "x.docx"
    p.write_bytes(b"")
    with pytest.raises(UnsupportedFormatError):
        load_document(p)


def test_rejects_oversized_document(tmp_path: Path, monkeypatch):
    import chunklab.core.text as text_mod

    monkeypatch.setattr(text_mod, "MAX_DOCUMENT_BYTES", 10)
    p = tmp_path / "big.txt"
    p.write_text("x" * 11)
    with pytest.raises(DocumentTooLargeError, match="10 bytes"):
        load_document(p)
    assert MAX_DOCUMENT_BYTES == 10 * 1024 * 1024


def test_load_document_rejects_text_only_whitespace(tmp_path: Path):
    p = tmp_path / "empty.md"
    p.write_text("   \n\n\t\n", encoding="utf-8")
    with pytest.raises(EmptyDocumentError, match="empty.md"):
        load_document(p)


def test_load_documents_skips_empty_document_when_on_error_given(tmp_path: Path):
    empty = tmp_path / "empty.md"
    empty.write_text("\n")
    good = tmp_path / "good.md"
    good.write_text("hello")
    seen: list[tuple[Path, Exception]] = []
    docs = load_documents([empty, good], on_error=lambda p, e: seen.append((p, e)))
    assert [d.id for d in docs] == ["good"]
    assert len(seen) == 1
    assert isinstance(seen[0][1], EmptyDocumentError)


def test_load_documents_rejects_duplicate_ids(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "same.md").write_text("one")
    (tmp_path / "b" / "same.md").write_text("two")
    with pytest.raises(ValueError, match="duplicate document id"):
        load_documents([tmp_path / "a" / "same.md", tmp_path / "b" / "same.md"])


def test_load_documents_skips_bad_file_when_on_error_given(tmp_path: Path):
    bad = tmp_path / "x.docx"
    bad.write_bytes(b"")
    good = tmp_path / "good.md"
    good.write_text("hello")
    seen: list[tuple[Path, Exception]] = []
    docs = load_documents([bad, good], on_error=lambda p, e: seen.append((p, e)))
    assert [d.id for d in docs] == ["good"]
    assert len(seen) == 1
    assert seen[0][0] == bad
    assert isinstance(seen[0][1], UnsupportedFormatError)


def test_load_documents_raises_without_on_error(tmp_path: Path):
    bad = tmp_path / "x.docx"
    bad.write_bytes(b"")
    good = tmp_path / "good.md"
    good.write_text("hello")
    with pytest.raises(UnsupportedFormatError):
        load_documents([bad, good])
