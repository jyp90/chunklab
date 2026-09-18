# chunklab M1: 코어 엔진 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 문서 로드 → 청킹(3종) → 임베딩(3계열, 캐시) → 검색(dense/bm25/hybrid) → span 기반 메트릭(hit@k, MRR, NDCG, precision, IoU) → 조합 매트릭스 실행을 headless CLI로 돌려 JSON을 내고, baseline 대비 회귀를 감지하는 코어 엔진.

**Architecture:** `chunklab.core`는 UI/서버 의존 없는 순수 라이브러리. 청커·임베더·LLM은 Protocol + 문자열 spec 레지스트리(`"openai:text-embedding-3-small"`)로 플러그인화. 모든 청크는 정규화된 문서 텍스트의 `(start, end)` offset을 보존하며 `chunk.text == doc.text[start:end]`가 불변식. 정답은 청커 독립적인 `Span(doc_id, start, end)`. `chunklab.cli`는 `chunklab.core.runner`를 호출하는 얇은 typer 래퍼.

**Tech Stack:** Python 3.11+, uv, ruff, ty, pytest, numpy, pydantic v2, PyYAML, typer, rank-bm25, pymupdf, openai SDK, google-genai SDK, sentence-transformers(optional extra `local`).

**Spec:** `docs/superpowers/specs/2026-09-18-rag-playground-design.md`

## Global Constraints

- Python `>=3.11`. 도구: uv / ruff / ty (modern-python 스택).
- 기본 설치 의존성에 torch/sentence-transformers 금지. `chunklab[local]` extra로만.
- `chunk.text == doc.text[chunk.start:chunk.end]` 불변식은 모든 청커가 지켜야 하며 테스트로 강제.
- chunk_size / overlap 단위는 **문자(characters)**. 토크나이저 의존성 없음 (v1 결정).
- hit 판정: `overlap(chunk, span) / len(span) >= hit_threshold` (기본 0.5). 기준은 span 길이.
- precision/IoU는 **문자 단위** 집합 연산 (Chroma 방식의 토큰 대신). 결정적이고 토크나이저 불필요.
- 임베딩 캐시 키 = `sha256(embedder.name + "\x00" + text)`. SQLite.
- API 키는 환경변수(`OPENAI_API_KEY`, `GEMINI_API_KEY`)에서만. 코드·설정 파일에 저장 금지.
- 외부 SDK(openai, google.genai, sentence_transformers, fitz)는 **함수 내부에서 lazy import** — import 실패가 패키지 로드를 막지 않게.
- 조합 단위 부분 실패 허용: 한 조합의 예외는 `ComboResult.error`에 기록, 나머지는 계속.
- 커밋 메시지는 Conventional Commits (`feat:`, `test:`, `chore:`, `docs:`). 커밋 끝에 `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- 실제 네트워크를 쓰는 테스트는 `CHUNKLAB_LIVE_TESTS=1`일 때만 실행 (기본 skip).

## 파일 구조

```
chunklab/
├── pyproject.toml
├── README.md
├── src/chunklab/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── models.py            # Document, Span, Chunk, Question, RetrievedChunk
│   │   ├── text.py              # 파일 로드(md/txt/pdf) + 정규화
│   │   ├── chunkers/
│   │   │   ├── __init__.py      # CHUNKERS 레지스트리, build_chunker
│   │   │   ├── base.py          # Chunker Protocol, make_chunk, assert_chunks_valid
│   │   │   ├── recursive.py
│   │   │   ├── sentence_window.py
│   │   │   └── markdown.py
│   │   ├── embedders/
│   │   │   ├── __init__.py      # build_embedder(spec)
│   │   │   ├── base.py          # Embedder Protocol, FakeEmbedder
│   │   │   ├── cache.py         # EmbeddingCache(SQLite), CachedEmbedder
│   │   │   ├── openai.py
│   │   │   ├── gemini.py
│   │   │   └── local.py
│   │   ├── retrieval/
│   │   │   ├── __init__.py
│   │   │   ├── dense.py         # DenseIndex
│   │   │   ├── bm25.py          # BM25Index
│   │   │   └── hybrid.py        # HybridIndex (RRF)
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   ├── overlap.py       # coverage, is_hit
│   │   │   └── ranking.py       # hit@k, MRR, NDCG, precision/IoU, evaluate, mean_metrics
│   │   ├── questions/
│   │   │   ├── __init__.py
│   │   │   ├── llm.py           # LLM Protocol, FakeLLM, OpenAIChat, GeminiChat, build_llm
│   │   │   ├── generate.py      # sample_passages, generate_questions
│   │   │   └── io.py            # save_questions / load_questions (JSON)
│   │   └── runner/
│   │       ├── __init__.py
│   │       ├── config.py        # ExperimentConfig(YAML), Combo, expand_matrix
│   │       ├── run.py           # run_experiment → RunResult
│   │       └── compare.py       # fail-below / baseline 회귀 검사, 표 출력
│   └── cli/
│       ├── __init__.py
│       └── main.py              # typer app: run, generate-questions
└── tests/
    ├── conftest.py
    ├── fixtures/sample.md
    ├── test_models.py
    ├── test_text.py
    ├── chunkers/test_recursive.py, test_sentence_window.py, test_markdown.py, test_registry.py
    ├── embedders/test_fake.py, test_cache.py, test_registry.py
    ├── retrieval/test_dense.py, test_bm25.py, test_hybrid.py
    ├── metrics/test_overlap.py, test_ranking.py
    ├── questions/test_generate.py, test_io.py
    ├── runner/test_config.py, test_run.py, test_compare.py
    └── cli/test_main.py
```

---

### Task 1: 프로젝트 스캐폴딩

**Files:**
- Create: `pyproject.toml`, `src/chunklab/__init__.py`, `src/chunklab/core/__init__.py`, `tests/conftest.py`, `tests/fixtures/sample.md`, `tests/test_smoke.py`, `.gitignore`

**Interfaces:**
- Produces: 패키지 `chunklab` import 가능, `uv run pytest` 동작, fixture `sample_doc_path`.

- [ ] **Step 1: pyproject.toml 작성**

```toml
[project]
name = "chunklab"
version = "0.1.0"
description = "Benchmark your chunking before you ship it."
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
dependencies = [
    "numpy>=1.26",
    "pydantic>=2.7",
    "pyyaml>=6.0",
    "typer>=0.12",
    "rank-bm25>=0.2.2",
    "pymupdf>=1.24",
    "openai>=1.40",
    "google-genai>=1.0",
]

[project.optional-dependencies]
local = ["sentence-transformers>=3.0"]

[project.scripts]
chunklab = "chunklab.cli.main:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/chunklab"]

[tool.uv]
dev-dependencies = ["pytest>=8.0", "ruff>=0.6", "ty"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: 패키지 파일과 .gitignore**

`src/chunklab/__init__.py`:
```python
__version__ = "0.1.0"
```

`src/chunklab/core/__init__.py`: 빈 파일.

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
dist/
.pytest_cache/
.ruff_cache/
*.egg-info/
```

- [ ] **Step 3: 테스트 fixture 문서**

`tests/fixtures/sample.md`:
```markdown
# Refund Policy

Customers may request a refund within 30 days of purchase. Refunds are issued to the original payment method. Digital goods are non-refundable once downloaded.

## Exceptions

Defective products can be returned at any time. Contact support with your order number to start a claim.

# Shipping

Orders ship within 2 business days. International shipping takes 7 to 14 days. Tracking numbers are emailed when the package leaves the warehouse.

## Lost Packages

If a package is marked delivered but not received, wait 48 hours and then file a lost package report.
```

`tests/conftest.py`:
```python
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_doc_path() -> Path:
    return FIXTURES / "sample.md"
```

- [ ] **Step 4: smoke 테스트 작성**

`tests/test_smoke.py`:
```python
import chunklab


def test_version():
    assert chunklab.__version__ == "0.1.0"
```

- [ ] **Step 5: 의존성 설치 및 테스트 실행**

Run: `uv sync && uv run pytest -q`
Expected: `1 passed`

- [ ] **Step 6: lint 확인**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: 오류 없음

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: scaffold chunklab package with uv/ruff/pytest"
```

---

### Task 2: 도메인 모델 (models.py)

**Files:**
- Create: `src/chunklab/core/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces:
  - `Document(id: str, text: str, source: str = "")` frozen dataclass
  - `Span(doc_id: str, start: int, end: int)` frozen; `.length -> int`; `.overlap(other: Span) -> int`
  - `Chunk(doc_id: str, start: int, end: int, text: str, metadata: dict = {})` frozen; `.span -> Span`
  - `Question(id: str, text: str, spans: tuple[Span, ...])` frozen
  - `RetrievedChunk(chunk: Chunk, score: float, rank: int)` frozen, rank는 1-based

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_models.py`:
```python
import pytest

from chunklab.core.models import Chunk, Document, Question, RetrievedChunk, Span


def test_span_length_and_overlap():
    a = Span("d", 0, 10)
    b = Span("d", 5, 20)
    assert a.length == 10
    assert a.overlap(b) == 5
    assert b.overlap(a) == 5


def test_span_no_overlap_different_doc():
    assert Span("d1", 0, 10).overlap(Span("d2", 0, 10)) == 0


def test_span_no_overlap_disjoint():
    assert Span("d", 0, 10).overlap(Span("d", 10, 20)) == 0


def test_span_rejects_empty_or_negative():
    with pytest.raises(ValueError):
        Span("d", 5, 5)
    with pytest.raises(ValueError):
        Span("d", -1, 5)


def test_chunk_span_property():
    c = Chunk("d", 3, 8, "hello")
    assert c.span == Span("d", 3, 8)


def test_chunk_is_hashable_with_metadata():
    c = Chunk("d", 0, 5, "hello", {"heading_path": ("A",)})
    assert hash(c) == hash(Chunk("d", 0, 5, "hello"))


def test_question_and_retrieved_chunk_construct():
    q = Question("q1", "what?", (Span("d", 0, 5),))
    r = RetrievedChunk(Chunk("d", 0, 5, "hello"), 0.9, 1)
    assert q.spans[0].doc_id == "d"
    assert r.rank == 1


def test_document_defaults():
    d = Document("d", "text")
    assert d.source == ""
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_models.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'chunklab.core.models'`

- [ ] **Step 3: 구현**

`src/chunklab/core/models.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source: str = ""


@dataclass(frozen=True)
class Span:
    doc_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"span start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(f"span end must be > start, got start={self.start} end={self.end}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlap(self, other: Span) -> int:
        if self.doc_id != other.doc_id:
            return 0
        return max(0, min(self.end, other.end) - max(self.start, other.start))


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    start: int
    end: int
    text: str
    metadata: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def span(self) -> Span:
        return Span(self.doc_id, self.start, self.end)


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    spans: tuple[Span, ...]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_models.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/chunklab/core/models.py tests/test_models.py
git commit -m "feat(core): add Document, Span, Chunk, Question models"
```

---

### Task 3: 문서 로드와 정규화 (text.py)

**Files:**
- Create: `src/chunklab/core/text.py`
- Test: `tests/test_text.py`

**Interfaces:**
- Consumes: `Document`
- Produces:
  - `normalize(text: str) -> str`
  - `load_document(path: Path) -> Document` — id는 `path.stem`, text는 정규화됨, source는 절대경로 문자열
  - `load_documents(paths: Iterable[Path]) -> list[Document]` — 중복 id면 `ValueError`
  - `UnsupportedFormatError(ValueError)`
  - `DocumentTooLargeError(ValueError)` — 파일 크기 > `MAX_DOCUMENT_BYTES`(10 * 1024 * 1024)면 발생 (스펙 3장 에러 처리)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_text.py`:
```python
from pathlib import Path

import pytest

from chunklab.core.text import (
    MAX_DOCUMENT_BYTES,
    DocumentTooLargeError,
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


def test_load_pdf(tmp_path: Path):
    import fitz

    p = tmp_path / "doc.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Refunds within 30 days.")
    pdf.save(p)
    pdf.close()

    doc = load_document(p)
    assert "Refunds within 30 days." in doc.text


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


def test_load_documents_rejects_duplicate_ids(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "same.md").write_text("one")
    (tmp_path / "b" / "same.md").write_text("two")
    with pytest.raises(ValueError, match="duplicate document id"):
        load_documents([tmp_path / "a" / "same.md", tmp_path / "b" / "same.md"])
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_text.py -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 구현**

`src/chunklab/core/text.py`:
```python
from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from chunklab.core.models import Document

_TEXT_SUFFIXES = {".md", ".txt", ".markdown"}
_PDF_SUFFIXES = {".pdf"}
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024


class UnsupportedFormatError(ValueError):
    pass


class DocumentTooLargeError(ValueError):
    pass


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _read_pdf(path: Path) -> str:
    import fitz  # pymupdf; lazy import

    with fitz.open(path) as pdf:
        return "\n\n".join(page.get_text() for page in pdf)


def load_document(path: Path) -> Document:
    size = path.stat().st_size
    if size > MAX_DOCUMENT_BYTES:
        raise DocumentTooLargeError(
            f"{path} is {size} bytes; limit is {MAX_DOCUMENT_BYTES} bytes"
        )
    suffix = path.suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        raw = path.read_text(encoding="utf-8")
    elif suffix in _PDF_SUFFIXES:
        raw = _read_pdf(path)
    else:
        raise UnsupportedFormatError(f"unsupported file type: {path.suffix} ({path})")
    return Document(id=path.stem, text=normalize(raw), source=str(path.resolve()))


def load_documents(paths: Iterable[Path]) -> list[Document]:
    docs: list[Document] = []
    seen: dict[str, Path] = {}
    for p in paths:
        p = Path(p)
        if p.stem in seen:
            raise ValueError(f"duplicate document id '{p.stem}': {seen[p.stem]} and {p}")
        seen[p.stem] = p
        docs.append(load_document(p))
    return docs
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_text.py -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add src/chunklab/core/text.py tests/test_text.py
git commit -m "feat(core): load md/txt/pdf documents with text normalization"
```

---

### Task 4: 청커 기반 + RecursiveChunker

**Files:**
- Create: `src/chunklab/core/chunkers/__init__.py`(빈 파일, 레지스트리는 Task 6), `src/chunklab/core/chunkers/base.py`, `src/chunklab/core/chunkers/recursive.py`
- Test: `tests/chunkers/__init__.py`(빈), `tests/chunkers/test_base.py`, `tests/chunkers/test_recursive.py`

**Interfaces:**
- Consumes: `Document`, `Chunk`
- Produces:
  - `Chunker` Protocol: `name: str`, `chunk(doc: Document) -> list[Chunk]`
  - `make_chunk(doc: Document, start: int, end: int, **metadata) -> Chunk | None` — 양끝 공백 트림, 비면 None
  - `assert_chunks_valid(doc: Document, chunks: list[Chunk]) -> None` — 불변식 검사(테스트 헬퍼, 위반 시 AssertionError)
  - `RecursiveChunker(chunk_size: int = 512, overlap: int = 0)`; `name = "recursive"`

- [ ] **Step 1: base 테스트 작성**

`tests/chunkers/test_base.py`:
```python
from chunklab.core.chunkers.base import assert_chunks_valid, make_chunk
from chunklab.core.models import Chunk, Document


def test_make_chunk_trims_whitespace_and_keeps_offsets():
    doc = Document("d", "  hello world  \n")
    c = make_chunk(doc, 0, len(doc.text))
    assert c is not None
    assert (c.start, c.end) == (2, 13)
    assert c.text == "hello world"
    assert doc.text[c.start : c.end] == c.text


def test_make_chunk_returns_none_for_whitespace_only():
    doc = Document("d", "   \n")
    assert make_chunk(doc, 0, 4) is None


def test_make_chunk_passes_metadata():
    doc = Document("d", "abc")
    c = make_chunk(doc, 0, 3, heading_path=("H",))
    assert c is not None
    assert c.metadata == {"heading_path": ("H",)}


def test_assert_chunks_valid_detects_offset_mismatch():
    doc = Document("d", "abcdef")
    bad = [Chunk("d", 0, 3, "xyz")]
    try:
        assert_chunks_valid(doc, bad)
    except AssertionError:
        return
    raise AssertionError("expected AssertionError")
```

- [ ] **Step 2: recursive 테스트 작성**

`tests/chunkers/test_recursive.py`:
```python
from pathlib import Path

from chunklab.core.chunkers.base import assert_chunks_valid
from chunklab.core.chunkers.recursive import RecursiveChunker
from chunklab.core.models import Document
from chunklab.core.text import load_document


def _covers_all_non_whitespace(doc: Document, chunks) -> bool:
    covered = set()
    for c in chunks:
        covered.update(range(c.start, c.end))
    return all(i in covered for i, ch in enumerate(doc.text) if not ch.isspace())


def test_short_doc_is_single_chunk():
    doc = Document("d", "short text\n")
    chunks = RecursiveChunker(chunk_size=100).chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"


def test_offsets_invariant_and_size_bound(sample_doc_path: Path):
    doc = load_document(sample_doc_path)
    chunker = RecursiveChunker(chunk_size=120, overlap=0)
    chunks = chunker.chunk(doc)
    assert_chunks_valid(doc, chunks)
    assert len(chunks) > 3
    assert all(len(c.text) <= 120 for c in chunks)
    assert _covers_all_non_whitespace(doc, chunks)


def test_chunks_are_ordered_and_non_overlapping_without_overlap(sample_doc_path: Path):
    doc = load_document(sample_doc_path)
    chunks = RecursiveChunker(chunk_size=150).chunk(doc)
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        assert prev.end <= cur.start


def test_overlap_extends_start_backwards(sample_doc_path: Path):
    doc = load_document(sample_doc_path)
    base = RecursiveChunker(chunk_size=150, overlap=0).chunk(doc)
    over = RecursiveChunker(chunk_size=150, overlap=30).chunk(doc)
    assert_chunks_valid(doc, over)
    assert len(base) == len(over)
    assert over[0].start == base[0].start
    for b, o in zip(base[1:], over[1:], strict=True):
        assert o.start <= b.start
        assert b.start - o.start <= 30
        assert o.end == b.end


def test_prefers_paragraph_boundaries():
    doc = Document("d", "para one is here.\n\npara two is here.\n\npara three is here.\n")
    # each paragraph incl. separator is 19-20 chars; 20 forbids merging neighbours
    chunks = RecursiveChunker(chunk_size=20).chunk(doc)
    assert [c.text for c in chunks] == [
        "para one is here.",
        "para two is here.",
        "para three is here.",
    ]


def test_hard_split_when_no_separator():
    doc = Document("d", "x" * 25)
    chunks = RecursiveChunker(chunk_size=10).chunk(doc)
    assert [c.text for c in chunks] == ["x" * 10, "x" * 10, "x" * 5]
    assert_chunks_valid(doc, chunks)
```

- [ ] **Step 3: 실패 확인**

Run: `uv run pytest tests/chunkers -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: base 구현**

`src/chunklab/core/chunkers/base.py`:
```python
from __future__ import annotations

from typing import Protocol, runtime_checkable

from chunklab.core.models import Chunk, Document


@runtime_checkable
class Chunker(Protocol):
    name: str

    def chunk(self, doc: Document) -> list[Chunk]: ...


def make_chunk(doc: Document, start: int, end: int, **metadata) -> Chunk | None:
    text = doc.text
    end = min(end, len(text))
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return Chunk(doc.id, start, end, text[start:end], dict(metadata))


def assert_chunks_valid(doc: Document, chunks: list[Chunk]) -> None:
    for c in chunks:
        assert c.doc_id == doc.id, f"chunk doc_id {c.doc_id} != {doc.id}"
        assert 0 <= c.start < c.end <= len(doc.text), f"bad offsets {c.start}:{c.end}"
        assert doc.text[c.start : c.end] == c.text, f"text mismatch at {c.start}:{c.end}"
```

- [ ] **Step 5: recursive 구현**

`src/chunklab/core/chunkers/recursive.py`:
```python
from __future__ import annotations

import re
from dataclasses import dataclass

from chunklab.core.chunkers.base import make_chunk
from chunklab.core.models import Chunk, Document

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

Segment = tuple[int, int]


@dataclass
class RecursiveChunker:
    chunk_size: int = 512
    overlap: int = 0
    separators: tuple[str, ...] = DEFAULT_SEPARATORS
    name: str = "recursive"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if not 0 <= self.overlap < self.chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    def chunk(self, doc: Document) -> list[Chunk]:
        segments = self._split(doc.text, 0, list(self.separators))
        merged = self._merge(segments)
        chunks: list[Chunk] = []
        prev_start = 0
        for i, (s, e) in enumerate(merged):
            start = s if i == 0 else max(s - self.overlap, prev_start)
            c = make_chunk(doc, start, e)
            if c is not None:
                chunks.append(c)
            prev_start = s
        return chunks

    def _split(self, text: str, offset: int, separators: list[str]) -> list[Segment]:
        if not text:
            return []
        if len(text) <= self.chunk_size or not separators:
            return [(offset, offset + len(text))]
        sep, rest = separators[0], separators[1:]
        if sep == "":
            return [
                (offset + i, offset + min(i + self.chunk_size, len(text)))
                for i in range(0, len(text), self.chunk_size)
            ]
        pieces: list[Segment] = []
        pos = 0
        for m in re.finditer(re.escape(sep), text):
            pieces.append((pos, m.end()))
            pos = m.end()
        if pos < len(text):
            pieces.append((pos, len(text)))
        if len(pieces) <= 1:
            return self._split(text, offset, rest)
        out: list[Segment] = []
        for s, e in pieces:
            if e - s > self.chunk_size:
                out.extend(self._split(text[s:e], offset + s, rest))
            else:
                out.append((offset + s, offset + e))
        return out

    def _merge(self, segments: list[Segment]) -> list[Segment]:
        merged: list[Segment] = []
        cur: Segment | None = None
        for s, e in segments:
            if cur is None:
                cur = (s, e)
            elif e - cur[0] <= self.chunk_size:
                cur = (cur[0], e)
            else:
                merged.append(cur)
                cur = (s, e)
        if cur is not None:
            merged.append(cur)
        return merged
```

- [ ] **Step 6: 통과 확인**

Run: `uv run pytest tests/chunkers -q`
Expected: `10 passed`

- [ ] **Step 7: Commit**

```bash
git add src/chunklab/core/chunkers tests/chunkers
git commit -m "feat(chunkers): add Chunker protocol and offset-preserving RecursiveChunker"
```

---

### Task 5: SentenceWindowChunker

**Files:**
- Create: `src/chunklab/core/chunkers/sentence_window.py`
- Test: `tests/chunkers/test_sentence_window.py`

**Interfaces:**
- Consumes: `make_chunk`, `Document`
- Produces:
  - `split_sentences(text: str) -> list[tuple[int, int]]` — 문장 (start, end) offset, 공백만인 문장 제외
  - `SentenceWindowChunker(window: int = 1)`; `name = "sentence_window"`; 각 문장 i에 대해 `[i-window, i+window]` 범위를 하나의 청크로. `metadata["center"] = i`

설계 메모: 고전적 sentence-window는 "단일 문장 임베딩 + 검색 후 윈도우 확장"이지만 v1은 윈도우 전체를 청크로 임베딩. 단순화이며 README에 명시.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/chunkers/test_sentence_window.py`:
```python
from pathlib import Path

from chunklab.core.chunkers.base import assert_chunks_valid
from chunklab.core.chunkers.sentence_window import SentenceWindowChunker, split_sentences
from chunklab.core.models import Document
from chunklab.core.text import load_document


def test_split_sentences_offsets():
    text = "First one. Second one!  Third?\n"
    sents = split_sentences(text)
    assert [text[s:e] for s, e in sents] == ["First one.", "Second one!", "Third?"]


def test_split_sentences_treats_blank_line_as_boundary():
    text = "no period here\n\nnext para\n"
    sents = split_sentences(text)
    assert [text[s:e] for s, e in sents] == ["no period here", "next para"]


def test_window_zero_is_one_chunk_per_sentence():
    doc = Document("d", "A one. B two. C three.\n")
    chunks = SentenceWindowChunker(window=0).chunk(doc)
    assert [c.text for c in chunks] == ["A one.", "B two.", "C three."]
    assert [c.metadata["center"] for c in chunks] == [0, 1, 2]


def test_window_one_spans_neighbors():
    doc = Document("d", "A one. B two. C three.\n")
    chunks = SentenceWindowChunker(window=1).chunk(doc)
    assert [c.text for c in chunks] == [
        "A one. B two.",
        "A one. B two. C three.",
        "B two. C three.",
    ]
    assert_chunks_valid(doc, chunks)


def test_sample_doc_invariant(sample_doc_path: Path):
    doc = load_document(sample_doc_path)
    chunks = SentenceWindowChunker(window=2).chunk(doc)
    assert_chunks_valid(doc, chunks)
    assert len(chunks) == len(split_sentences(doc.text))
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/chunkers/test_sentence_window.py -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 구현**

`src/chunklab/core/chunkers/sentence_window.py`:
```python
from __future__ import annotations

import re
from dataclasses import dataclass

from chunklab.core.chunkers.base import make_chunk
from chunklab.core.models import Chunk, Document

_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+|\n{2,}")


def split_sentences(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _BOUNDARY_RE.finditer(text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    out: list[tuple[int, int]] = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            out.append((s, e))
    return out


@dataclass
class SentenceWindowChunker:
    window: int = 1
    name: str = "sentence_window"

    def __post_init__(self) -> None:
        if self.window < 0:
            raise ValueError("window must be >= 0")

    def chunk(self, doc: Document) -> list[Chunk]:
        sents = split_sentences(doc.text)
        chunks: list[Chunk] = []
        for i in range(len(sents)):
            lo = max(0, i - self.window)
            hi = min(len(sents) - 1, i + self.window)
            c = make_chunk(doc, sents[lo][0], sents[hi][1], center=i)
            if c is not None:
                chunks.append(c)
        return chunks
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/chunkers/test_sentence_window.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/chunklab/core/chunkers/sentence_window.py tests/chunkers/test_sentence_window.py
git commit -m "feat(chunkers): add SentenceWindowChunker"
```

---

### Task 6: MarkdownChunker + 레지스트리

**Files:**
- Create: `src/chunklab/core/chunkers/markdown.py`
- Modify: `src/chunklab/core/chunkers/__init__.py`
- Test: `tests/chunkers/test_markdown.py`, `tests/chunkers/test_registry.py`

**Interfaces:**
- Consumes: `RecursiveChunker`, `make_chunk`
- Produces:
  - `MarkdownChunker(chunk_size: int = 1024, overlap: int = 0)`; `name = "markdown"`; 헤딩 단위 섹션, 섹션이 chunk_size 초과 시 내부를 RecursiveChunker로 재분할. `metadata["heading_path"] = tuple[str, ...]`
  - `CHUNKERS: dict[str, type]` = `{"recursive", "sentence_window", "markdown"}`
  - `build_chunker(name: str, **params) -> Chunker` — 미등록 이름은 `KeyError`

- [ ] **Step 1: markdown 테스트 작성**

`tests/chunkers/test_markdown.py`:
```python
from pathlib import Path

from chunklab.core.chunkers.base import assert_chunks_valid
from chunklab.core.chunkers.markdown import MarkdownChunker
from chunklab.core.models import Document
from chunklab.core.text import load_document


def test_sections_by_heading_with_paths(sample_doc_path: Path):
    doc = load_document(sample_doc_path)
    chunks = MarkdownChunker(chunk_size=2000).chunk(doc)
    assert_chunks_valid(doc, chunks)
    paths = [c.metadata["heading_path"] for c in chunks]
    assert paths == [
        ("Refund Policy",),
        ("Refund Policy", "Exceptions"),
        ("Shipping",),
        ("Shipping", "Lost Packages"),
    ]
    assert chunks[0].text.startswith("# Refund Policy")
    assert chunks[1].text.startswith("## Exceptions")


def test_preamble_before_first_heading_has_empty_path():
    doc = Document("d", "intro text\n\n# H1\n\nbody\n")
    chunks = MarkdownChunker().chunk(doc)
    assert chunks[0].text == "intro text"
    assert chunks[0].metadata["heading_path"] == ()
    assert chunks[1].metadata["heading_path"] == ("H1",)


def test_large_section_is_subdivided_with_correct_offsets():
    body = " ".join(f"word{i}" for i in range(200))
    doc = Document("d", f"# Big\n\n{body}\n")
    chunks = MarkdownChunker(chunk_size=200).chunk(doc)
    assert len(chunks) > 3
    assert_chunks_valid(doc, chunks)
    assert all(c.metadata["heading_path"] == ("Big",) for c in chunks)
    assert all(len(c.text) <= 200 for c in chunks)


def test_no_headings_falls_back_to_whole_doc():
    doc = Document("d", "plain text only\n")
    chunks = MarkdownChunker().chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].metadata["heading_path"] == ()
```

- [ ] **Step 2: registry 테스트 작성**

`tests/chunkers/test_registry.py`:
```python
import pytest

from chunklab.core.chunkers import CHUNKERS, build_chunker
from chunklab.core.chunkers.base import Chunker


def test_registry_has_three_v1_chunkers():
    assert set(CHUNKERS) == {"recursive", "sentence_window", "markdown"}


def test_build_chunker_passes_params():
    c = build_chunker("recursive", chunk_size=64, overlap=8)
    assert isinstance(c, Chunker)
    assert c.name == "recursive"
    assert c.chunk_size == 64  # type: ignore[attr-defined]


def test_build_unknown_chunker():
    with pytest.raises(KeyError, match="unknown chunker"):
        build_chunker("semantic")
```

- [ ] **Step 3: 실패 확인**

Run: `uv run pytest tests/chunkers/test_markdown.py tests/chunkers/test_registry.py -q`
Expected: FAIL — `ModuleNotFoundError` / `ImportError`

- [ ] **Step 4: markdown 구현**

`src/chunklab/core/chunkers/markdown.py`:
```python
from __future__ import annotations

import re
from dataclasses import dataclass

from chunklab.core.chunkers.base import make_chunk
from chunklab.core.chunkers.recursive import RecursiveChunker
from chunklab.core.models import Chunk, Document

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)

Section = tuple[int, int, tuple[str, ...]]


@dataclass
class MarkdownChunker:
    chunk_size: int = 1024
    overlap: int = 0
    name: str = "markdown"

    def chunk(self, doc: Document) -> list[Chunk]:
        inner = RecursiveChunker(chunk_size=self.chunk_size, overlap=self.overlap)
        chunks: list[Chunk] = []
        for s, e, path in self._sections(doc.text):
            if e - s <= self.chunk_size:
                c = make_chunk(doc, s, e, heading_path=path)
                if c is not None:
                    chunks.append(c)
                continue
            sub = Document(id=doc.id, text=doc.text[s:e], source=doc.source)
            for sc in inner.chunk(sub):
                chunks.append(
                    Chunk(doc.id, sc.start + s, sc.end + s, sc.text, {"heading_path": path})
                )
        return chunks

    @staticmethod
    def _sections(text: str) -> list[Section]:
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return [(0, len(text), ())]
        out: list[Section] = []
        if matches[0].start() > 0:
            out.append((0, matches[0].start(), ()))
        stack: list[tuple[int, str]] = []
        for i, m in enumerate(matches):
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            out.append((m.start(), end, tuple(t for _, t in stack)))
        return out
```

- [ ] **Step 5: registry 구현**

`src/chunklab/core/chunkers/__init__.py`:
```python
from __future__ import annotations

from chunklab.core.chunkers.base import Chunker
from chunklab.core.chunkers.markdown import MarkdownChunker
from chunklab.core.chunkers.recursive import RecursiveChunker
from chunklab.core.chunkers.sentence_window import SentenceWindowChunker

CHUNKERS: dict[str, type] = {
    "recursive": RecursiveChunker,
    "sentence_window": SentenceWindowChunker,
    "markdown": MarkdownChunker,
}


def build_chunker(name: str, **params) -> Chunker:
    try:
        cls = CHUNKERS[name]
    except KeyError:
        raise KeyError(f"unknown chunker '{name}'; available: {sorted(CHUNKERS)}") from None
    return cls(**params)


__all__ = ["CHUNKERS", "Chunker", "build_chunker"]
```

- [ ] **Step 6: 통과 확인**

Run: `uv run pytest tests/chunkers -q`
Expected: `22 passed`

- [ ] **Step 7: Commit**

```bash
git add src/chunklab/core/chunkers tests/chunkers
git commit -m "feat(chunkers): add MarkdownChunker and chunker registry"
```

---

### Task 7: 메트릭 (overlap.py, ranking.py)

**Files:**
- Create: `src/chunklab/core/metrics/__init__.py`, `src/chunklab/core/metrics/overlap.py`, `src/chunklab/core/metrics/ranking.py`
- Test: `tests/metrics/__init__.py`(빈), `tests/metrics/test_overlap.py`, `tests/metrics/test_ranking.py`

**Interfaces:**
- Consumes: `Chunk`, `Span`
- Produces (overlap.py):
  - `coverage(chunk: Chunk, gold: Span) -> float` — `overlap / gold.length`, 0.0~1.0
  - `best_coverage(chunk: Chunk, golds: Sequence[Span]) -> float`
  - `is_hit(chunk: Chunk, golds: Sequence[Span], threshold: float) -> bool`
- Produces (ranking.py):
  - `QuestionMetrics(hit: bool, reciprocal_rank: float, ndcg: float, precision: float, iou: float)` frozen
  - `hit_at_k(chunks, golds, k, threshold) -> bool`
  - `reciprocal_rank(chunks, golds, k, threshold) -> float`
  - `ndcg_at_k(chunks, golds, k) -> float` — gain = best_coverage, IDCG = golds 수(≤k)만큼 gain 1.0
  - `char_precision_iou(chunks, golds) -> tuple[float, float]` — 문자 집합 기준
  - `evaluate(chunks: Sequence[Chunk], golds: Sequence[Span], k: int, threshold: float) -> QuestionMetrics` (chunks는 rank 순, 처음 k개만 사용)
  - `mean_metrics(items: Sequence[QuestionMetrics], k: int) -> dict[str, float]` — 키 `f"hit@{k}"`, `"mrr"`, `"ndcg"`, `"precision"`, `"iou"`; 빈 입력은 모두 0.0
  - `__init__.py`는 위 전부 re-export

- [ ] **Step 1: overlap 테스트**

`tests/metrics/test_overlap.py`:
```python
from chunklab.core.metrics.overlap import best_coverage, coverage, is_hit
from chunklab.core.models import Chunk, Span


def _c(s, e, doc="d"):
    return Chunk(doc, s, e, "x" * (e - s))


def test_coverage_full():
    assert coverage(_c(0, 100), Span("d", 10, 20)) == 1.0


def test_coverage_partial_is_relative_to_gold_length():
    assert coverage(_c(0, 15), Span("d", 10, 20)) == 0.5


def test_coverage_zero_when_disjoint_or_other_doc():
    assert coverage(_c(0, 5), Span("d", 10, 20)) == 0.0
    assert coverage(_c(0, 50, doc="other"), Span("d", 10, 20)) == 0.0


def test_best_coverage_takes_max():
    golds = [Span("d", 10, 20), Span("d", 100, 110)]
    assert best_coverage(_c(0, 15), golds) == 0.5
    assert best_coverage(_c(0, 15), []) == 0.0


def test_is_hit_threshold_inclusive():
    golds = [Span("d", 10, 20)]
    assert is_hit(_c(0, 15), golds, threshold=0.5) is True
    assert is_hit(_c(0, 14), golds, threshold=0.5) is False
```

- [ ] **Step 2: ranking 테스트**

`tests/metrics/test_ranking.py`:
```python
import math

import pytest

from chunklab.core.metrics.ranking import (
    QuestionMetrics,
    char_precision_iou,
    evaluate,
    hit_at_k,
    mean_metrics,
    ndcg_at_k,
    reciprocal_rank,
)
from chunklab.core.models import Chunk, Span


def _c(s, e):
    return Chunk("d", s, e, "x" * (e - s))


GOLD = [Span("d", 100, 200)]
MISS = _c(0, 50)
HIT = _c(90, 210)
HALF = _c(150, 300)  # covers 50 of 100 gold chars


def test_hit_at_k_respects_k():
    ranked = [MISS, MISS, HIT]
    assert hit_at_k(ranked, GOLD, k=3, threshold=0.5) is True
    assert hit_at_k(ranked, GOLD, k=2, threshold=0.5) is False


def test_reciprocal_rank():
    assert reciprocal_rank([MISS, HIT], GOLD, k=5, threshold=0.5) == 0.5
    assert reciprocal_rank([HIT], GOLD, k=5, threshold=0.5) == 1.0
    assert reciprocal_rank([MISS, MISS], GOLD, k=5, threshold=0.5) == 0.0
    assert reciprocal_rank([MISS, MISS, HIT], GOLD, k=2, threshold=0.5) == 0.0


def test_ndcg_perfect_first_rank():
    assert ndcg_at_k([HIT, MISS], GOLD, k=2) == pytest.approx(1.0)


def test_ndcg_graded_by_coverage_and_position():
    # gain at rank 2 = 0.5 -> dcg = 0.5 / log2(3); idcg = 1.0
    expected = 0.5 / math.log2(3)
    assert ndcg_at_k([MISS, HALF], GOLD, k=2) == pytest.approx(expected)


def test_ndcg_two_golds_idcg_uses_two_slots():
    golds = [Span("d", 100, 200), Span("d", 500, 600)]
    ranked = [_c(100, 200), _c(0, 10), _c(500, 600)]
    dcg = 1.0 + 1.0 / math.log2(4)
    idcg = 1.0 + 1.0 / math.log2(3)
    assert ndcg_at_k(ranked, golds, k=3) == pytest.approx(dcg / idcg)


def test_char_precision_iou():
    # retrieved chars: [90,210) = 120 chars; gold [100,200) = 100 chars
    p, iou = char_precision_iou([HIT], GOLD)
    assert p == pytest.approx(100 / 120)
    assert iou == pytest.approx(100 / 120)  # union is also 120


def test_char_precision_iou_overlapping_chunks_counted_once():
    p, iou = char_precision_iou([_c(100, 150), _c(120, 200)], GOLD)
    assert p == pytest.approx(1.0)
    assert iou == pytest.approx(1.0)


def test_char_precision_iou_empty_retrieval():
    assert char_precision_iou([], GOLD) == (0.0, 0.0)


def test_evaluate_uses_top_k_only():
    m = evaluate([MISS, MISS, HIT], GOLD, k=2, threshold=0.5)
    assert m == QuestionMetrics(hit=False, reciprocal_rank=0.0, ndcg=0.0, precision=0.0, iou=0.0)


def test_mean_metrics_keys_and_values():
    a = QuestionMetrics(True, 1.0, 1.0, 0.8, 0.8)
    b = QuestionMetrics(False, 0.0, 0.0, 0.0, 0.0)
    out = mean_metrics([a, b], k=5)
    assert out == {"hit@5": 0.5, "mrr": 0.5, "ndcg": 0.5, "precision": 0.4, "iou": 0.4}


def test_mean_metrics_empty():
    assert mean_metrics([], k=3) == {"hit@3": 0.0, "mrr": 0.0, "ndcg": 0.0, "precision": 0.0, "iou": 0.0}
```

- [ ] **Step 3: 실패 확인**

Run: `uv run pytest tests/metrics -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: overlap 구현**

`src/chunklab/core/metrics/overlap.py`:
```python
from __future__ import annotations

from collections.abc import Sequence

from chunklab.core.models import Chunk, Span


def coverage(chunk: Chunk, gold: Span) -> float:
    return chunk.span.overlap(gold) / gold.length


def best_coverage(chunk: Chunk, golds: Sequence[Span]) -> float:
    return max((coverage(chunk, g) for g in golds), default=0.0)


def is_hit(chunk: Chunk, golds: Sequence[Span], threshold: float) -> bool:
    return best_coverage(chunk, golds) >= threshold
```

- [ ] **Step 5: ranking 구현**

`src/chunklab/core/metrics/ranking.py`:
```python
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from chunklab.core.metrics.overlap import best_coverage, is_hit
from chunklab.core.models import Chunk, Span


@dataclass(frozen=True)
class QuestionMetrics:
    hit: bool
    reciprocal_rank: float
    ndcg: float
    precision: float
    iou: float


def hit_at_k(chunks: Sequence[Chunk], golds: Sequence[Span], k: int, threshold: float) -> bool:
    return any(is_hit(c, golds, threshold) for c in chunks[:k])


def reciprocal_rank(
    chunks: Sequence[Chunk], golds: Sequence[Span], k: int, threshold: float
) -> float:
    for rank, c in enumerate(chunks[:k], start=1):
        if is_hit(c, golds, threshold):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(chunks: Sequence[Chunk], golds: Sequence[Span], k: int) -> float:
    if not golds:
        return 0.0
    dcg = sum(
        best_coverage(c, golds) / math.log2(rank + 1)
        for rank, c in enumerate(chunks[:k], start=1)
    )
    ideal_slots = min(len(golds), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_slots + 1))
    return dcg / idcg if idcg else 0.0


def _positions(spans: Sequence[Span]) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for s in spans:
        out.update((s.doc_id, i) for i in range(s.start, s.end))
    return out


def char_precision_iou(chunks: Sequence[Chunk], golds: Sequence[Span]) -> tuple[float, float]:
    retrieved = _positions([c.span for c in chunks])
    if not retrieved:
        return 0.0, 0.0
    gold = _positions(golds)
    inter = len(retrieved & gold)
    union = len(retrieved | gold)
    return inter / len(retrieved), (inter / union if union else 0.0)


def evaluate(
    chunks: Sequence[Chunk], golds: Sequence[Span], k: int, threshold: float
) -> QuestionMetrics:
    top = list(chunks[:k])
    precision, iou = char_precision_iou(top, golds)
    return QuestionMetrics(
        hit=hit_at_k(top, golds, k, threshold),
        reciprocal_rank=reciprocal_rank(top, golds, k, threshold),
        ndcg=ndcg_at_k(top, golds, k),
        precision=precision,
        iou=iou,
    )


def mean_metrics(items: Sequence[QuestionMetrics], k: int) -> dict[str, float]:
    n = len(items)

    def avg(values: list[float]) -> float:
        return sum(values) / n if n else 0.0

    return {
        f"hit@{k}": avg([1.0 if m.hit else 0.0 for m in items]),
        "mrr": avg([m.reciprocal_rank for m in items]),
        "ndcg": avg([m.ndcg for m in items]),
        "precision": avg([m.precision for m in items]),
        "iou": avg([m.iou for m in items]),
    }
```

`src/chunklab/core/metrics/__init__.py`:
```python
from chunklab.core.metrics.overlap import best_coverage, coverage, is_hit
from chunklab.core.metrics.ranking import (
    QuestionMetrics,
    char_precision_iou,
    evaluate,
    hit_at_k,
    mean_metrics,
    ndcg_at_k,
    reciprocal_rank,
)

__all__ = [
    "QuestionMetrics",
    "best_coverage",
    "char_precision_iou",
    "coverage",
    "evaluate",
    "hit_at_k",
    "is_hit",
    "mean_metrics",
    "ndcg_at_k",
    "reciprocal_rank",
]
```

- [ ] **Step 6: 통과 확인**

Run: `uv run pytest tests/metrics -q`
Expected: `17 passed`

- [ ] **Step 7: Commit**

```bash
git add src/chunklab/core/metrics tests/metrics
git commit -m "feat(metrics): span-based hit@k, MRR, NDCG, char precision/IoU"
```

---

### Task 8: 임베더 기반 + FakeEmbedder + SQLite 캐시

**Files:**
- Create: `src/chunklab/core/embedders/__init__.py`(빈, Task 9에서 채움), `src/chunklab/core/embedders/base.py`, `src/chunklab/core/embedders/cache.py`
- Test: `tests/embedders/__init__.py`(빈), `tests/embedders/test_fake.py`, `tests/embedders/test_cache.py`

**Interfaces:**
- Produces:
  - `Embedder` Protocol: `name: str`, `embed(texts: list[str]) -> np.ndarray` (shape `(n, d)`, float32, L2 정규화된 행)
  - `FakeEmbedder(dim: int = 256)`; `name = "fake"`; 결정적 bag-of-words 해싱(단어가 겹치면 코사인 유사도 높음). 테스트·오프라인 데모용
  - `EmbeddingCache(path: Path)`; `.get_many(keys: list[str]) -> dict[str, np.ndarray]`; `.put_many(items: dict[str, np.ndarray]) -> None`; `.close()`; 컨텍스트 매니저
  - `CachedEmbedder(inner: Embedder, cache: EmbeddingCache)`; `.name == inner.name`; `.embed`는 미스만 inner에 위임, `.misses: int` 카운터(테스트용)
  - `cache_key(embedder_name: str, text: str) -> str` = sha256 hex

- [ ] **Step 1: fake 테스트**

`tests/embedders/test_fake.py`:
```python
import numpy as np

from chunklab.core.embedders.base import Embedder, FakeEmbedder


def test_fake_embedder_shape_dtype_normalized():
    e = FakeEmbedder(dim=64)
    v = e.embed(["hello world", "another text"])
    assert isinstance(e, Embedder)
    assert v.shape == (2, 64)
    assert v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


def test_fake_embedder_deterministic():
    a = FakeEmbedder().embed(["same text"])
    b = FakeEmbedder().embed(["same text"])
    assert np.array_equal(a, b)


def test_fake_embedder_similarity_reflects_shared_words():
    e = FakeEmbedder()
    v = e.embed(["refund within 30 days", "refund policy 30 days", "shipping tracking number"])
    sim_close = float(v[0] @ v[1])
    sim_far = float(v[0] @ v[2])
    assert sim_close > sim_far


def test_fake_embedder_empty_input():
    v = FakeEmbedder(dim=8).embed([])
    assert v.shape == (0, 8)
```

- [ ] **Step 2: cache 테스트**

`tests/embedders/test_cache.py`:
```python
from pathlib import Path

import numpy as np

from chunklab.core.embedders.base import FakeEmbedder
from chunklab.core.embedders.cache import CachedEmbedder, EmbeddingCache, cache_key


def test_cache_key_depends_on_name_and_text():
    assert cache_key("a", "t") != cache_key("b", "t")
    assert cache_key("a", "t") != cache_key("a", "u")
    assert cache_key("a", "t") == cache_key("a", "t")


def test_cache_roundtrip(tmp_path: Path):
    with EmbeddingCache(tmp_path / "c.db") as cache:
        vec = np.arange(4, dtype=np.float32)
        cache.put_many({"k1": vec})
        got = cache.get_many(["k1", "missing"])
        assert set(got) == {"k1"}
        assert np.array_equal(got["k1"], vec)


def test_cache_persists_across_instances(tmp_path: Path):
    p = tmp_path / "c.db"
    with EmbeddingCache(p) as c1:
        c1.put_many({"k": np.ones(3, dtype=np.float32)})
    with EmbeddingCache(p) as c2:
        assert "k" in c2.get_many(["k"])


def test_cached_embedder_only_calls_inner_for_misses(tmp_path: Path):
    inner = FakeEmbedder(dim=16)
    with EmbeddingCache(tmp_path / "c.db") as cache:
        ce = CachedEmbedder(inner, cache)
        assert ce.name == "fake"
        first = ce.embed(["a", "b", "c"])
        assert ce.misses == 3
        second = ce.embed(["b", "c", "d"])
        assert ce.misses == 4
        assert np.array_equal(first[1], second[0])
        assert np.array_equal(first[2], second[1])
        assert second.shape == (3, 16)


def test_cached_embedder_preserves_order_and_duplicates(tmp_path: Path):
    inner = FakeEmbedder(dim=8)
    with EmbeddingCache(tmp_path / "c.db") as cache:
        ce = CachedEmbedder(inner, cache)
        out = ce.embed(["x", "y", "x"])
        direct = inner.embed(["x", "y", "x"])
        assert np.array_equal(out, direct)
```

- [ ] **Step 3: 실패 확인**

Run: `uv run pytest tests/embedders -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: base 구현**

`src/chunklab/core/embedders/base.py`:
```python
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

_WORD_RE = re.compile(r"\w+")


@runtime_checkable
class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def l2_normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.size == 0:
        return m
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


@dataclass
class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder for tests and offline demos."""

    dim: int = 256
    name: str = "fake"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in _WORD_RE.findall(text.lower()):
                h = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h, "little") % self.dim
                out[row, idx] += 1.0
        return l2_normalize(out)
```

- [ ] **Step 5: cache 구현**

`src/chunklab/core/embedders/cache.py`:
```python
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

from chunklab.core.embedders.base import Embedder


def cache_key(embedder_name: str, text: str) -> str:
    return hashlib.sha256(f"{embedder_name}\x00{text}".encode()).hexdigest()


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, dim INTEGER, vec BLOB)"
        )
        self._conn.commit()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        if not keys:
            return {}
        out: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 500):
            batch = keys[i : i + 500]
            marks = ",".join("?" * len(batch))
            rows = self._conn.execute(
                f"SELECT key, dim, vec FROM embeddings WHERE key IN ({marks})", batch
            ).fetchall()
            for key, dim, blob in rows:
                out[key] = np.frombuffer(blob, dtype=np.float32).reshape(dim).copy()
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        rows = [
            (k, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes())
            for k, v in items.items()
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings (key, dim, vec) VALUES (?, ?, ?)", rows
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> EmbeddingCache:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class CachedEmbedder:
    def __init__(self, inner: Embedder, cache: EmbeddingCache) -> None:
        self._inner = inner
        self._cache = cache
        self.name = inner.name
        self.misses = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [cache_key(self.name, t) for t in texts]
        found = self._cache.get_many(list(dict.fromkeys(keys)))
        missing_texts: dict[str, str] = {}
        for k, t in zip(keys, texts, strict=True):
            if k not in found:
                missing_texts[k] = t
        if missing_texts:
            vecs = self._inner.embed(list(missing_texts.values()))
            self.misses += len(missing_texts)
            new = dict(zip(missing_texts.keys(), vecs, strict=True))
            self._cache.put_many(new)
            found.update(new)
        if not texts:
            probe = self._inner.embed([])
            return np.zeros((0, probe.shape[1]), dtype=np.float32)
        return np.stack([found[k] for k in keys]).astype(np.float32)
```

- [ ] **Step 6: 통과 확인**

Run: `uv run pytest tests/embedders -q`
Expected: `9 passed`

- [ ] **Step 7: Commit**

```bash
git add src/chunklab/core/embedders tests/embedders
git commit -m "feat(embedders): Embedder protocol, FakeEmbedder, SQLite embedding cache"
```

---

### Task 9: OpenAI / Gemini / Local 임베더 + 레지스트리

**Files:**
- Create: `src/chunklab/core/embedders/openai.py`, `gemini.py`, `local.py`
- Modify: `src/chunklab/core/embedders/__init__.py`
- Test: `tests/embedders/test_registry.py`, `tests/embedders/test_live.py`

**Interfaces:**
- Produces:
  - `OpenAIEmbedder(model: str = "text-embedding-3-small", batch_size: int = 100)`; `name = f"openai:{model}"`
  - `GeminiEmbedder(model: str = "gemini-embedding-001", batch_size: int = 100)`; `name = f"gemini:{model}"`
  - `LocalEmbedder(model: str = "all-MiniLM-L6-v2")`; `name = f"local:{model}"`; sentence-transformers 미설치 시 `ImportError("... pip install 'chunklab[local]'")`
  - `build_embedder(spec: str) -> Embedder` — spec 형식 `"provider[:model]"`, provider ∈ `{fake, openai, gemini, local}`. `fake`는 `FakeEmbedder()`. 미지 provider는 `KeyError`
  - `MissingApiKeyError(RuntimeError)` — 환경변수 없을 때 `embed()` 호출 시점에 발생

- [ ] **Step 1: registry 테스트**

`tests/embedders/test_registry.py`:
```python
import pytest

from chunklab.core.embedders import MissingApiKeyError, build_embedder
from chunklab.core.embedders.base import FakeEmbedder
from chunklab.core.embedders.gemini import GeminiEmbedder
from chunklab.core.embedders.local import LocalEmbedder
from chunklab.core.embedders.openai import OpenAIEmbedder


def test_build_fake():
    assert isinstance(build_embedder("fake"), FakeEmbedder)


def test_build_openai_default_and_custom_model():
    e = build_embedder("openai")
    assert isinstance(e, OpenAIEmbedder)
    assert e.name == "openai:text-embedding-3-small"
    assert build_embedder("openai:text-embedding-3-large").name == "openai:text-embedding-3-large"


def test_build_gemini_and_local_names():
    g = build_embedder("gemini")
    assert isinstance(g, GeminiEmbedder)
    assert g.name == "gemini:gemini-embedding-001"
    loc = build_embedder("local:all-MiniLM-L6-v2")
    assert isinstance(loc, LocalEmbedder)
    assert loc.name == "local:all-MiniLM-L6-v2"


def test_build_unknown_provider():
    with pytest.raises(KeyError, match="unknown embedder provider"):
        build_embedder("cohere:embed-v3")


def test_openai_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError, match="OPENAI_API_KEY"):
        OpenAIEmbedder().embed(["x"])


def test_gemini_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError, match="GEMINI_API_KEY"):
        GeminiEmbedder().embed(["x"])


def test_local_missing_dependency_message(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name.startswith("sentence_transformers"):
            raise ImportError("no module")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"chunklab\[local\]"):
        LocalEmbedder().embed(["x"])
```

- [ ] **Step 2: live 테스트 (기본 skip)**

`tests/embedders/test_live.py`:
```python
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
```

- [ ] **Step 3: 실패 확인**

Run: `uv run pytest tests/embedders -q`
Expected: FAIL — `ImportError` (모듈 없음)

- [ ] **Step 4: 구현**

`src/chunklab/core/embedders/openai.py`:
```python
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from chunklab.core.embedders.base import l2_normalize


class MissingApiKeyError(RuntimeError):
    pass


@dataclass
class OpenAIEmbedder:
    model: str = "text-embedding-3-small"
    batch_size: int = 100
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"openai:{self.model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not os.environ.get("OPENAI_API_KEY"):
            raise MissingApiKeyError("OPENAI_API_KEY is not set")
        from openai import OpenAI  # lazy import

        client = OpenAI()
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = client.embeddings.create(model=self.model, input=texts[i : i + self.batch_size])
            rows.extend(item.embedding for item in resp.data)
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return l2_normalize(np.asarray(rows, dtype=np.float32))
```

`src/chunklab/core/embedders/gemini.py`:
```python
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from chunklab.core.embedders.base import l2_normalize
from chunklab.core.embedders.openai import MissingApiKeyError


@dataclass
class GeminiEmbedder:
    model: str = "gemini-embedding-001"
    batch_size: int = 100
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"gemini:{self.model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise MissingApiKeyError("GEMINI_API_KEY is not set")
        from google import genai  # lazy import

        client = genai.Client()
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = client.models.embed_content(
                model=self.model, contents=texts[i : i + self.batch_size]
            )
            rows.extend(e.values for e in resp.embeddings)
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return l2_normalize(np.asarray(rows, dtype=np.float32))
```

구현 시 확인: `uv run python -c "from google import genai; help(genai.Client().models.embed_content)"`로 `contents` 파라미터명과 `resp.embeddings[i].values` 속성을 실제 설치된 SDK 버전에서 확인. 다르면 이 파일만 수정.

`src/chunklab/core/embedders/local.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chunklab.core.embedders.base import l2_normalize


@dataclass
class LocalEmbedder:
    model: str = "all-MiniLM-L6-v2"
    name: str = field(init=False)
    _model: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = f"local:{self.model}"

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # lazy import
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Install local embedding support with: pip install 'chunklab[local]'"
                ) from e
            self._model = SentenceTransformer(self.model)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        if not texts:
            dim = model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype=np.float32)
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return l2_normalize(np.asarray(vecs, dtype=np.float32))
```

`src/chunklab/core/embedders/__init__.py`:
```python
from __future__ import annotations

from chunklab.core.embedders.base import Embedder, FakeEmbedder
from chunklab.core.embedders.cache import CachedEmbedder, EmbeddingCache
from chunklab.core.embedders.gemini import GeminiEmbedder
from chunklab.core.embedders.local import LocalEmbedder
from chunklab.core.embedders.openai import MissingApiKeyError, OpenAIEmbedder

_PROVIDERS = {
    "fake": lambda model: FakeEmbedder(),
    "openai": lambda model: OpenAIEmbedder(model) if model else OpenAIEmbedder(),
    "gemini": lambda model: GeminiEmbedder(model) if model else GeminiEmbedder(),
    "local": lambda model: LocalEmbedder(model) if model else LocalEmbedder(),
}


def build_embedder(spec: str) -> Embedder:
    provider, _, model = spec.partition(":")
    try:
        factory = _PROVIDERS[provider]
    except KeyError:
        raise KeyError(
            f"unknown embedder provider '{provider}' in '{spec}'; available: {sorted(_PROVIDERS)}"
        ) from None
    return factory(model)


__all__ = [
    "CachedEmbedder",
    "Embedder",
    "EmbeddingCache",
    "FakeEmbedder",
    "GeminiEmbedder",
    "LocalEmbedder",
    "MissingApiKeyError",
    "OpenAIEmbedder",
    "build_embedder",
]
```

- [ ] **Step 5: 통과 확인**

Run: `uv run pytest tests/embedders -q`
Expected: `16 passed, 2 skipped`

- [ ] **Step 6: Commit**

```bash
git add src/chunklab/core/embedders tests/embedders
git commit -m "feat(embedders): OpenAI, Gemini, local embedders with spec registry"
```

---

### Task 10: 검색 (dense / bm25 / hybrid)

**Files:**
- Create: `src/chunklab/core/retrieval/__init__.py`, `dense.py`, `bm25.py`, `hybrid.py`
- Test: `tests/retrieval/__init__.py`(빈), `tests/retrieval/test_dense.py`, `test_bm25.py`, `test_hybrid.py`

**Interfaces:**
- Consumes: `Chunk`, `RetrievedChunk`, `Embedder`
- Produces:
  - `Retriever` Protocol: `search(query: str, k: int) -> list[RetrievedChunk]` (rank 1-based, score 내림차순)
  - `DenseIndex(chunks: list[Chunk], embeddings: np.ndarray, embedder: Embedder)` — 코사인(정규화 내적)
  - `BM25Index(chunks: list[Chunk])` — rank-bm25 `BM25Okapi`, 토크나이저 `tokenize(text) -> list[str]` (`\w+` 소문자)
  - `HybridIndex(dense: Retriever, sparse: Retriever, rrf_k: int = 60)` — 각에서 `2k` 후보, RRF `score = Σ 1/(rrf_k + rank)`
  - `__init__.py` re-export + `build_index(chunks, embeddings, embedder, hybrid: bool) -> Retriever`

- [ ] **Step 1: 테스트 작성**

`tests/retrieval/test_dense.py`:
```python
from chunklab.core.embedders.base import FakeEmbedder
from chunklab.core.models import Chunk
from chunklab.core.retrieval.dense import DenseIndex

CHUNKS = [
    Chunk("d", 0, 10, "refunds are issued within 30 days"),
    Chunk("d", 10, 20, "orders ship within 2 business days"),
    Chunk("d", 20, 30, "contact support with your order number"),
]


def test_dense_ranks_lexically_similar_chunk_first():
    emb = FakeEmbedder()
    idx = DenseIndex(CHUNKS, emb.embed([c.text for c in CHUNKS]), emb)
    res = idx.search("how long do refunds take, 30 days?", k=2)
    assert len(res) == 2
    assert res[0].chunk == CHUNKS[0]
    assert res[0].rank == 1 and res[1].rank == 2
    assert res[0].score >= res[1].score


def test_dense_k_larger_than_corpus():
    emb = FakeEmbedder()
    idx = DenseIndex(CHUNKS, emb.embed([c.text for c in CHUNKS]), emb)
    assert len(idx.search("anything", k=10)) == 3
```

`tests/retrieval/test_bm25.py`:
```python
from chunklab.core.models import Chunk
from chunklab.core.retrieval.bm25 import BM25Index, tokenize

CHUNKS = [
    Chunk("d", 0, 10, "Refunds are issued within 30 days."),
    Chunk("d", 10, 20, "Orders ship within 2 business days."),
    Chunk("d", 20, 30, "Contact support with your order number."),
]


def test_tokenize_lowercases_and_splits():
    assert tokenize("Refunds, within 30 Days!") == ["refunds", "within", "30", "days"]


def test_bm25_exact_term_match_ranks_first():
    idx = BM25Index(CHUNKS)
    res = idx.search("support order number", k=3)
    assert res[0].chunk == CHUNKS[2]
    assert [r.rank for r in res] == [1, 2, 3]


def test_bm25_returns_at_most_k():
    assert len(BM25Index(CHUNKS).search("days", k=1)) == 1
```

`tests/retrieval/test_hybrid.py`:
```python
from chunklab.core.models import Chunk, RetrievedChunk
from chunklab.core.retrieval.hybrid import HybridIndex, rrf_fuse

A = Chunk("d", 0, 1, "a")
B = Chunk("d", 1, 2, "b")
C = Chunk("d", 2, 3, "c")


class _Fixed:
    def __init__(self, order):
        self.order = order

    def search(self, query, k):
        return [RetrievedChunk(c, 1.0 / (i + 1), i + 1) for i, c in enumerate(self.order[:k])]


def test_rrf_fuse_prefers_chunk_ranked_well_in_both():
    fused = rrf_fuse([[A, B, C], [B, A, C]], rrf_k=60)
    # A: 1/61 + 1/62 ; B: 1/62 + 1/61 -> tie broken by first appearance (A first)
    assert [c for c, _ in fused][:2] == [A, B]
    assert fused[0][1] == fused[1][1]


def test_hybrid_search_ranks_and_truncates():
    idx = HybridIndex(_Fixed([A, B, C]), _Fixed([C, A, B]))
    res = idx.search("q", k=2)
    assert len(res) == 2
    assert res[0].chunk == A  # ranks 1 and 2
    assert [r.rank for r in res] == [1, 2]
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/retrieval -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 구현**

`src/chunklab/core/retrieval/dense.py`:
```python
from __future__ import annotations

import numpy as np

from chunklab.core.embedders.base import Embedder, l2_normalize
from chunklab.core.models import Chunk, RetrievedChunk


class DenseIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder: Embedder) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings length mismatch")
        self._chunks = chunks
        self._matrix = l2_normalize(embeddings)
        self._embedder = embedder

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        q = l2_normalize(self._embedder.embed([query]))[0]
        scores = self._matrix @ q
        k = min(k, len(self._chunks))
        top = np.argsort(-scores, kind="stable")[:k]
        return [
            RetrievedChunk(self._chunks[i], float(scores[i]), rank)
            for rank, i in enumerate(top, start=1)
        ]
```

`src/chunklab/core/retrieval/bm25.py`:
```python
from __future__ import annotations

import re

import numpy as np

from chunklab.core.models import Chunk, RetrievedChunk

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        if self._bm25 is None:
            return []
        scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float64)
        k = min(k, len(self._chunks))
        top = np.argsort(-scores, kind="stable")[:k]
        return [
            RetrievedChunk(self._chunks[i], float(scores[i]), rank)
            for rank, i in enumerate(top, start=1)
        ]
```

`src/chunklab/core/retrieval/hybrid.py`:
```python
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from chunklab.core.models import Chunk, RetrievedChunk


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[RetrievedChunk]: ...


def rrf_fuse(rankings: Sequence[Sequence[Chunk]], rrf_k: int = 60) -> list[tuple[Chunk, float]]:
    scores: dict[Chunk, float] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk] = scores.get(chunk, 0.0) + 1.0 / (rrf_k + rank)
    # dict preserves first-insertion order; stable sort keeps it for ties
    return sorted(scores.items(), key=lambda kv: -kv[1])


class HybridIndex:
    def __init__(self, dense: Retriever, sparse: Retriever, rrf_k: int = 60) -> None:
        self._dense = dense
        self._sparse = sparse
        self._rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        cand = 2 * k
        dense = [r.chunk for r in self._dense.search(query, cand)]
        sparse = [r.chunk for r in self._sparse.search(query, cand)]
        fused = rrf_fuse([dense, sparse], self._rrf_k)[:k]
        return [RetrievedChunk(c, s, rank) for rank, (c, s) in enumerate(fused, start=1)]
```

`src/chunklab/core/retrieval/__init__.py`:
```python
from __future__ import annotations

import numpy as np

from chunklab.core.embedders.base import Embedder
from chunklab.core.models import Chunk
from chunklab.core.retrieval.bm25 import BM25Index, tokenize
from chunklab.core.retrieval.dense import DenseIndex
from chunklab.core.retrieval.hybrid import HybridIndex, Retriever, rrf_fuse


def build_index(
    chunks: list[Chunk], embeddings: np.ndarray, embedder: Embedder, hybrid: bool
) -> Retriever:
    dense = DenseIndex(chunks, embeddings, embedder)
    if not hybrid:
        return dense
    return HybridIndex(dense, BM25Index(chunks))


__all__ = [
    "BM25Index",
    "DenseIndex",
    "HybridIndex",
    "Retriever",
    "build_index",
    "rrf_fuse",
    "tokenize",
]
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/retrieval -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/chunklab/core/retrieval tests/retrieval
git commit -m "feat(retrieval): dense, BM25 and RRF hybrid indexes"
```

---

### Task 11: 테스트 질문 생성 (LLM 추상화 + 합성 + JSON I/O)

**Files:**
- Create: `src/chunklab/core/questions/__init__.py`, `llm.py`, `generate.py`, `io.py`
- Test: `tests/questions/__init__.py`(빈), `tests/questions/test_llm.py`, `test_generate.py`, `test_io.py`

**Interfaces:**
- Produces (llm.py):
  - `LLM` Protocol: `complete(prompt: str) -> str`
  - `FakeLLM(reply: str = "What is described in this passage?")`; `.prompts: list[str]` 기록
  - `OpenAIChat(model: str = "gpt-4o-mini")`, `GeminiChat(model: str = "gemini-2.5-flash")` — 키 없으면 `MissingApiKeyError`
  - `build_llm(spec: str) -> LLM` — `"fake" | "openai[:model]" | "gemini[:model]"`
- Produces (generate.py):
  - `sample_passages(doc: Document, n: int, seed: int = 0, min_len: int = 200, max_len: int = 600) -> list[Span]` — 빈 줄 기준 문단 중 `len >= min_len`을 셔플·선택, `max_len` 초과 시 앞에서 자름. 후보 부족 시 있는 만큼 반환
  - `generate_questions(docs: Sequence[Document], llm: LLM, per_doc: int, seed: int = 0) -> list[Question]` — id `f"{doc.id}-q{i}"`, 응답 첫 줄만 사용, 빈 응답은 스킵
  - `QUESTION_PROMPT: str`
- Produces (io.py):
  - `save_questions(questions: Sequence[Question], path: Path) -> None`
  - `load_questions(path: Path) -> list[Question]`
  - JSON 형식: `{"version": 1, "questions": [{"id","text","spans":[{"doc_id","start","end"}]}]}`

- [ ] **Step 1: 테스트 작성**

`tests/questions/test_llm.py`:
```python
import pytest

from chunklab.core.embedders import MissingApiKeyError
from chunklab.core.questions.llm import FakeLLM, GeminiChat, OpenAIChat, build_llm


def test_fake_llm_records_prompts():
    llm = FakeLLM(reply="Q?")
    assert llm.complete("p1") == "Q?"
    assert llm.prompts == ["p1"]


def test_build_llm_specs():
    assert isinstance(build_llm("fake"), FakeLLM)
    assert isinstance(build_llm("openai"), OpenAIChat)
    assert build_llm("openai:gpt-4o").model == "gpt-4o"  # type: ignore[attr-defined]
    assert isinstance(build_llm("gemini"), GeminiChat)
    with pytest.raises(KeyError, match="unknown llm provider"):
        build_llm("anthropic")


def test_missing_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        OpenAIChat().complete("x")
    with pytest.raises(MissingApiKeyError):
        GeminiChat().complete("x")
```

`tests/questions/test_generate.py`:
```python
from pathlib import Path

from chunklab.core.models import Document
from chunklab.core.questions.generate import QUESTION_PROMPT, generate_questions, sample_passages
from chunklab.core.questions.llm import FakeLLM
from chunklab.core.text import load_document


def _doc_with_paragraphs(n: int, size: int) -> Document:
    paras = [f"Paragraph {i}. " + ("word " * (size // 5)) for i in range(n)]
    return Document("d", "\n\n".join(paras) + "\n")


def test_sample_passages_respects_min_len_and_n():
    doc = _doc_with_paragraphs(10, 300)
    spans = sample_passages(doc, n=3, seed=1, min_len=200, max_len=600)
    assert len(spans) == 3
    for s in spans:
        assert s.doc_id == "d"
        assert 200 <= s.length <= 600
        assert doc.text[s.start : s.end].startswith("Paragraph")


def test_sample_passages_deterministic_by_seed():
    doc = _doc_with_paragraphs(10, 300)
    assert sample_passages(doc, 4, seed=7) == sample_passages(doc, 4, seed=7)
    assert sample_passages(doc, 4, seed=7) != sample_passages(doc, 4, seed=8)


def test_sample_passages_truncates_to_max_len():
    doc = _doc_with_paragraphs(2, 2000)
    spans = sample_passages(doc, 2, min_len=100, max_len=500)
    assert all(s.length == 500 for s in spans)


def test_sample_passages_returns_fewer_when_short_doc():
    doc = Document("d", "tiny\n\nalso tiny\n")
    assert sample_passages(doc, 5, min_len=200) == []


def test_generate_questions_builds_questions_with_spans(sample_doc_path: Path):
    doc = load_document(sample_doc_path)
    llm = FakeLLM(reply="What is the refund window?\nignored second line")
    qs = generate_questions([doc], llm, per_doc=2, seed=0, min_len=50)
    assert len(qs) == 2
    assert qs[0].id == "sample-q0"
    assert qs[0].text == "What is the refund window?"
    assert len(qs[0].spans) == 1
    assert qs[0].spans[0].doc_id == "sample"
    assert len(llm.prompts) == 2
    assert QUESTION_PROMPT.split("{passage}")[0] in llm.prompts[0]
    passage = doc.text[qs[0].spans[0].start : qs[0].spans[0].end]
    assert passage in llm.prompts[0]


def test_generate_questions_skips_empty_reply():
    doc = _doc_with_paragraphs(3, 300)
    qs = generate_questions([doc], FakeLLM(reply="   \n"), per_doc=3)
    assert qs == []
```

`tests/questions/test_io.py`:
```python
import json
from pathlib import Path

from chunklab.core.models import Question, Span
from chunklab.core.questions.io import load_questions, save_questions


def test_roundtrip(tmp_path: Path):
    qs = [
        Question("q1", "what?", (Span("d", 0, 10),)),
        Question("q2", "why?", (Span("d", 5, 15), Span("e", 0, 3))),
    ]
    p = tmp_path / "q.json"
    save_questions(qs, p)
    data = json.loads(p.read_text())
    assert data["version"] == 1
    assert data["questions"][1]["spans"][1] == {"doc_id": "e", "start": 0, "end": 3}
    assert load_questions(p) == qs
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/questions -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 구현**

`src/chunklab/core/questions/llm.py`:
```python
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from chunklab.core.embedders.openai import MissingApiKeyError


@runtime_checkable
class LLM(Protocol):
    def complete(self, prompt: str) -> str: ...


@dataclass
class FakeLLM:
    reply: str = "What is described in this passage?"
    prompts: list[str] = field(default_factory=list)

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


@dataclass
class OpenAIChat:
    model: str = "gpt-4o-mini"

    def complete(self, prompt: str) -> str:
        if not os.environ.get("OPENAI_API_KEY"):
            raise MissingApiKeyError("OPENAI_API_KEY is not set")
        from openai import OpenAI  # lazy import

        resp = OpenAI().chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content or ""


@dataclass
class GeminiChat:
    model: str = "gemini-2.5-flash"

    def complete(self, prompt: str) -> str:
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise MissingApiKeyError("GEMINI_API_KEY is not set")
        from google import genai  # lazy import

        resp = genai.Client().models.generate_content(model=self.model, contents=prompt)
        return resp.text or ""


_PROVIDERS = {
    "fake": lambda model: FakeLLM(),
    "openai": lambda model: OpenAIChat(model) if model else OpenAIChat(),
    "gemini": lambda model: GeminiChat(model) if model else GeminiChat(),
}


def build_llm(spec: str) -> LLM:
    provider, _, model = spec.partition(":")
    try:
        factory = _PROVIDERS[provider]
    except KeyError:
        raise KeyError(
            f"unknown llm provider '{provider}' in '{spec}'; available: {sorted(_PROVIDERS)}"
        ) from None
    return factory(model)
```

`src/chunklab/core/questions/generate.py`:
```python
from __future__ import annotations

import random
import re
from collections.abc import Sequence

from chunklab.core.models import Document, Question, Span
from chunklab.core.questions.llm import LLM

QUESTION_PROMPT = (
    "You are creating retrieval test data. Write ONE question that can be answered ONLY "
    "using the passage below. The question must be specific to the passage's content and "
    "must not mention 'the passage'. Output only the question, nothing else.\n\n"
    "Passage:\n\"\"\"\n{passage}\n\"\"\""
)

_PARA_RE = re.compile(r"\n{2,}")


def _paragraph_spans(doc: Document) -> list[Span]:
    spans: list[Span] = []
    pos = 0
    text = doc.text
    for m in _PARA_RE.finditer(text):
        if m.start() > pos:
            spans.append(Span(doc.id, pos, m.start()))
        pos = m.end()
    if pos < len(text):
        end = len(text.rstrip("\n"))
        if end > pos:
            spans.append(Span(doc.id, pos, end))
    return spans


def sample_passages(
    doc: Document, n: int, seed: int = 0, min_len: int = 200, max_len: int = 600
) -> list[Span]:
    candidates = [s for s in _paragraph_spans(doc) if s.length >= min_len]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen = sorted(candidates[:n], key=lambda s: s.start)
    return [
        Span(s.doc_id, s.start, min(s.end, s.start + max_len)) for s in chosen
    ]


def generate_questions(
    docs: Sequence[Document],
    llm: LLM,
    per_doc: int,
    seed: int = 0,
    min_len: int = 200,
    max_len: int = 600,
) -> list[Question]:
    out: list[Question] = []
    for doc in docs:
        spans = sample_passages(doc, per_doc, seed=seed, min_len=min_len, max_len=max_len)
        for i, span in enumerate(spans):
            passage = doc.text[span.start : span.end]
            reply = llm.complete(QUESTION_PROMPT.format(passage=passage))
            first_line = next((ln.strip() for ln in reply.splitlines() if ln.strip()), "")
            if not first_line:
                continue
            out.append(Question(f"{doc.id}-q{i}", first_line, (span,)))
    return out
```

`src/chunklab/core/questions/io.py`:
```python
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from chunklab.core.models import Question, Span

FORMAT_VERSION = 1


def save_questions(questions: Sequence[Question], path: Path) -> None:
    payload = {
        "version": FORMAT_VERSION,
        "questions": [
            {
                "id": q.id,
                "text": q.text,
                "spans": [{"doc_id": s.doc_id, "start": s.start, "end": s.end} for s in q.spans],
            }
            for q in questions
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_questions(path: Path) -> list[Question]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported questions file version: {data.get('version')}")
    return [
        Question(
            q["id"],
            q["text"],
            tuple(Span(s["doc_id"], s["start"], s["end"]) for s in q["spans"]),
        )
        for q in data["questions"]
    ]
```

`src/chunklab/core/questions/__init__.py`:
```python
from chunklab.core.questions.generate import QUESTION_PROMPT, generate_questions, sample_passages
from chunklab.core.questions.io import load_questions, save_questions
from chunklab.core.questions.llm import LLM, FakeLLM, GeminiChat, OpenAIChat, build_llm

__all__ = [
    "LLM",
    "QUESTION_PROMPT",
    "FakeLLM",
    "GeminiChat",
    "OpenAIChat",
    "build_llm",
    "generate_questions",
    "load_questions",
    "sample_passages",
    "save_questions",
]
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/questions -q`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add src/chunklab/core/questions tests/questions
git commit -m "feat(questions): LLM abstraction, synthetic question generation, JSON I/O"
```

---

### Task 12: 실험 설정 (YAML) + 매트릭스 확장

**Files:**
- Create: `src/chunklab/core/runner/__init__.py`(빈, Task 14에서 채움), `src/chunklab/core/runner/config.py`
- Test: `tests/runner/__init__.py`(빈), `tests/runner/test_config.py`

**Interfaces:**
- Produces:
  - `ChunkerGrid(BaseModel)`: `name: str`, `params: dict[str, list[Any]] = {}`
  - `RetrievalGrid(BaseModel)`: `top_k: list[int] = [5]`, `hybrid: list[bool] = [False]`
  - `ExperimentConfig(BaseModel)`: `documents: list[str]`(glob), `questions: str`, `chunkers: list[ChunkerGrid]`, `embedders: list[str]`, `retrieval: RetrievalGrid`, `hit_threshold: float = 0.5`, `cache_path: str = "~/.chunklab/cache.db"`; `@classmethod from_yaml(path: Path) -> ExperimentConfig`; `resolve_documents(base_dir: Path) -> list[Path]` (정렬, 중복 제거)
  - `Combo` frozen dataclass: `chunker: str`, `chunker_params: tuple[tuple[str, Any], ...]`(키 정렬), `embedder: str`, `top_k: int`, `hybrid: bool`; `.id -> str` 예: `recursive(chunk_size=256,overlap=0)|fake|k=5|hybrid=False`; `.params_dict() -> dict`
  - `expand_matrix(cfg: ExperimentConfig) -> list[Combo]` — 데카르트 곱, 결정적 순서

- [ ] **Step 1: 테스트 작성**

`tests/runner/test_config.py`:
```python
from pathlib import Path

import pytest

from chunklab.core.runner.config import Combo, ExperimentConfig, expand_matrix

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
    assert combos[0] == Combo(
        "recursive", (("chunk_size", 256), ("overlap", 0)), "fake", 5, False
    )
    assert len({c.id for c in combos}) == len(combos)


def test_combo_id_and_params_dict():
    c = Combo("recursive", (("chunk_size", 256), ("overlap", 0)), "fake", 5, True)
    assert c.id == "recursive(chunk_size=256,overlap=0)|fake|k=5|hybrid=True"
    assert c.params_dict() == {"chunk_size": 256, "overlap": 0}


def test_chunker_without_params_yields_one_variant():
    cfg = ExperimentConfig(
        documents=["x.md"], questions="q.json",
        chunkers=[{"name": "markdown"}], embedders=["fake"],
    )
    combos = expand_matrix(cfg)
    assert len(combos) == 1
    assert combos[0].id == "markdown()|fake|k=5|hybrid=False"


def test_resolve_documents_globs_relative_to_base(tmp_path: Path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "b.md").write_text("b")
    (tmp_path / "docs" / "a.md").write_text("a")
    cfg = ExperimentConfig(documents=["docs/*.md", "docs/a.md"], questions="q", chunkers=[], embedders=[])
    paths = cfg.resolve_documents(tmp_path)
    assert [p.name for p in paths] == ["a.md", "b.md"]


def test_invalid_yaml_missing_required(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("documents: []\n")
    with pytest.raises(ValueError):
        ExperimentConfig.from_yaml(p)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/runner/test_config.py -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 구현**

`src/chunklab/core/runner/config.py`:
```python
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ChunkerGrid(BaseModel):
    name: str
    params: dict[str, list[Any]] = Field(default_factory=dict)


class RetrievalGrid(BaseModel):
    top_k: list[int] = Field(default_factory=lambda: [5])
    hybrid: list[bool] = Field(default_factory=lambda: [False])


class ExperimentConfig(BaseModel):
    documents: list[str]
    questions: str
    chunkers: list[ChunkerGrid]
    embedders: list[str]
    retrieval: RetrievalGrid = Field(default_factory=RetrievalGrid)
    hit_threshold: float = 0.5
    cache_path: str = "~/.chunklab/cache.db"

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def resolve_documents(self, base_dir: Path) -> list[Path]:
        found: set[Path] = set()
        for pattern in self.documents:
            for p in Path(base_dir).glob(pattern):
                if p.is_file():
                    found.add(p.resolve())
        return sorted(found)


@dataclass(frozen=True)
class Combo:
    chunker: str
    chunker_params: tuple[tuple[str, Any], ...]
    embedder: str
    top_k: int
    hybrid: bool

    @property
    def id(self) -> str:
        params = ",".join(f"{k}={v}" for k, v in self.chunker_params)
        return f"{self.chunker}({params})|{self.embedder}|k={self.top_k}|hybrid={self.hybrid}"

    def params_dict(self) -> dict[str, Any]:
        return dict(self.chunker_params)


def _chunker_variants(grid: ChunkerGrid) -> list[tuple[tuple[str, Any], ...]]:
    if not grid.params:
        return [()]
    keys = sorted(grid.params)
    return [
        tuple(zip(keys, values, strict=True))
        for values in itertools.product(*(grid.params[k] for k in keys))
    ]


def expand_matrix(cfg: ExperimentConfig) -> list[Combo]:
    combos: list[Combo] = []
    for grid in cfg.chunkers:
        for params in _chunker_variants(grid):
            for embedder in cfg.embedders:
                for top_k in cfg.retrieval.top_k:
                    for hybrid in cfg.retrieval.hybrid:
                        combos.append(Combo(grid.name, params, embedder, top_k, hybrid))
    return combos
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/runner/test_config.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/chunklab/core/runner tests/runner
git commit -m "feat(runner): YAML experiment config and combo matrix expansion"
```

---

### Task 13: 실험 실행 (run.py)

**Files:**
- Create: `src/chunklab/core/runner/run.py`
- Test: `tests/runner/test_run.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `Combo`, `expand_matrix`, `load_documents`, `load_questions`, `build_chunker`, `build_embedder`, `EmbeddingCache`, `CachedEmbedder`, `build_index`, `evaluate`, `mean_metrics`
- Produces:
  - `ComboResult` dataclass: `combo_id: str`, `chunker: str`, `chunker_params: dict`, `embedder: str`, `top_k: int`, `hybrid: bool`, `metrics: dict[str, float]`, `n_chunks: int`, `per_question: list[dict]`, `error: str | None`
  - `RunResult` dataclass: `run_id: str`, `created_at: str`(ISO 8601 UTC), `config: dict`, `combos: list[ComboResult]`; `.to_json(path)`, `.from_json(path)`, `.to_dict()`
  - `run_experiment(cfg: ExperimentConfig, base_dir: Path, embedder_factory: Callable[[str], Embedder] = build_embedder, progress: Callable[[str], None] | None = None) -> RunResult`
  - `validate_questions(questions, docs) -> None` — 미지 doc_id 또는 범위 초과 span이면 `ValueError`
  - per_question 항목: `{"id", "hit", "reciprocal_rank", "ndcg", "precision", "iou", "retrieved": [{"doc_id","start","end","score"}]}`

- [ ] **Step 1: 테스트 작성**

`tests/runner/test_run.py`:
```python
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
            Question("q-refund", "how many days for a refund?", (Span("sample", refund, refund_end),)),
            Question("q-lost", "package marked delivered but not received", (Span("sample", lost, lost_end),)),
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
cache_path: "{tmp_path / 'cache.db'}"
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
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/runner/test_run.py -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 구현**

`src/chunklab/core/runner/run.py`:
```python
from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chunklab.core.chunkers import build_chunker
from chunklab.core.embedders import CachedEmbedder, EmbeddingCache, build_embedder
from chunklab.core.embedders.base import Embedder
from chunklab.core.metrics import evaluate, mean_metrics
from chunklab.core.models import Chunk, Document, Question
from chunklab.core.questions.io import load_questions
from chunklab.core.retrieval import build_index
from chunklab.core.runner.config import Combo, ExperimentConfig, expand_matrix
from chunklab.core.text import load_documents


@dataclass
class ComboResult:
    combo_id: str
    chunker: str
    chunker_params: dict[str, Any]
    embedder: str
    top_k: int
    hybrid: bool
    metrics: dict[str, float] = field(default_factory=dict)
    n_chunks: int = 0
    per_question: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass
class RunResult:
    run_id: str
    created_at: str
    config: dict[str, Any]
    combos: list[ComboResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "config": self.config,
            "combos": [asdict(c) for c in self.combos],
        }

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> RunResult:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            run_id=data["run_id"],
            created_at=data["created_at"],
            config=data["config"],
            combos=[ComboResult(**c) for c in data["combos"]],
        )


def validate_questions(questions: Sequence[Question], docs: Sequence[Document]) -> None:
    lengths = {d.id: len(d.text) for d in docs}
    for q in questions:
        for s in q.spans:
            if s.doc_id not in lengths:
                raise ValueError(f"question '{q.id}' references unknown doc_id '{s.doc_id}'")
            if s.end > lengths[s.doc_id]:
                raise ValueError(
                    f"question '{q.id}' span {s.start}:{s.end} out of range "
                    f"for doc '{s.doc_id}' (len={lengths[s.doc_id]})"
                )


def _chunk_all(combo: Combo, docs: Sequence[Document]) -> list[Chunk]:
    chunker = build_chunker(combo.chunker, **combo.params_dict())
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(chunker.chunk(d))
    return chunks


def _run_combo(
    combo: Combo,
    chunks: list[Chunk],
    questions: Sequence[Question],
    embedder: Embedder,
    hit_threshold: float,
) -> ComboResult:
    result = ComboResult(
        combo.id, combo.chunker, combo.params_dict(), combo.embedder, combo.top_k, combo.hybrid
    )
    if not chunks:
        raise ValueError("chunker produced no chunks")
    embeddings = embedder.embed([c.text for c in chunks])
    index = build_index(chunks, embeddings, embedder, combo.hybrid)
    per_q_metrics = []
    for q in questions:
        retrieved = index.search(q.text, combo.top_k)
        m = evaluate([r.chunk for r in retrieved], q.spans, combo.top_k, hit_threshold)
        per_q_metrics.append(m)
        result.per_question.append(
            {
                "id": q.id,
                "hit": m.hit,
                "reciprocal_rank": m.reciprocal_rank,
                "ndcg": m.ndcg,
                "precision": m.precision,
                "iou": m.iou,
                "retrieved": [
                    {
                        "doc_id": r.chunk.doc_id,
                        "start": r.chunk.start,
                        "end": r.chunk.end,
                        "score": r.score,
                    }
                    for r in retrieved
                ],
            }
        )
    result.metrics = mean_metrics(per_q_metrics, combo.top_k)
    result.n_chunks = len(chunks)
    return result


def run_experiment(
    cfg: ExperimentConfig,
    base_dir: Path,
    embedder_factory: Callable[[str], Embedder] = build_embedder,
    progress: Callable[[str], None] | None = None,
) -> RunResult:
    base_dir = Path(base_dir)
    docs = load_documents(cfg.resolve_documents(base_dir))
    if not docs:
        raise ValueError(f"no documents matched {cfg.documents}")
    questions = load_questions(base_dir / cfg.questions)
    validate_questions(questions, docs)

    results: list[ComboResult] = []
    chunk_cache: dict[tuple[str, tuple], list[Chunk]] = {}
    with EmbeddingCache(Path(cfg.cache_path).expanduser()) as cache:
        embedders = {spec: CachedEmbedder(embedder_factory(spec), cache) for spec in cfg.embedders}
        for combo in expand_matrix(cfg):
            try:
                key = (combo.chunker, combo.chunker_params)
                if key not in chunk_cache:
                    chunk_cache[key] = _chunk_all(combo, docs)
                results.append(
                    _run_combo(
                        combo, chunk_cache[key], questions, embedders[combo.embedder],
                        cfg.hit_threshold,
                    )
                )
            except Exception as e:  # noqa: BLE001 - partial failure is a feature
                results.append(
                    ComboResult(
                        combo.id, combo.chunker, combo.params_dict(), combo.embedder,
                        combo.top_k, combo.hybrid, error=f"{type(e).__name__}: {e}",
                    )
                )
            if progress:
                progress(combo.id)

    return RunResult(
        run_id=uuid.uuid4().hex[:12],
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        config=cfg.model_dump(),
        combos=results,
    )
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/runner/test_run.py -q`
Expected: `6 passed`

`test_run_experiment_finds_refund_span_with_fake_embedder`가 실패하면 fake 임베더의 어휘 겹침 특성상 질문 문구를 span 텍스트 단어("refund", "30 days")와 더 겹치게 조정한다. 엔진 로직은 바꾸지 않는다.

- [ ] **Step 5: Commit**

```bash
git add src/chunklab/core/runner/run.py tests/runner/test_run.py
git commit -m "feat(runner): run experiment matrix with per-combo isolation and JSON results"
```

---

### Task 14: 회귀 비교 (compare.py) + runner 패키지 export

**Files:**
- Create: `src/chunklab/core/runner/compare.py`
- Modify: `src/chunklab/core/runner/__init__.py`
- Test: `tests/runner/test_compare.py`

**Interfaces:**
- Consumes: `RunResult`, `ComboResult`
- Produces:
  - `parse_thresholds(exprs: Sequence[str]) -> dict[str, float]` — `"hit@5=0.8"` 형식, 잘못되면 `ValueError`
  - `check_thresholds(result: RunResult, thresholds: dict[str, float]) -> list[str]` — 위반 메시지 목록. 에러난 조합은 `"<id>: failed (<error>)"`로 위반 처리. 메트릭 키가 없는 조합(예: top_k 다름)은 스킵
  - `check_regression(result: RunResult, baseline: RunResult, max_drop: float) -> list[str]` — combo_id로 매칭, 같은 메트릭 키에 대해 `current < baseline - max_drop`면 위반. baseline에 없는 조합은 스킵
  - `format_table(result: RunResult) -> str` — 조합별 메트릭 고정폭 표 (에러 조합은 ERROR 표기)
  - `runner/__init__.py`가 `ExperimentConfig, Combo, expand_matrix, RunResult, ComboResult, run_experiment, parse_thresholds, check_thresholds, check_regression, format_table` export

- [ ] **Step 1: 테스트 작성**

`tests/runner/test_compare.py`:
```python
import pytest

from chunklab.core.runner import (
    ComboResult,
    RunResult,
    check_regression,
    check_thresholds,
    format_table,
    parse_thresholds,
)


def _combo(cid: str, **metrics) -> ComboResult:
    return ComboResult(cid, "recursive", {}, "fake", 5, False, metrics=dict(metrics), n_chunks=3)


def _result(*combos: ComboResult) -> RunResult:
    return RunResult("r", "2026-01-01T00:00:00+00:00", {}, list(combos))


def test_parse_thresholds():
    assert parse_thresholds(["hit@5=0.8", "mrr=0.5"]) == {"hit@5": 0.8, "mrr": 0.5}
    with pytest.raises(ValueError):
        parse_thresholds(["hit@5"])
    with pytest.raises(ValueError):
        parse_thresholds(["hit@5=high"])


def test_check_thresholds_reports_violations_and_errors():
    res = _result(
        _combo("a", **{"hit@5": 0.9, "mrr": 0.7}),
        _combo("b", **{"hit@5": 0.5, "mrr": 0.7}),
        ComboResult("c", "recursive", {}, "fake", 5, False, error="RuntimeError: x"),
    )
    v = check_thresholds(res, {"hit@5": 0.8})
    assert v == ["b: hit@5 0.500 < 0.800", "c: failed (RuntimeError: x)"]


def test_check_thresholds_skips_missing_metric_keys():
    res = _result(_combo("a", **{"hit@3": 0.1}))
    assert check_thresholds(res, {"hit@5": 0.8}) == []


def test_check_regression_flags_drops_beyond_max_drop():
    base = _result(_combo("a", **{"hit@5": 0.9, "iou": 0.5}), _combo("b", **{"hit@5": 0.8}))
    cur = _result(_combo("a", **{"hit@5": 0.8, "iou": 0.48}), _combo("z", **{"hit@5": 0.1}))
    v = check_regression(cur, base, max_drop=0.05)
    assert v == ["a: hit@5 dropped 0.900 -> 0.800 (max drop 0.050)"]


def test_format_table_contains_ids_metrics_and_error():
    res = _result(
        _combo("recursive(chunk_size=256)|fake|k=5|hybrid=False", **{"hit@5": 0.75, "mrr": 0.5}),
        ComboResult("bad|fake|k=5|hybrid=False", "bad", {}, "fake", 5, False, error="Boom: x"),
    )
    table = format_table(res)
    assert "recursive(chunk_size=256)|fake|k=5|hybrid=False" in table
    assert "0.750" in table
    assert "ERROR" in table
    assert "hit@5" in table.splitlines()[0]
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/runner/test_compare.py -q`
Expected: FAIL — `ImportError`

- [ ] **Step 3: 구현**

`src/chunklab/core/runner/compare.py`:
```python
from __future__ import annotations

from collections.abc import Sequence

from chunklab.core.runner.run import RunResult


def parse_thresholds(exprs: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for expr in exprs:
        metric, sep, value = expr.partition("=")
        if not sep or not metric:
            raise ValueError(f"expected METRIC=VALUE, got '{expr}'")
        try:
            out[metric.strip()] = float(value)
        except ValueError:
            raise ValueError(f"threshold value must be a number, got '{value}' in '{expr}'") from None
    return out


def check_thresholds(result: RunResult, thresholds: dict[str, float]) -> list[str]:
    violations: list[str] = []
    for c in result.combos:
        if c.error:
            violations.append(f"{c.combo_id}: failed ({c.error})")
            continue
        for metric, minimum in thresholds.items():
            if metric in c.metrics and c.metrics[metric] < minimum:
                violations.append(f"{c.combo_id}: {metric} {c.metrics[metric]:.3f} < {minimum:.3f}")
    return violations


def check_regression(result: RunResult, baseline: RunResult, max_drop: float) -> list[str]:
    base_by_id = {c.combo_id: c for c in baseline.combos if not c.error}
    violations: list[str] = []
    for c in result.combos:
        b = base_by_id.get(c.combo_id)
        if b is None or c.error:
            continue
        for metric, prev in b.metrics.items():
            cur = c.metrics.get(metric)
            if cur is not None and cur < prev - max_drop:
                violations.append(
                    f"{c.combo_id}: {metric} dropped {prev:.3f} -> {cur:.3f} (max drop {max_drop:.3f})"
                )
    return violations


def format_table(result: RunResult) -> str:
    metric_keys: list[str] = []
    for c in result.combos:
        for k in c.metrics:
            if k not in metric_keys:
                metric_keys.append(k)
    id_width = max([len("combo")] + [len(c.combo_id) for c in result.combos])
    header = f"{'combo':<{id_width}}  " + "  ".join(f"{k:>9}" for k in metric_keys)
    lines = [header, "-" * len(header)]
    for c in result.combos:
        if c.error:
            lines.append(f"{c.combo_id:<{id_width}}  ERROR: {c.error}")
            continue
        cells = "  ".join(f"{c.metrics.get(k, float('nan')):>9.3f}" for k in metric_keys)
        lines.append(f"{c.combo_id:<{id_width}}  {cells}")
    return "\n".join(lines)
```

`src/chunklab/core/runner/__init__.py`:
```python
from chunklab.core.runner.compare import (
    check_regression,
    check_thresholds,
    format_table,
    parse_thresholds,
)
from chunklab.core.runner.config import Combo, ExperimentConfig, expand_matrix
from chunklab.core.runner.run import ComboResult, RunResult, run_experiment, validate_questions

__all__ = [
    "Combo",
    "ComboResult",
    "ExperimentConfig",
    "RunResult",
    "check_regression",
    "check_thresholds",
    "expand_matrix",
    "format_table",
    "parse_thresholds",
    "run_experiment",
    "validate_questions",
]
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/runner -q`
Expected: `17 passed`

- [ ] **Step 5: Commit**

```bash
git add src/chunklab/core/runner tests/runner
git commit -m "feat(runner): threshold and baseline regression checks, result table"
```

---

### Task 15: CLI (`chunklab run`, `chunklab generate-questions`)

**Files:**
- Create: `src/chunklab/cli/__init__.py`(빈), `src/chunklab/cli/main.py`
- Test: `tests/cli/__init__.py`(빈), `tests/cli/test_main.py`

**Interfaces:**
- Consumes: runner 전체, `load_documents`, `build_llm`, `generate_questions`, `save_questions`
- Produces: typer `app`
  - `chunklab run CONFIG [--out result.json] [--baseline prev.json] [--max-drop 0.05] [--fail-below METRIC=VALUE ...] [--quiet]`
    - base_dir = CONFIG의 부모 디렉터리
    - 표 출력 → 결과 JSON 저장(기본 `result.json`, CONFIG 옆) → 위반 있으면 stderr에 나열 후 exit 1, 없으면 exit 0
  - `chunklab generate-questions PATHS... --out questions.json [--llm fake] [--per-doc 10] [--seed 0]`
    - 생성 개수 출력, exit 0
  - `chunklab version`

- [ ] **Step 1: 테스트 작성**

`tests/cli/test_main.py`:
```python
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
cache_path: "{tmp_path / 'cache.db'}"
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
            "generate-questions", str(sample_doc_path),
            "--out", str(out), "--llm", "fake", "--per-doc", "2", "--min-len", "50",
        ],
    )
    assert r.exit_code == 0, r.output
    qs = load_questions(out)
    assert len(qs) == 2
    assert all(q.spans[0].doc_id == "sample" for q in qs)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/cli -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 구현**

`src/chunklab/cli/main.py`:
```python
from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

import chunklab
from chunklab.core.questions import build_llm, generate_questions, save_questions
from chunklab.core.runner import (
    ExperimentConfig,
    RunResult,
    check_regression,
    check_thresholds,
    format_table,
    parse_thresholds,
    run_experiment,
)
from chunklab.core.text import load_documents

app = typer.Typer(help="Benchmark your chunking before you ship it.", no_args_is_help=True)


@app.command()
def version() -> None:
    typer.echo(f"chunklab {chunklab.__version__}")


@app.command()
def run(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Experiment YAML")],
    out: Annotated[Path | None, typer.Option("--out", help="Result JSON path")] = None,
    baseline: Annotated[Path | None, typer.Option("--baseline", exists=True)] = None,
    max_drop: Annotated[float, typer.Option("--max-drop")] = 0.05,
    fail_below: Annotated[
        list[str] | None, typer.Option("--fail-below", help="METRIC=VALUE, repeatable")
    ] = None,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
) -> None:
    cfg = ExperimentConfig.from_yaml(config)
    base_dir = config.resolve().parent
    out = out or base_dir / "result.json"
    thresholds = parse_thresholds(fail_below or [])

    def progress(combo_id: str) -> None:
        if not quiet:
            typer.echo(f"  done: {combo_id}", err=True)

    result = run_experiment(cfg, base_dir, progress=progress)
    result.to_json(out)
    typer.echo(format_table(result))
    typer.echo(f"\nsaved: {out}")

    violations = check_thresholds(result, thresholds)
    if baseline is not None:
        violations += check_regression(result, RunResult.from_json(baseline), max_drop)
    if violations:
        typer.echo("\nFAILED:", err=True)
        for v in violations:
            typer.echo(f"  - {v}", err=True)
        raise typer.Exit(code=1)
    typer.echo("\nOK")


@app.command("generate-questions")
def generate_questions_cmd(
    paths: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out")],
    llm: Annotated[str, typer.Option("--llm", help="fake | openai[:model] | gemini[:model]")] = "openai",
    per_doc: Annotated[int, typer.Option("--per-doc")] = 10,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    min_len: Annotated[int, typer.Option("--min-len")] = 200,
    max_len: Annotated[int, typer.Option("--max-len")] = 600,
) -> None:
    docs = load_documents(paths)
    questions = generate_questions(
        docs, build_llm(llm), per_doc=per_doc, seed=seed, min_len=min_len, max_len=max_len
    )
    save_questions(questions, out)
    typer.echo(f"generated {len(questions)} questions from {len(docs)} document(s) -> {out}")


def main() -> None:  # pragma: no cover
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/cli -q`
Expected: `5 passed`

`test_run_baseline_regression_exits_1`에서 exit 0이면 CliRunner가 stderr를 `r.output`에 합치는지 확인(typer 버전에 따라 `mix_stderr` 기본값 다름). 필요하면 `CliRunner(mix_stderr=True)` 또는 `r.stderr` 사용.

- [ ] **Step 5: 설치된 엔트리포인트 확인**

Run: `uv run chunklab version && uv run chunklab --help`
Expected: `chunklab 0.1.0`, 명령 `run`, `generate-questions`, `version` 노출

- [ ] **Step 6: Commit**

```bash
git add src/chunklab/cli tests/cli
git commit -m "feat(cli): chunklab run and generate-questions commands"
```

---

### Task 16: README + 예제 실험 + 전체 검증

**Files:**
- Create: `README.md`, `examples/quickstart/exp.yaml`, `examples/quickstart/docs/refund-policy.md`, `examples/quickstart/questions.json`
- Test: `tests/test_examples.py`

**Interfaces:**
- Consumes: CLI 전체
- Produces: 클론 직후 `uv run chunklab run examples/quickstart/exp.yaml`가 API 키 없이(fake 임베더) 동작하는 예제

- [ ] **Step 1: 예제 문서·설정**

`examples/quickstart/docs/refund-policy.md`: `tests/fixtures/sample.md`와 동일 내용 복사.

`examples/quickstart/exp.yaml`:
```yaml
# Runs offline with the deterministic "fake" embedder.
# Swap in "openai:text-embedding-3-small" or "gemini:gemini-embedding-001" once you set an API key.
documents: ["docs/*.md"]
questions: questions.json
chunkers:
  - name: recursive
    params: {chunk_size: [128, 256], overlap: [0, 32]}
  - name: sentence_window
    params: {window: [1, 2]}
  - name: markdown
    params: {chunk_size: [512]}
embedders: ["fake"]
retrieval:
  top_k: [3]
  hybrid: [false, true]
hit_threshold: 0.5
cache_path: ".chunklab-cache.db"
```

- [ ] **Step 2: 예제 질문 파일 생성 (span offset은 스크립트로 계산)**

Run:
```bash
uv run python - <<'EOF'
from pathlib import Path
from chunklab.core.models import Question, Span
from chunklab.core.questions.io import save_questions
from chunklab.core.text import load_document

doc = load_document(Path("examples/quickstart/docs/refund-policy.md"))
t = doc.text
def span(start_text, end_text):
    s = t.index(start_text); e = t.index(end_text) + len(end_text)
    return Span(doc.id, s, e)
qs = [
    Question("refund-window", "How long do customers have to request a refund?",
             (span("Customers may request", "original payment method."),)),
    Question("digital-goods", "Can I get a refund on downloaded digital goods?",
             (span("Digital goods", "once downloaded."),)),
    Question("defective", "What happens if the product is defective?",
             (span("Defective products", "start a claim."),)),
    Question("intl-shipping", "How long does international shipping take?",
             (span("International shipping", "7 to 14 days."),)),
    Question("lost-package", "Package says delivered but I never got it, what do I do?",
             (span("If a package is marked", "lost package report."),)),
]
save_questions(qs, Path("examples/quickstart/questions.json"))
print("ok", len(qs))
EOF
```
Expected: `ok 5`

- [ ] **Step 3: 예제 테스트**

`tests/test_examples.py`:
```python
from pathlib import Path

from typer.testing import CliRunner

from chunklab.cli.main import app

ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_example_runs_offline(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(ROOT / "examples" / "quickstart")
    r = CliRunner().invoke(
        app, ["run", "exp.yaml", "--out", str(tmp_path / "r.json"), "--quiet"]
    )
    assert r.exit_code == 0, r.output
    assert "markdown(chunk_size=512)|fake|k=3|hybrid=False" in r.stdout
    assert (tmp_path / "r.json").exists()
```

- [ ] **Step 4: README 작성**

`README.md`:
````markdown
# chunklab

**Benchmark your chunking before you ship it.**

chunklab runs your documents through a matrix of chunking strategies × embedding models × retrieval settings, scores each combination against test questions with span-level ground truth, and tells you which setup actually finds the answer. Everything runs locally: your documents and API keys never leave your machine.

## Install

```bash
pip install chunklab            # OpenAI / Gemini embeddings (bring your own key)
pip install 'chunklab[local]'   # + sentence-transformers for offline embeddings
```

## Quickstart (no API key needed)

```bash
git clone https://github.com/<you>/chunklab && cd chunklab/examples/quickstart
chunklab run exp.yaml
```

You get a table like:

```
combo                                                    hit@3        mrr       ndcg  precision        iou
recursive(chunk_size=128,overlap=0)|fake|k=3|hybrid=False  0.800      0.700      0.712      0.412      0.398
...
```

## How it works

1. **Documents** (`.md`, `.txt`, `.pdf`) are parsed once into normalized plain text. Every chunk keeps its `(start, end)` character offsets into that text.
2. **Questions** carry one or more *golden spans* `(doc_id, start, end)` — chunker-independent ground truth. Generate them automatically:
   ```bash
   export OPENAI_API_KEY=...
   chunklab generate-questions docs/*.md --out questions.json --llm openai --per-doc 10
   ```
   or write them by hand (see `examples/quickstart/questions.json`).
3. **Metrics** per combination:
   - `hit@k` — a retrieved chunk covers ≥ `hit_threshold` (default 50%) of a golden span
   - `mrr` — reciprocal rank of the first hit
   - `ndcg` — graded by span coverage
   - `precision` / `iou` — character-level: how much of what you retrieved was actually the answer. Without these, bigger chunks always "win".

## Use it in CI

```bash
chunklab run exp.yaml --baseline last-good.json --max-drop 0.05 --fail-below hit@5=0.8
```
Exit code 1 if any combination fails a threshold or regresses beyond `--max-drop` versus the baseline.

## Experiment config

```yaml
documents: ["docs/**/*.md", "specs/*.pdf"]
questions: questions.json
chunkers:
  - name: recursive          # separators: paragraph > line > sentence > word
    params: {chunk_size: [256, 512], overlap: [0, 50]}
  - name: sentence_window    # sentence i ± window sentences
    params: {window: [1, 2]}
  - name: markdown           # heading sections, recursive fallback for long ones
    params: {chunk_size: [1024]}
embedders:
  - openai:text-embedding-3-small
  - gemini:gemini-embedding-001
  - local:all-MiniLM-L6-v2
retrieval:
  top_k: [5]
  hybrid: [false, true]      # dense only vs dense + BM25 (reciprocal rank fusion)
hit_threshold: 0.5
cache_path: ~/.chunklab/cache.db   # embeddings cached by (model, text)
```

`chunk_size` and `overlap` are in characters. Embeddings are cached so repeated runs only pay for new text.

## Notes

- `sentence_window` embeds the whole window (v1 simplification).
- The `fake` embedder is a deterministic hashed bag-of-words — fine for demos and CI smoke tests, not for real decisions.
- API keys are read from `OPENAI_API_KEY` / `GEMINI_API_KEY` only.

## License

MIT
````

- [ ] **Step 5: 전체 검증**

Run:
```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check src && uv run pytest -q
```
Expected: lint/format 클린, ty 오류 0(또는 서드파티 stub 부재 경고만), 테스트 전부 통과 (`2 skipped`는 live 테스트)

`ruff format --check` 실패 시 `uv run ruff format .` 후 재확인. ty가 SDK 타입 부재로 오류를 내면 해당 lazy import 라인에 `# type: ignore[import-not-found]` 추가.

- [ ] **Step 6: 실제 CLI로 예제 실행**

Run: `cd examples/quickstart && uv run chunklab run exp.yaml && cd ../..`
Expected: 표 출력, `saved: .../result.json`, `OK`, exit 0. `examples/quickstart/result.json`과 `.chunklab-cache.db`가 생성됨 → `.gitignore`에 `examples/**/result.json`, `.chunklab-cache.db` 추가.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "docs: README and offline quickstart example"
```

---

## 완료 기준 (M1)

- `uv run pytest -q` 전부 통과
- `uv run chunklab run examples/quickstart/exp.yaml` 이 API 키 없이 동작하고 5×2 조합 표 출력
- `CHUNKLAB_LIVE_TESTS=1 OPENAI_API_KEY=... GEMINI_API_KEY=... uv run pytest tests/embedders/test_live.py` 통과 (수동 1회)
- `--baseline` + `--fail-below`로 exit 1 재현 가능

## M2로 넘기는 것

- FastAPI 서버 + HTMX UI (`chunklab ui`)
- span 드래그 라벨링 UI, "이 구간으로 질문 만들기"(`generate_questions`의 span 지정 버전 — `Question` 생성 로직은 `generate.py`에 함수 하나 추가)
- 코드 스니펫 export (LangChain / LlamaIndex / 순수 Python)
- SQLite 실험 이력 저장 (M1은 JSON 파일만)
