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
                        combo,
                        chunk_cache[key],
                        questions,
                        embedders[combo.embedder],
                        cfg.hit_threshold,
                    )
                )
            except Exception as e:  # noqa: BLE001 - partial failure is a feature
                results.append(
                    ComboResult(
                        combo.id,
                        combo.chunker,
                        combo.params_dict(),
                        combo.embedder,
                        combo.top_k,
                        combo.hybrid,
                        error=f"{type(e).__name__}: {e}",
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
