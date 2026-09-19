from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from chunklab.core.embedders import build_embedder
from chunklab.core.embedders.base import Embedder
from chunklab.core.models import Document, Question
from chunklab.core.runner import ExperimentConfig, expand_matrix, run_matrix, validate_questions
from chunklab.server.forms import config_to_yaml
from chunklab.server.store import Store


@dataclass
class RunStatus:
    run_id: str
    state: str
    done: int
    total: int
    error: str | None = None


class RunManager:
    def __init__(
        self,
        store: Store,
        cache_file: Path,
        embedder_factory: Callable[[str], Embedder] = build_embedder,
    ) -> None:
        self._store = store
        self._cache_file = Path(cache_file)
        self._factory = embedder_factory
        self._status: dict[str, RunStatus] = {}
        self._lock = threading.Lock()

    def status(self, run_id: str) -> RunStatus | None:
        with self._lock:
            st = self._status.get(run_id)
            return replace(st) if st else None

    def start(
        self,
        cfg: ExperimentConfig,
        docs: Sequence[Document],
        questions: Sequence[Question],
    ) -> str:
        validate_questions(questions, docs)
        run_id = uuid.uuid4().hex[:12]
        total = len(expand_matrix(cfg))
        yaml_text = config_to_yaml(cfg)
        with self._lock:
            self._status[run_id] = RunStatus(run_id, "running", 0, total)
        self._store.save_run(run_id, "running", yaml_text)

        def progress(_: str) -> None:
            with self._lock:
                self._status[run_id].done += 1

        def work() -> None:
            try:
                result = run_matrix(
                    cfg, list(docs), list(questions), self._cache_file, self._factory, progress
                )
                result = replace(result, run_id=run_id)
                self._store.save_run(run_id, "done", yaml_text, result=result)
                with self._lock:
                    self._status[run_id].state = "done"
            except Exception as e:  # noqa: BLE001 - surfaced to the UI
                msg = f"{type(e).__name__}: {e}"
                self._store.save_run(run_id, "error", yaml_text, error=msg)
                with self._lock:
                    self._status[run_id].state = "error"
                    self._status[run_id].error = msg

        threading.Thread(target=work, name=f"chunklab-run-{run_id}", daemon=True).start()
        return run_id
