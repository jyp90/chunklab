from __future__ import annotations

import glob
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator


class ChunkerGrid(BaseModel):
    name: str
    params: dict[str, list[Any]] = Field(default_factory=dict)

    @field_validator("params")
    @classmethod
    def _params_must_be_scalar(
        cls, params: dict[str, list[Any]], info: ValidationInfo
    ) -> dict[str, list[Any]]:
        name = info.data.get("name", "?")
        for key, values in params.items():
            for value in values:
                if not isinstance(value, int | float | str | bool):
                    raise ValueError(
                        f"chunker '{name}' param '{key}' has non-scalar value {value!r}"
                    )
        return params


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
            # `Path(base) / pattern` keeps an absolute pattern unchanged; glob.glob
            # (unlike Path.glob) accepts absolute patterns.
            for hit in glob.glob(str(Path(base_dir) / pattern), recursive=True):
                p = Path(hit)
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
