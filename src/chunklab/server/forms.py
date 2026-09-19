from __future__ import annotations

from collections.abc import Mapping

import yaml

from chunklab.core.runner import ExperimentConfig


def parse_grid(value: str, *, allow_zero: bool = False) -> list[int]:
    out: list[int] = []
    for part in (p.strip() for p in value.split(",")):
        if not part:
            continue
        if not part.lstrip("-").isdigit():
            raise ValueError(f"'{part}' is not an integer")
        n = int(part)
        if n < 0 or (n == 0 and not allow_zero):
            raise ValueError("grid values must be positive integers")
        out.append(n)
    return out


def _get(form: Mapping[str, str | list[str]], key: str, default: str = "") -> str:
    v = form.get(key, default)
    return v[0] if isinstance(v, list) else v


def _getlist(form: Mapping[str, str | list[str]], key: str) -> list[str]:
    v = form.get(key)
    if v is None:
        return []
    return list(v) if isinstance(v, list) else [v]


def config_from_form(form: Mapping[str, str | list[str]]) -> ExperimentConfig:
    chunkers: list[dict] = []
    if _get(form, "chunker_recursive"):
        chunkers.append(
            {
                "name": "recursive",
                "params": {
                    "chunk_size": parse_grid(_get(form, "recursive_chunk_size", "512")) or [512],
                    "overlap": parse_grid(_get(form, "recursive_overlap", "0"), allow_zero=True)
                    or [0],
                },
            }
        )
    if _get(form, "chunker_sentence_window"):
        chunkers.append(
            {
                "name": "sentence_window",
                "params": {
                    "window": parse_grid(_get(form, "sentence_window_window", "1"), allow_zero=True)
                    or [1],
                },
            }
        )
    if _get(form, "chunker_markdown"):
        chunkers.append(
            {
                "name": "markdown",
                "params": {
                    "chunk_size": parse_grid(_get(form, "markdown_chunk_size", "1024")) or [1024],
                },
            }
        )
    if not chunkers:
        raise ValueError("select at least one chunker")
    embedders = _getlist(form, "embedders") + [
        s.strip() for s in _get(form, "embedder_custom").split(",") if s.strip()
    ]
    if not embedders:
        raise ValueError("select at least one embedder")
    hybrid_mode = _get(form, "hybrid", "dense")
    hybrid = {"dense": [False], "hybrid": [True], "both": [False, True]}.get(hybrid_mode)
    if hybrid is None:
        raise ValueError(f"unknown hybrid mode '{hybrid_mode}' (dense|hybrid|both)")
    return ExperimentConfig(
        documents=["docs/*"],
        questions="questions.json",
        chunkers=chunkers,
        embedders=list(dict.fromkeys(embedders)),
        retrieval={"top_k": parse_grid(_get(form, "top_k", "5")) or [5], "hybrid": hybrid},
        hit_threshold=float(_get(form, "hit_threshold", "0.5") or 0.5),
        cache_path=".chunklab-cache.db",
    )


def config_to_yaml(cfg: ExperimentConfig) -> str:
    return yaml.safe_dump(cfg.model_dump(), sort_keys=False, allow_unicode=True)
