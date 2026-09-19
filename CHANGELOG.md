# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-19

Initial release: core engine (M1) and web UI (M2).

### Added

- Core benchmarking engine: recursive, sentence-window, and markdown chunkers.
- Embedder support for OpenAI, Gemini, and local (sentence-transformers)
  model families, plus a deterministic `fake` embedder for tests/demos.
- Span-level retrieval metrics: hit@k, MRR, NDCG, precision, and IoU.
- Automatic question generation from documents via an LLM provider.
- CLI commands: `run`, `recommend`, `snippet`, `generate-questions`, `version`.
- Baseline regression support for `run` via `--baseline`/`--max-drop` and
  threshold checks via `--fail-below`, for use in CI.
- Web UI: document upload and drag-to-label question authoring, experiment
  runner with live progress, results table with highlighted snippets and
  recommended-combo badge, and run history.
- Local-only security middleware restricting the web UI to
  `127.0.0.1`/`localhost` and rejecting cross-origin writes.
- Claude Code skill (`skills/chunking-benchmark`) for measured chunking
  recommendations.

[0.1.0]: https://github.com/jyp90/chunklab/releases/tag/v0.1.0
