---
name: chunking-benchmark
description: Use when a developer is choosing or validating a RAG chunking strategy — "what chunk size should I use", "how should I split these docs", "retrieval quality dropped", "pick a chunker for LangChain/LlamaIndex". Runs chunklab against the user's own documents and returns a measured recommendation with copy-paste code, instead of a rule-of-thumb answer.
---

# Chunking benchmark (chunklab)

Do not answer chunking questions from memory ("512 with 10% overlap is usually fine"). Measure on the user's documents with `chunklab`, then recommend with numbers.

## When this applies

- The user asks how to chunk / split documents for RAG, embeddings, or a vector store.
- The user asks which chunk size, overlap, or splitter to use, or whether hybrid (BM25 + dense) is worth it.
- The user reports retrieval quality regressions after changing documents or chunking.
- The user is wiring LangChain / LlamaIndex splitters and wants defaults that fit their corpus.

Skip it when the question is about generation quality, reranking, or vector-DB operations — chunklab only measures the chunk → retrieve step.

## Preconditions (check, don't assume)

1. `chunklab` available: check with `chunklab version`. If missing, install it as a tool (small; no torch unless you add the `[local]` extra):
   ```bash
   uv tool install chunkmatrix                                   # PyPI (CLI is still `chunklab`)
   uv tool install git+https://github.com/jyp90/chunklab          # if the PyPI release is not available yet
   ```
   (`pipx install ...` works the same way.) Do not `pip install` into the user's project venv unless they ask.
2. Documents: a folder of `.md` / `.txt` / `.pdf` (≤10 MB each). Ask for the path if not obvious from the repo.
3. Embedding + LLM key: `OPENAI_API_KEY` or `GEMINI_API_KEY` in the environment. If neither is set, say so and run offline with the `fake` embedder/LLM only to demonstrate the workflow — never present offline numbers as a real recommendation.

## Workflow

Work in a scratch directory inside the repo (e.g. `.chunklab/`) so artifacts do not pollute the source tree; suggest adding it to `.gitignore`.

### 1. Test questions

Prefer questions the user already has (FAQ, support tickets, eval set). Otherwise generate:

```bash
chunklab generate-questions docs/*.md --out questions.json --llm openai --per-doc 8 --min-len 120
```

- Exit 2 with "no questions generated" → paragraphs are shorter than `--min-len`; lower it.
- Tell the user these are synthetic and should be skimmed; the web UI (`chunklab ui`) lets them edit or drag-label spans.

### 2. Experiment config

Write `exp.yaml` from this template, adjusting to the corpus (Markdown-heavy → keep `markdown`; PDFs/plain text → drop it):

```yaml
documents: ["docs/*"]
questions: questions.json
chunkers:
  - name: recursive
    params: {chunk_size: [256, 512, 1024], overlap: [0, 64]}
  - name: sentence_window
    params: {window: [1, 2]}
  - name: markdown
    params: {chunk_size: [800]}
embedders: ["openai:text-embedding-3-small"]   # match what the user's pipeline uses
retrieval:
  top_k: [5]                                     # match the user's retriever k
  hybrid: [false, true]
hit_threshold: 0.5
cache_path: .chunklab-cache.db
```

`chunk_size`/`overlap` are characters. Use the same `top_k` and embedding model the production pipeline uses — otherwise the numbers do not transfer.

### 3. Run and read

```bash
chunklab run exp.yaml --json > result.json        # progress/warnings go to stderr
chunklab recommend result.json --json             # {"combo_id": ..., "reason": ...}
```

Interpretation rules (these are what `recommend` implements — repeat them to the user):

- `hit@k` first: fraction of questions whose gold span is ≥50% covered by a retrieved chunk. Anything within 0.05 of the best is "equally good at finding the answer".
- Among those, highest `precision` wins: share of retrieved characters that were actually answer text. This is the anti-"just use huge chunks" guard.
- Ties → fewer chunks (cheaper index).
- If every combo has `hit@k` < 0.6, the problem is probably the questions or the documents (scanned PDFs, tables), not the chunker. Open `result.json` `per_question` and look at what was retrieved for misses before recommending anything.

Also report the second-best combo and the spread: a 0.02 difference is noise on 10 questions; say so.

### 4. Hand over code

```bash
chunklab snippet result.json --combo "<combo_id>" --framework langchain   # or llamaindex | python
```

Paste the snippet into the user's stack. Frameworks differ from chunklab's implementation in edge cases (sentence boundaries, header handling), so state that the numbers are from chunklab's chunkers and invite a re-check with `chunklab ui` if the port matters.

### 5. Leave a regression hook

Commit `exp.yaml` + `questions.json` and record the baseline:

```bash
chunklab run exp.yaml --out .chunklab/baseline.json
# CI:
chunklab run exp.yaml --baseline .chunklab/baseline.json --max-drop 0.05 --fail-below hit@5=0.8
```

Exit codes: 0 ok, 1 metric failure/regression, 2 config or usage error. `hit@k` keys are per-combo, so a `top_k: [3, 5]` grid needs one `--fail-below` per k.

## Reporting to the user

Lead with the recommendation and its two numbers, then the runner-up, then caveats (synthetic questions, embedding model used, document count). Offer `chunklab ui` for a visual check of what each chunker retrieved. Keep raw tables out of chat unless asked; point at `result.json`.

## Pitfalls

- Running with `fake` embedder and presenting it as a result. `fake` is a hashed bag-of-words for demos only.
- Changing `hit_threshold` between baseline and current run — chunklab flags it, don't "fix" by deleting the baseline.
- Documents that yield `EmptyDocumentError` (image-only PDFs) are skipped with a warning; tell the user OCR is out of scope.
- Absolute glob patterns work, but keep `documents:` relative so `exp.yaml` is portable for CI.
