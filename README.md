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
`hit@k` keys are per-combo, so a `top_k: [3, 5]` grid needs `--fail-below hit@3=… --fail-below hit@5=…`; a metric name no combo produces is an error (exit code 2).

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
- API keys are read from `OPENAI_API_KEY` and `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) only.

## License

MIT
