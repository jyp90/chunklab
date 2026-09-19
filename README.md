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
git clone https://github.com/jyp90/chunklab && cd chunklab/examples/quickstart
chunklab run exp.yaml
```

You get a table like:

```
combo                                                           hit@3        mrr       ndcg  precision        iou
-----------------------------------------------------------------------------------------------------------------
recursive(chunk_size=128,overlap=0)|fake|k=3|hybrid=False       0.800      0.800      0.800      0.206      0.206
recursive(chunk_size=128,overlap=0)|fake|k=3|hybrid=True        0.800      0.800      0.800      0.201      0.201
recursive(chunk_size=128,overlap=32)|fake|k=3|hybrid=False      1.000      0.867      0.900      0.235      0.235
...
```

## Web UI

```bash
chunklab ui              # opens http://127.0.0.1:7860 — everything stays on your machine
```
1. **Documents & Questions** — upload md/txt/pdf, drag a passage in the viewer → *Add question* (or *Generate question from selection* with your LLM key), or *Auto-generate* N questions per document. Export/import `questions.json` to share with the CLI.
2. **Experiment** — tick chunkers, type parameter grids, pick embedders, *Run*. Progress updates live. *Export exp.yaml* gives you the same run for `chunklab run` / CI.
3. **Results** — metrics table with a **Recommended** badge (highest precision among combos within 0.05 of the best hit@k, ties → fewer chunks), per-question drill-down with the gold span and retrieved chunks highlighted on the original text, and copy-paste snippets for LangChain / LlamaIndex / plain Python.

State lives in `./chunklab.db` (SQLite) plus `./docs/` and the embedding cache. Delete them to start over.

The server only answers requests addressed to `127.0.0.1`/`localhost` and rejects cross-origin writes, so another site in your browser cannot reach it via DNS rebinding or a hidden form post.

## For agents / scripts

```bash
chunklab run exp.yaml --json > result.json          # machine-readable
chunklab recommend result.json --json               # {"combo_id": ..., "reason": ...}
chunklab snippet result.json --combo "<id>" --framework langchain
```

## Claude Code skill

`skills/chunking-benchmark/SKILL.md` teaches Claude Code to benchmark instead of guess when you ask "how should I chunk these docs?". Install it for your user or project:

```bash
mkdir -p ~/.claude/skills && cp -r skills/chunking-benchmark ~/.claude/skills/     # all projects
# or: mkdir -p .claude/skills && cp -r skills/chunking-benchmark .claude/skills/  # this repo only
```

Then ask Claude: *"pick a chunking strategy for docs/ and give me the LangChain code"* — it runs `generate-questions → run --json → recommend → snippet` and answers with measured numbers.

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
Exit codes:

| code | meaning |
| ---- | ------- |
| 0 | every combination ran and passed the checks |
| 1 | the run completed but a combination failed a threshold or regressed beyond `--max-drop` |
| 2 | usage or configuration error — bad flag, invalid YAML, missing file, unknown provider, missing API key |

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
