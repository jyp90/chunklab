# Show HN draft

**Title (≤80 chars):**
Show HN: Chunklab – benchmark RAG chunking on your own docs, locally, with a UI

**Body:**

I kept picking chunk sizes for RAG pipelines by gut feeling ("512 with some overlap"), and every time the documents changed I had no idea whether retrieval got better or worse. So I built a small local tool that turns the question into a measurement.

You give it a folder of md/txt/pdf and a handful of test questions (it can generate them with your own OpenAI/Gemini key, or you drag-select the answer passage in the browser). It then runs a grid — recursive / sentence-window / markdown chunkers × chunk sizes × embedding models × dense vs dense+BM25 — and scores every combination.

Two things I think are a bit different from other chunking evals:

- Ground truth is a *character span* in the source text, not "chunk #7". That makes the same test set valid across every chunker, and it lets the tool report **precision** (how much of what you retrieved was actually the answer) next to hit@k. Without precision, the biggest chunk always "wins".
- It's meant to stay in your loop, not be a one-off notebook: `chunklab run --baseline last.json --fail-below hit@5=0.8` exits non-zero in CI when a doc or config change regresses retrieval, and `chunklab recommend` / `chunklab snippet` give an agent (or you) a one-line answer plus LangChain / LlamaIndex code.

Everything runs on 127.0.0.1; documents and API keys never leave the machine. The UI is FastAPI + htmx, no build step, `pip install` and go.

Repo: https://github.com/jyp90/chunklab

Things I'd love feedback on: whether span-overlap ≥50% is the right hit definition, what chunkers you'd want next (semantic / Chonkie wrappers are on the list), and whether the CI regression angle is actually useful to anyone besides me.

---

**Comment prepared for "how is this different from X":**

- *Ragas / DeepEval*: those evaluate the generated answer; this only measures retrieval and only varies chunking/retrieval knobs, so it runs in seconds with no judge LLM.
- *Chroma's chunking_evaluation*: same metric family (their paper is where the precision/IoU idea comes from) but a research repo without a UI, questions labelling, or CI mode.
- *The other `chunklab` on PyPI*: similar goal (statistical comparison of chunkers). Ours adds the browser labelling UI, hybrid retrieval axis, CI regression gate and agent CLI; theirs focuses on significance testing. (Adjust once the final package name is decided.)
