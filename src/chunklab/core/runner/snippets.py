from __future__ import annotations

from chunklab.core.runner.run import ComboResult

FRAMEWORKS = ("langchain", "llamaindex", "python")

_DEFAULT_MODELS = {
    "openai": "text-embedding-3-small",
    "gemini": "gemini-embedding-001",
    "local": "all-MiniLM-L6-v2",
    "fake": "",
}


def _split_embedder(spec: str) -> tuple[str, str]:
    provider, _, model = spec.partition(":")
    return provider, model or _DEFAULT_MODELS.get(provider, "")


def _lc_embeddings(spec: str) -> str:
    p, m = _split_embedder(spec)
    return {
        "openai": (
            "from langchain_openai import OpenAIEmbeddings\n"
            f'embeddings = OpenAIEmbeddings(model="{m}")'
        ),
        "gemini": (
            "from langchain_google_genai import GoogleGenerativeAIEmbeddings\n"
            f'embeddings = GoogleGenerativeAIEmbeddings(model="models/{m}")'
        ),
        "local": (
            "from langchain_huggingface import HuggingFaceEmbeddings\n"
            f'embeddings = HuggingFaceEmbeddings(model_name="{m}")'
        ),
    }.get(p, "# embeddings: plug in your provider here\nembeddings = ...")


def _lc_splitter(c: ComboResult) -> str:
    p = c.chunker_params
    if c.chunker == "recursive":
        return (
            "from langchain_text_splitters import RecursiveCharacterTextSplitter\n"
            f"splitter = RecursiveCharacterTextSplitter(chunk_size={p.get('chunk_size', 512)}, "
            f"chunk_overlap={p.get('overlap', 0)}, separators=['\\n\\n', '\\n', '. ', ' ', ''])"
        )
    if c.chunker == "markdown":
        return (
            "from langchain_text_splitters import MarkdownHeaderTextSplitter, "
            "RecursiveCharacterTextSplitter\n"
            "headers = MarkdownHeaderTextSplitter("
            "headers_to_split_on=[('#','h1'),('##','h2'),('###','h3')])\n"
            f"splitter = RecursiveCharacterTextSplitter(chunk_size={p.get('chunk_size', 1024)}, "
            f"chunk_overlap={p.get('overlap', 0)})\n"
            "# docs = splitter.split_documents(headers.split_text(markdown_text))"
        )
    return (
        "# sentence_window: no direct LangChain equivalent - use chunklab's SentenceWindowChunker\n"
        "from chunklab.core.chunkers import build_chunker\n"
        f"chunker = build_chunker('sentence_window', window={p.get('window', 1)})"
    )


def _langchain(c: ComboResult) -> str:
    retriever = (
        (
            "from langchain_community.retrievers import BM25Retriever\n"
            "from langchain.retrievers import EnsembleRetriever\n"
            f"dense = vectorstore.as_retriever(search_kwargs={{'k': {c.top_k}}})\n"
            f"bm25 = BM25Retriever.from_documents(docs); bm25.k = {c.top_k}\n"
            "retriever = EnsembleRetriever(retrievers=[dense, bm25], weights=[0.5, 0.5])"
        )
        if c.hybrid
        else f"retriever = vectorstore.as_retriever(search_kwargs={{'k': {c.top_k}}})"
    )
    return (
        f"# chunklab recommendation: {c.combo_id}\n"
        f"{_lc_splitter(c)}\n{_lc_embeddings(c.embedder)}\n"
        "from langchain_community.vectorstores import FAISS\n"
        "docs = splitter.split_documents(raw_documents)\n"
        "vectorstore = FAISS.from_documents(docs, embeddings)\n"
        f"{retriever}\n"
    )


def _li_parser(c: ComboResult) -> str:
    p = c.chunker_params
    if c.chunker == "recursive":
        return (
            "from llama_index.core.node_parser import SentenceSplitter\n"
            f"parser = SentenceSplitter(chunk_size={p.get('chunk_size', 512)}, "
            f"chunk_overlap={p.get('overlap', 0)})"
        )
    if c.chunker == "sentence_window":
        return (
            "from llama_index.core.node_parser import SentenceWindowNodeParser\n"
            "parser = SentenceWindowNodeParser.from_defaults("
            f"window_size={p.get('window', 1)})"
        )
    return (
        "from llama_index.core.node_parser import MarkdownNodeParser\n"
        f"parser = MarkdownNodeParser()  # chunk_size {p.get('chunk_size', 1024)} enforced by a "
        "SentenceSplitter post-step if needed"
    )


def _li_embed(spec: str) -> str:
    p, m = _split_embedder(spec)
    return {
        "openai": (
            "from llama_index.embeddings.openai import OpenAIEmbedding\n"
            f'embed_model = OpenAIEmbedding(model="{m}")'
        ),
        "gemini": (
            "from llama_index.embeddings.gemini import GeminiEmbedding\n"
            f'embed_model = GeminiEmbedding(model_name="models/{m}")'
        ),
        "local": (
            "from llama_index.embeddings.huggingface import HuggingFaceEmbedding\n"
            f'embed_model = HuggingFaceEmbedding(model_name="{m}")'
        ),
    }.get(p, "embed_model = ...  # plug in your provider")


def _llamaindex(c: ComboResult) -> str:
    retriever = (
        (
            "from llama_index.retrievers.bm25 import BM25Retriever\n"
            "from llama_index.core.retrievers import QueryFusionRetriever\n"
            f"retriever = QueryFusionRetriever([index.as_retriever(similarity_top_k={c.top_k}), "
            f"BM25Retriever.from_defaults(nodes=nodes, similarity_top_k={c.top_k})], "
            f"similarity_top_k={c.top_k}, mode='reciprocal_rerank')"
        )
        if c.hybrid
        else f"retriever = index.as_retriever(similarity_top_k={c.top_k})"
    )
    return (
        f"# chunklab recommendation: {c.combo_id}\n"
        f"{_li_parser(c)}\n{_li_embed(c.embedder)}\n"
        "from llama_index.core import VectorStoreIndex\n"
        "nodes = parser.get_nodes_from_documents(documents)\n"
        f"index = VectorStoreIndex(nodes, embed_model=embed_model)\n{retriever}\n"
    )


def _python(c: ComboResult) -> str:
    params = ", ".join(f"{k}={v!r}" for k, v in c.chunker_params.items())
    return (
        f"# chunklab recommendation: {c.combo_id}\n"
        "from chunklab.core.chunkers import build_chunker\n"
        "from chunklab.core.embedders import build_embedder\n"
        "from chunklab.core.retrieval import build_index\n"
        "from chunklab.core.text import load_document\n\n"
        f'chunker = build_chunker("{c.chunker}"{", " + params if params else ""})\n'
        f'embedder = build_embedder("{c.embedder}")\n'
        "chunks = [ch for path in paths for ch in chunker.chunk(load_document(path))]\n"
        "index = build_index(chunks, embedder.embed([ch.text for ch in chunks]), embedder, "
        f"hybrid={c.hybrid})\n"
        f'results = index.search("your question", k={c.top_k})\n'
    )


def snippet(combo: ComboResult, framework: str) -> str:
    try:
        fn = {"langchain": _langchain, "llamaindex": _llamaindex, "python": _python}[framework]
    except KeyError:
        raise KeyError(f"unknown framework '{framework}'; available: {list(FRAMEWORKS)}") from None
    return fn(combo)
