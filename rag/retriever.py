"""
RAG Retriever — hybrid search: dense (OllamaEmbeddings + ChromaDB cosine)
fused with BM25 (lexical) via Reciprocal Rank Fusion.
"""
import logging
import re

import chromadb
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi

from config import get_settings

log = logging.getLogger("saipa_engine.rag.retriever")

# ChromaDB's get() is bounded internally at ~100 items regardless of the
# limit parameter — paginate with a safe batch size to fetch everything.
_GET_BATCH_SIZE = 500

# Standard Reciprocal Rank Fusion constant — dampens the impact of rank 1
# vs rank 2 without needing per-leg weights we have no data to calibrate.
_RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-záéíóúñü0-9]+", text.lower())


def _fetch_all_chunks(
    collection, where: dict | None
) -> tuple[list[str], list[str], list[dict]]:
    """Paginates collection.get() to fetch every chunk (optionally filtered
    server-side by `where`). Returns (ids, documents, metadatas)."""
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    offset = 0
    while True:
        result = collection.get(
            include=["documents", "metadatas"],
            where=where,
            limit=_GET_BATCH_SIZE,
            offset=offset,
        )
        batch_ids = result.get("ids") or []
        ids.extend(batch_ids)
        docs.extend(result.get("documents") or [])
        metas.extend(result.get("metadatas") or [])
        if len(batch_ids) < _GET_BATCH_SIZE:
            break
        offset += _GET_BATCH_SIZE
    return ids, docs, metas


def retrieve(
    query: str,
    course_id: int,
    top_k: int = 5,
    allowed_sources: list[str] | None = None,
) -> list[dict]:
    """
    Returns the top_k most relevant document chunks for the query, fusing
    dense (embedding) and BM25 (lexical) search via Reciprocal Rank Fusion.
    If the collection is empty or doesn't exist, returns [] gracefully.

    allowed_sources: if given, restricts results to chunks whose `source`
    metadata is in this list — applied server-side to BOTH the dense and
    the BM25 leg (not a post-filter, which would break top_k).

    Each result dict has: content, source, score (fused RRF score, higher
    = more relevant — NOT a distance; callers only ever read content/source).
    """
    settings = get_settings()
    collection_name = f"course_{course_id}"

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        log.info("Collection %s not found — no context for course %d", collection_name, course_id)
        return []

    if collection.count() == 0:
        log.info("Collection %s is empty — returning no context", collection_name)
        return []

    where = {"source": {"$in": allowed_sources}} if allowed_sources else None
    # Safety ceiling for n_results — collection.count() is the *unfiltered*
    # total, same margin the original dense-only code used.
    candidate_k = min(max(top_k * 3, top_k), collection.count())

    # ── Dense leg ──────────────────────────────────────────────────────────
    dense_ids: list[str] = []
    dense_docs: dict[str, dict] = {}
    try:
        embeddings = OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )
        query_embedding = embeddings.embed_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
            where=where,
            include=["documents", "metadatas"],
        )
        dense_ids = results["ids"][0]
        for id_, doc, meta in zip(dense_ids, results["documents"][0], results["metadatas"][0]):
            dense_docs[id_] = {"content": doc, "source": meta.get("source", "unknown")}
    except Exception as e:
        log.warning("Dense retrieval failed for course %d: %s", course_id, e)

    # ── BM25 leg ───────────────────────────────────────────────────────────
    bm25_ids: list[str] = []
    bm25_docs: dict[str, dict] = {}
    try:
        ids, docs, metas = _fetch_all_chunks(collection, where)
        if ids:
            tokenized_corpus = [_tokenize(d) for d in docs]
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(_tokenize(query))
            ranked = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:candidate_k]
            bm25_ids = [ids[i] for i in ranked]
            for i in ranked:
                bm25_docs[ids[i]] = {"content": docs[i], "source": metas[i].get("source", "unknown")}
    except Exception as e:
        log.warning("BM25 retrieval failed for course %d: %s", course_id, e)

    if not dense_ids and not bm25_ids:
        return []

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────
    fused_scores: dict[str, float] = {}
    for rank, id_ in enumerate(dense_ids):
        fused_scores[id_] = fused_scores.get(id_, 0.0) + 1.0 / (_RRF_K + rank + 1)
    for rank, id_ in enumerate(bm25_ids):
        fused_scores[id_] = fused_scores.get(id_, 0.0) + 1.0 / (_RRF_K + rank + 1)

    all_docs = {**dense_docs, **bm25_docs}
    top_ids = sorted(fused_scores, key=lambda i: fused_scores[i], reverse=True)[:top_k]

    output = [
        {"content": all_docs[i]["content"], "source": all_docs[i]["source"], "score": fused_scores[i]}
        for i in top_ids
    ]

    log.info(
        "Retrieved %d chunks for course %d (dense=%d bm25=%d fused=%d)",
        len(output), course_id, len(dense_ids), len(bm25_ids), len(output),
    )
    return output
