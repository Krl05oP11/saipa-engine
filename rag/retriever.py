"""
RAG Retriever — Fase 1 implementation.
Queries ChromaDB for relevant chunks using OllamaEmbeddings.
"""
import logging

import chromadb
from langchain_ollama import OllamaEmbeddings

from config import get_settings

log = logging.getLogger("saipa_engine.rag.retriever")


def retrieve(query: str, course_id: int, top_k: int = 5) -> list[dict]:
    """
    Returns the top_k most relevant document chunks for the query.
    If the collection is empty or doesn't exist, returns [] gracefully.

    Each result dict has: content, source, score.
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

    embeddings = OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )

    try:
        query_embedding = embeddings.embed_query(query)
        n = min(top_k, collection.count())
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        log.warning("Retrieval failed for course %d: %s", course_id, e)
        return []

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "content": doc,
            "source": meta.get("source", "unknown"),
            "score": float(dist),
        })

    log.info("Retrieved %d chunks for course %d", len(output), course_id)
    return output
