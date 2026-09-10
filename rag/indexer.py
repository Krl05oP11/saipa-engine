"""
RAG Indexer — Fase 1 implementation.
Splits text into chunks and indexes them into ChromaDB with Ollama embeddings.
"""
import logging

import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings

log = logging.getLogger("saipa_engine.rag.indexer")


def index_text(content: str, source: str, course_id: int) -> dict:
    """
    Splits content into chunks and adds them to ChromaDB for the given course.

    Args:
        content:   Raw text to index.
        source:    Identifier for the document (e.g. "page:42", "resource:intro.pdf").
        course_id: Moodle course ID — determines the ChromaDB collection.

    Returns:
        dict with keys: course_id, chunk_count, status
    """
    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(content)

    if not chunks:
        log.warning("index_text: no chunks produced for source '%s' course %d", source, course_id)
        return {"course_id": course_id, "chunk_count": 0, "status": "empty"}

    embeddings = OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )

    collection_name = f"course_{course_id}"
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_or_create_collection(collection_name)

    ids = [f"{source}__{i}" for i in range(len(chunks))]
    embeddings_list = embeddings.embed_documents(chunks)

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings_list,
        metadatas=[{"source": source, "course_id": course_id} for _ in chunks],
    )

    log.info("Indexed %d chunks for course %d source '%s'", len(chunks), course_id, source)
    return {"course_id": course_id, "chunk_count": len(chunks), "status": "ok"}
