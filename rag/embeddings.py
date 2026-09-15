"""
Embeddings — always sentence-transformers, CPU, no GPU or API key
needed (E8: not provider-switchable, unlike the LLM, there's no admin
decision here, it just runs). Multilingual model since course content
is in Spanish.

The model downloads from HuggingFace Hub on first construction
(hundreds of MB) -- callers must NOT let that happen lazily inside a
request (it would eat the caller's timeout budget). main.py warms this
up at startup instead; get_embedder() is cached so later calls are
instant.
"""
import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config import get_settings

log = logging.getLogger("saipa_engine.rag.embeddings")


class SentenceTransformerEmbedder:
    """Minimal interface matching what rag/indexer.py and
    rag/retriever.py already call on OllamaEmbeddings."""

    def __init__(self, model_name: str):
        log.info("Loading sentence-transformers model %r...", model_name)
        self._model = SentenceTransformer(model_name)
        log.info("sentence-transformers model %r loaded.", model_name)

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()


@lru_cache
def get_embedder() -> SentenceTransformerEmbedder:
    settings = get_settings()
    return SentenceTransformerEmbedder(settings.sentence_transformers_model)
