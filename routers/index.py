"""
Index router — POST /index
Receives text content from Moodle and indexes it into ChromaDB for RAG retrieval.
"""
import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from rag.indexer import index_text
from routers.health import _verify_token

log = logging.getLogger("saipa_engine.routers.index")

router = APIRouter()


class IndexRequest(BaseModel):
    course_id: int
    content: str
    source: str


class IndexResponse(BaseModel):
    course_id: int
    chunk_count: int
    status: str


@router.post("/index", response_model=IndexResponse, tags=["rag"])
async def index(
    req: IndexRequest,
    token: None = Depends(_verify_token),
) -> IndexResponse:
    """
    Indexes text content into ChromaDB for the specified course.
    Called by Moodle when course content changes.
    """
    result = index_text(
        content=req.content,
        source=req.source,
        course_id=req.course_id,
    )
    return IndexResponse(**result)
