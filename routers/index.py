"""
Index router — POST /index, /index-pdf-bytes, /index-pptx-bytes
Receives content from Moodle (text, or PDF/PPTX bytes) and indexes it
into ChromaDB for RAG retrieval.
"""
import asyncio
import base64
import binascii
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from rag.indexer import index_text
from rag.pdf_extract import extract_pdf_text
from rag.pptx_convert import PptxConversionError, convert_pptx_to_pdf
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


class IndexPdfBytesRequest(BaseModel):
    course_id: int
    content_b64: str
    source: str
    page_from: int | None = None
    page_to: int | None = None


class IndexPptxBytesRequest(BaseModel):
    course_id: int
    content_b64: str
    source: str


def _decode_b64(content_b64: str) -> bytes:
    try:
        return base64.b64decode(content_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"content_b64 is not valid base64: {exc}",
        ) from exc


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


@router.post("/index-pdf-bytes", response_model=IndexResponse, tags=["rag"])
async def index_pdf_bytes(
    req: IndexPdfBytesRequest,
    token: None = Depends(_verify_token),
) -> IndexResponse:
    """
    Extracts text from a PDF (optionally a 1-indexed inclusive page
    range, for indexing one chapter of a larger file as its own source)
    and indexes it into ChromaDB.
    """
    pdf_bytes = _decode_b64(req.content_b64)

    try:
        text = extract_pdf_text(pdf_bytes, req.page_from, req.page_to)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="PDF has no extractable text layer (scanned/image-only PDF?)",
        )

    result = index_text(content=text, source=req.source, course_id=req.course_id)
    return IndexResponse(**result)


@router.post("/index-pptx-bytes", response_model=IndexResponse, tags=["rag"])
async def index_pptx_bytes(
    req: IndexPptxBytesRequest,
    token: None = Depends(_verify_token),
) -> IndexResponse:
    """
    Converts a PPTX to PDF via LibreOffice headless (captures SmartArt,
    tables and grouped shapes that direct XML extraction would miss),
    extracts its text, and indexes it into ChromaDB.
    """
    pptx_bytes = _decode_b64(req.content_b64)

    try:
        pdf_bytes = await asyncio.to_thread(convert_pptx_to_pdf, pptx_bytes)
    except PptxConversionError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    text = extract_pdf_text(pdf_bytes)
    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="PPTX has no extractable text after conversion (image-only slides?)",
        )

    result = index_text(content=text, source=req.source, course_id=req.course_id)
    return IndexResponse(**result)
