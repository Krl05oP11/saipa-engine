"""
PDF text extraction, with optional 1-indexed inclusive page ranges
(matching local_evalia's `pdf_page_ranges` admin setting convention).
"""
import io

from pypdf import PdfReader


def extract_pdf_text(pdf_bytes: bytes, page_from: int | None = None, page_to: int | None = None) -> str:
    """
    Extracts text from a PDF, optionally restricted to a 1-indexed
    inclusive page range (page_from=1 means the first page).

    Raises ValueError for an invalid or out-of-bounds range — distinct
    from returning empty text, which means the document (or the
    requested range) genuinely has no extractable text layer.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total = len(reader.pages)

    if page_from is not None:
        if page_from <= 0 or page_to is None or page_to < page_from:
            raise ValueError(f"invalid page range: from={page_from} to={page_to}")
        if page_from > total or page_to > total:
            raise ValueError(
                f"page range {page_from}-{page_to} out of bounds (document has {total} pages)"
            )
        pages = reader.pages[page_from - 1:page_to]
    else:
        pages = reader.pages

    return "\n\n".join(p.extract_text() or "" for p in pages)
