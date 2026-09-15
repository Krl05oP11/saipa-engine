"""
PPTX → PDF conversion via LibreOffice headless.

Sending raw bytes and converting server-side (instead of parsing PPTX
XML directly) captures SmartArt, tables and grouped shapes that XML
extraction misses — same rationale documented in local_evalia's
index_course.php on the calling side.
"""
import logging
import subprocess
import tempfile
import uuid
from pathlib import Path

log = logging.getLogger("saipa_engine.rag.pptx_convert")

_CONVERT_TIMEOUT_S = 120


class PptxConversionError(Exception):
    pass


def convert_pptx_to_pdf(pptx_bytes: bytes) -> bytes:
    """
    Blocking — call via asyncio.to_thread() from an async route.

    Each invocation gets its own LibreOffice user profile
    (-env:UserInstallation) because headless soffice shares a single
    profile by default and concurrent conversions can collide on its
    lock file — this service can receive concurrent index_course calls.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_path = tmp / "input.pptx"
        input_path.write_bytes(pptx_bytes)

        profile_dir = tmp / "lo_profile"
        cmd = [
            "soffice",
            "--headless",
            "--norestore",
            f"-env:UserInstallation=file://{profile_dir}",
            "--convert-to", "pdf",
            "--outdir", str(tmp),
            str(input_path),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=_CONVERT_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise PptxConversionError(
                f"LibreOffice conversion timed out after {_CONVERT_TIMEOUT_S}s"
            ) from exc

        output_path = tmp / "input.pdf"
        if result.returncode != 0 or not output_path.exists():
            detail = (result.stderr or result.stdout or "").strip()[:500]
            raise PptxConversionError(
                f"LibreOffice conversion failed (rc={result.returncode}): {detail}"
            )

        return output_path.read_bytes()
