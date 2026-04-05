import asyncio
import logging

import httpx
from google.auth.exceptions import RefreshError, TransportError as GoogleAuthTransportError
from google.cloud import storage

from backend import config

logger = logging.getLogger(__name__)

MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB


class GCSUnavailableError(RuntimeError):
    """Raised when GCS cannot be reached or auth token refresh fails."""


def _is_pdf(data: bytes) -> bool:
    return data[:4] == b"%PDF"


async def upload_pdf_from_url(pdf_url: str, session_id: str) -> str:
    """Download PDF from URL (with size cap + content checks) and upload to GCS."""
    logger.info("Fetching PDF from URL | session=%s | url=%s", session_id, pdf_url)

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "GET", pdf_url, follow_redirects=True, timeout=60.0
        ) as response:
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower() and "octet-stream" not in content_type.lower():
                logger.warning(
                    "URL content-type may not be PDF | session=%s | content-type=%s",
                    session_id, content_type,
                )

            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes(chunk_size=65_536):
                total += len(chunk)
                if total > MAX_PDF_BYTES:
                    raise ValueError(
                        f"PDF exceeds maximum download size of {MAX_PDF_BYTES // (1024 * 1024)} MB"
                    )
                chunks.append(chunk)

    pdf_bytes = b"".join(chunks)
    if not _is_pdf(pdf_bytes):
        raise ValueError("Downloaded content does not appear to be a valid PDF (bad magic bytes)")

    logger.info("PDF downloaded | session=%s | size=%.1f KB", session_id, len(pdf_bytes) / 1024)
    return await upload_pdf_bytes(pdf_bytes, session_id)


async def upload_pdf_bytes(pdf_bytes: bytes, session_id: str) -> str:
    """Upload PDF bytes to GCS. Returns GCS URI."""
    size_kb = len(pdf_bytes) / 1024
    logger.info("Uploading PDF to GCS | session=%s | size=%.1f KB", session_id, size_kb)
    try:
        gcs_uri = await asyncio.to_thread(_gcs_upload_sync, pdf_bytes, session_id)
    except (GoogleAuthTransportError, RefreshError) as exc:
        logger.error("GCS auth transport error | session=%s | %s", session_id, exc)
        raise GCSUnavailableError(
            "Google Cloud authentication timed out while uploading the PDF."
        ) from exc
    except Exception as exc:
        logger.error("GCS upload failed | session=%s | %s", session_id, exc, exc_info=True)
        raise
    logger.info("PDF uploaded | session=%s | uri=%s", session_id, gcs_uri)
    return gcs_uri


def _gcs_upload_sync(pdf_bytes: bytes, session_id: str) -> str:
    client = storage.Client()
    bucket = client.bucket(config.GCP_BUCKET_NAME)
    blob_name = f"papers/{session_id}/paper.pdf"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(pdf_bytes, content_type="application/pdf")
    return f"gs://{config.GCP_BUCKET_NAME}/{blob_name}"
