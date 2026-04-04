import asyncio

import httpx
from google.cloud import storage

from backend import config


async def upload_pdf_from_url(pdf_url: str, session_id: str) -> str:
    """Download PDF from URL and upload to GCS. Returns GCS URI."""
    async with httpx.AsyncClient() as client:
        response = await client.get(pdf_url, follow_redirects=True, timeout=60.0)
        response.raise_for_status()
    return await upload_pdf_bytes(response.content, session_id)


async def upload_pdf_bytes(pdf_bytes: bytes, session_id: str) -> str:
    """Upload PDF bytes to GCS. Returns GCS URI."""
    return await asyncio.to_thread(_gcs_upload_sync, pdf_bytes, session_id)


def _gcs_upload_sync(pdf_bytes: bytes, session_id: str) -> str:
    client = storage.Client()
    bucket = client.bucket(config.GCP_BUCKET_NAME)
    blob_name = f"papers/{session_id}/paper.pdf"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(pdf_bytes, content_type="application/pdf")
    return f"gs://{config.GCP_BUCKET_NAME}/{blob_name}"
