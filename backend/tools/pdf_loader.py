import httpx
from google.cloud import storage

from backend import config


def upload_pdf_from_url(pdf_url: str, session_id: str) -> str:
    """Download PDF from URL and upload to GCS. Returns GCS URI."""
    response = httpx.get(pdf_url, follow_redirects=True, timeout=60.0)
    response.raise_for_status()
    return upload_pdf_bytes(response.content, session_id)


def upload_pdf_bytes(pdf_bytes: bytes, session_id: str) -> str:
    """Upload PDF bytes to GCS. Returns GCS URI."""
    client = storage.Client()
    bucket = client.bucket(config.GCP_BUCKET_NAME)
    blob_name = f"papers/{session_id}/paper.pdf"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(pdf_bytes, content_type="application/pdf")
    return f"gs://{config.GCP_BUCKET_NAME}/{blob_name}"
