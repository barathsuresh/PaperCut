"""
GCS-backed session persistence.
Sessions are stored as JSON blobs at sessions/{session_id}.json in the GCS bucket.
All GCS calls are synchronous (google-cloud-storage has no async SDK) and are
offloaded to a thread pool via asyncio.to_thread so they never block the event loop.
"""
import asyncio
import json
import logging
from typing import Any, Dict

from google.cloud import storage

from backend import config

logger = logging.getLogger(__name__)

_SESSION_PREFIX = "sessions/"


# ── sync helpers (run in thread pool) ────────────────────────────────────────

def _list_session_blobs() -> list:
    client = storage.Client()
    return list(client.bucket(config.GCP_BUCKET_NAME).list_blobs(prefix=_SESSION_PREFIX))


def _download_json(blob_name: str) -> dict:
    client = storage.Client()
    blob = client.bucket(config.GCP_BUCKET_NAME).blob(blob_name)
    return json.loads(blob.download_as_text())


def _upload_json(blob_name: str, data: dict) -> None:
    client = storage.Client()
    blob = client.bucket(config.GCP_BUCKET_NAME).blob(blob_name)
    blob.upload_from_string(json.dumps(data), content_type="application/json")


# ── public async API ──────────────────────────────────────────────────────────

async def load_all_sessions() -> Dict[str, Dict[str, Any]]:
    """Load all persisted sessions from GCS at startup. Returns empty dict on any error."""
    try:
        blobs = await asyncio.to_thread(_list_session_blobs)
        sessions: Dict[str, Dict[str, Any]] = {}
        for blob in blobs:
            session_id = blob.name.removeprefix(_SESSION_PREFIX).removesuffix(".json")
            if not session_id:
                continue
            try:
                data = await asyncio.to_thread(_download_json, blob.name)
                sessions[session_id] = data
            except Exception as e:
                logger.warning("Could not load session %s: %s", session_id, e)
        logger.info("Loaded %d session(s) from GCS.", len(sessions))
        return sessions
    except Exception as e:
        logger.warning("Could not load sessions from GCS: %s", e)
        return {}


async def save_session(session_id: str, data: dict) -> None:
    """Persist a single session to GCS. Errors are logged and swallowed."""
    try:
        blob_name = f"{_SESSION_PREFIX}{session_id}.json"
        await asyncio.to_thread(_upload_json, blob_name, data)
    except Exception as e:
        logger.warning("Could not persist session %s: %s", session_id, e)
