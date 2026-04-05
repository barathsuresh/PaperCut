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

# Module-level singleton — avoids re-initialising credentials on every GCS call
_gcs_client: storage.Client | None = None


def _get_storage_client() -> storage.Client:
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    return _gcs_client


# ── sync helpers (run in thread pool) ────────────────────────────────────────

def _list_session_blobs() -> list:
    return list(
        _get_storage_client().bucket(config.GCP_BUCKET_NAME).list_blobs(prefix=_SESSION_PREFIX)
    )


def _download_json(blob_name: str) -> dict:
    blob = _get_storage_client().bucket(config.GCP_BUCKET_NAME).blob(blob_name)
    return json.loads(blob.download_as_text())


def _upload_json(blob_name: str, data: dict) -> None:
    blob = _get_storage_client().bucket(config.GCP_BUCKET_NAME).blob(blob_name)
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
            except json.JSONDecodeError as e:
                logger.warning("Corrupt session blob %s — skipping: %s", session_id, e)
            except Exception as e:
                logger.warning("Could not load session %s: %s", session_id, e)
        logger.info("Loaded %d session(s) from GCS.", len(sessions))
        return sessions
    except Exception as e:
        logger.warning("Could not load sessions from GCS: %s", e)
        return {}


def _delete_blob(blob_name: str) -> None:
    blob = _get_storage_client().bucket(config.GCP_BUCKET_NAME).blob(blob_name)
    blob.delete()


async def delete_session(session_id: str) -> None:
    """Delete the session JSON blob from GCS."""
    blob_name = f"{_SESSION_PREFIX}{session_id}.json"
    try:
        await asyncio.to_thread(_delete_blob, blob_name)
        logger.info("Session blob deleted | session=%s", session_id)
    except Exception as e:
        logger.warning("Could not delete session blob | session=%s | %s", session_id, e)


async def save_session(session_id: str, data: dict, _retries: int = 2) -> None:
    """Persist a single session to GCS with up to *_retries* retry attempts."""
    blob_name = f"{_SESSION_PREFIX}{session_id}.json"
    for attempt in range(1, _retries + 2):
        try:
            await asyncio.to_thread(_upload_json, blob_name, data)
            logger.debug(
                "Session persisted | session=%s | keys=%s", session_id, list(data.keys())
            )
            return
        except Exception as e:
            if attempt <= _retries:
                wait = 0.5 * attempt
                logger.warning(
                    "Session persist attempt %d/%d failed | session=%s | retrying in %.1fs | %s",
                    attempt, _retries + 1, session_id, wait, e,
                )
                await asyncio.sleep(wait)
            else:
                logger.error(
                    "Session persist FAILED after %d attempts | session=%s | %s",
                    _retries + 1, session_id, e,
                )
