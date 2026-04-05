"""
GCS artifact store — upload and download generated code files.

GCS layout per session:
  papers/{session_id}/scaffold/   ← Node 2 (PyTorch)
  papers/{session_id}/hardware/   ← Node 3 (CUDA)
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from google.cloud import storage

from backend import config
from backend.tools.gcs_utils import is_gcs_transport_error

logger = logging.getLogger(__name__)

_GCS_PREFIX_MAP = {
    "implementation": "scaffold",
    "acceleration":   "hardware",
}

def _get_client() -> storage.Client:
    return storage.Client()


def _upload_dir_sync(session_id: str, local_dir: Path, gcs_prefix: str) -> list[str]:
    """Synchronous: upload every file in local_dir to GCS. Returns uploaded filenames."""
    client  = _get_client()
    bucket  = client.bucket(config.GCP_BUCKET_NAME)
    uploaded: list[str] = []
    for f in local_dir.iterdir():
        if not f.is_file():
            continue
        blob_name = f"papers/{session_id}/{gcs_prefix}/{f.name}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(f))
        uploaded.append(f.name)
        logger.debug("Artifact uploaded | %s", blob_name)
    logger.info(
        "Artifacts uploaded | session=%s prefix=%s count=%d",
        session_id, gcs_prefix, len(uploaded),
    )
    return uploaded


def _download_blob_sync(session_id: str, gcs_prefix: str, filename: str) -> Optional[str]:
    """Synchronous: download a single file from GCS, return its text content."""
    client    = _get_client()
    bucket    = client.bucket(config.GCP_BUCKET_NAME)
    blob_name = f"papers/{session_id}/{gcs_prefix}/{filename}"
    blob      = bucket.blob(blob_name)
    if not blob.exists():
        return None
    content = blob.download_as_text()
    logger.debug("Artifact downloaded from GCS | %s", blob_name)
    return content


def _list_artifacts_sync(session_id: str, gcs_prefix: str) -> list[str]:
    client = _get_client()
    bucket = client.bucket(config.GCP_BUCKET_NAME)
    prefix = f"papers/{session_id}/{gcs_prefix}/"
    blobs = bucket.list_blobs(prefix=prefix)
    files: list[str] = []
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        name = blob.name.removeprefix(prefix)
        if name:
            files.append(name)
    return sorted(files)


def _delete_prefix_sync(session_id: str) -> int:
    """Synchronous: delete all blobs under papers/{session_id}/. Returns count."""
    client  = _get_client()
    bucket  = client.bucket(config.GCP_BUCKET_NAME)
    blobs   = list(bucket.list_blobs(prefix=f"papers/{session_id}/"))
    for blob in blobs:
        blob.delete()
    logger.info("GCS blobs deleted | session=%s count=%d", session_id, len(blobs))
    return len(blobs)


# ── Async wrappers ────────────────────────────────────────────────────────────

async def upload_artifacts(session_id: str, local_dir: Path, artifact_group: str) -> list[str]:
    """Upload a node output directory to GCS. artifact_group: 'implementation' | 'acceleration'"""
    gcs_prefix = _GCS_PREFIX_MAP.get(artifact_group, artifact_group)
    try:
        return await asyncio.to_thread(_upload_dir_sync, session_id, local_dir, gcs_prefix)
    except Exception as e:
        if is_gcs_transport_error(e):
            logger.error(
                "Artifact upload failed due to GCS auth transport timeout | session=%s group=%s | %s",
                session_id, artifact_group, e,
            )
        else:
            logger.error("Artifact upload failed | session=%s group=%s | %s", session_id, artifact_group, e)
        return []


async def download_artifact(session_id: str, artifact_group: str, filename: str) -> Optional[str]:
    """Download a single artifact file from GCS. Returns text content or None."""
    gcs_prefix = _GCS_PREFIX_MAP.get(artifact_group, artifact_group)
    try:
        return await asyncio.to_thread(_download_blob_sync, session_id, gcs_prefix, filename)
    except Exception as e:
        if is_gcs_transport_error(e):
            logger.error(
                "Artifact download failed due to GCS auth transport timeout | session=%s file=%s | %s",
                session_id, filename, e,
            )
        else:
            logger.error("Artifact download failed | session=%s file=%s | %s", session_id, filename, e)
        return None


async def list_artifacts(session_id: str, artifact_group: str) -> list[str]:
    """List artifact filenames from GCS for a session/group."""
    gcs_prefix = _GCS_PREFIX_MAP.get(artifact_group, artifact_group)
    try:
        return await asyncio.to_thread(_list_artifacts_sync, session_id, gcs_prefix)
    except Exception as e:
        if is_gcs_transport_error(e):
            logger.error(
                "Artifact listing failed due to GCS auth transport timeout | session=%s group=%s | %s",
                session_id, artifact_group, e,
            )
        else:
            logger.error("Artifact listing failed | session=%s group=%s | %s", session_id, artifact_group, e)
        return []


async def delete_all_artifacts(session_id: str) -> int:
    """Delete every GCS blob for a session (PDF, session.json, scaffold, hardware)."""
    try:
        return await asyncio.to_thread(_delete_prefix_sync, session_id)
    except Exception as e:
        if is_gcs_transport_error(e):
            logger.error("GCS delete failed due to auth transport timeout | session=%s | %s", session_id, e)
        else:
            logger.error("GCS delete failed | session=%s | %s", session_id, e)
        return 0
