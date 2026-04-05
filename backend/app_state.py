from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import HTTPException
from google.genai.errors import ClientError as GeminiClientError

from backend.schemas.session import SessionData, SessionStore
from backend.session_store import save_session
from backend.tools.artifact_store import upload_artifacts
from backend.tools.gemini_client import get_flash_model, pdf_part_from_gcs

logger = logging.getLogger(__name__)

sessions: SessionStore = {}


def raise_gemini_error(e: GeminiClientError) -> None:
    if e.code == 429:
        raise HTTPException(
            status_code=429,
            detail="Gemini rate limit hit. Wait a moment and try again.",
        )
    raise HTTPException(status_code=502, detail=f"Gemini API error: {e.message}")


def persist(session_id: str) -> None:
    data = sessions.get(session_id)
    if data is None:
        logger.warning("persist called for unknown session=%s — skipping", session_id)
        return
    asyncio.create_task(save_session(session_id, data))


async def fetch_and_store_title(session_id: str, gcs_uri: str) -> None:
    try:
        model = get_flash_model()
        response = await model.generate_content_async(
            [
                pdf_part_from_gcs(gcs_uri),
                "Return only the paper title, nothing else. No quotes, no punctuation beyond the title itself.",
            ]
        )
        title = (response.text or "").strip().strip('"').strip("'")
        if title and session_id in sessions:
            sessions[session_id]["name"] = title[:160]
            await save_session(session_id, sessions[session_id])
            logger.info("Session title set | session=%s | title=%s", session_id, title[:80])
    except Exception as e:
        logger.warning("Could not fetch paper title | session=%s | %s", session_id, e)


async def upload_node_artifacts(session_id: str, result: dict, artifact_group: str) -> list[str]:
    output_dir = result.get("output_dir")
    if not output_dir or result.get("status") != "completed":
        return []
    local_dir = Path(output_dir)
    if not local_dir.exists():
        return []
    uploaded = await upload_artifacts(session_id, local_dir, artifact_group)
    if uploaded and session_id in sessions:
        target_key = "node2_result" if artifact_group == "implementation" else "node3_result"
        sessions[session_id].setdefault(target_key, {})["uploaded_files"] = uploaded
        await save_session(session_id, sessions[session_id])
    logger.info(
        "Artifact upload done | session=%s group=%s files=%s",
        session_id, artifact_group, uploaded,
    )
    return uploaded
