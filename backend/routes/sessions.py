import asyncio
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from backend.app_state import persist, sessions
from backend.schemas.api import (
    ArtifactContentResponse,
    ChatHistoryEntry,
    DeleteResponse,
    SessionArtifactsResponse,
    SessionDetail,
    SessionSummary,
)
from backend.services.session_views import (
    build_artifact_content_response,
    build_session_detail,
    build_session_summary,
    clean_history_entries,
    get_session_or_404,
    resolve_artifact_target,
)
from backend.session_store import delete_session as delete_session_blob
from backend.tools.artifact_store import delete_all_artifacts, download_artifact, list_artifacts

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/sessions", response_model=List[SessionSummary])
async def list_sessions():
    result = [build_session_summary(sid, data) for sid, data in sessions.items()]
    result.sort(key=lambda s: s.uploaded_at or "", reverse=True)
    return result


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str):
    return build_session_detail(session_id, get_session_or_404(sessions, session_id))


@router.get("/sessions/{session_id}/history", response_model=List[ChatHistoryEntry])
async def get_session_history(session_id: str):
    session_data = get_session_or_404(sessions, session_id)
    return clean_history_entries(session_data.get("history"))


@router.get("/sessions/{session_id}/artifacts", response_model=SessionArtifactsResponse)
async def get_session_artifacts(session_id: str):
    data = get_session_or_404(sessions, session_id)
    node2 = data.get("node2_result") or {}
    node3 = data.get("node3_result") or {}
    implementation_files = sorted((node2.get("files") or {}).keys())
    acceleration_files = sorted(node3.get("stub_files") or [])

    if not implementation_files:
        implementation_files = await list_artifacts(session_id, "implementation")
    if not acceleration_files:
        acceleration_files = await list_artifacts(session_id, "acceleration")

    return SessionArtifactsResponse(
        session_id=session_id,
        implementation_files=implementation_files,
        acceleration_files=acceleration_files,
    )


@router.patch("/sessions/{session_id}/name")
async def rename_session_endpoint(session_id: str, body: dict):
    get_session_or_404(sessions, session_id)
    name = (body.get("name") or "").strip()[:160]
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")
    sessions[session_id]["name"] = name
    persist(session_id)
    return {"session_id": session_id, "name": name}


@router.delete("/sessions/{session_id}", response_model=DeleteResponse)
async def delete_session_endpoint(session_id: str):
    get_session_or_404(sessions, session_id)
    sessions.pop(session_id)
    await asyncio.gather(
        delete_session_blob(session_id),
        delete_all_artifacts(session_id),
        return_exceptions=True,
    )
    logger.info("Session deleted | session=%s", session_id)
    return DeleteResponse(session_id=session_id, deleted=True, message="Session deleted.")


@router.get(
    "/sessions/{session_id}/artifacts/{artifact_group}/{file_name:path}",
    response_model=ArtifactContentResponse,
)
async def get_artifact_content(session_id: str, artifact_group: str, file_name: str):
    session_data = get_session_or_404(sessions, session_id)
    content: Optional[str] = None

    try:
        target = resolve_artifact_target(session_data, artifact_group, file_name)
        if target.exists() and target.is_file():
            try:
                content = await asyncio.to_thread(target.read_text)
            except UnicodeDecodeError:
                raise HTTPException(status_code=422, detail="Artifact file is not valid text.")
    except HTTPException as e:
        if e.status_code == 400:
            raise

    if content is None:
        content = await download_artifact(session_id, artifact_group, file_name)
        if content is None:
            raise HTTPException(status_code=404, detail="Artifact file not found.")

    return build_artifact_content_response(session_id, artifact_group, file_name, content)
