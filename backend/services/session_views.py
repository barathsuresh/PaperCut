from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from backend.schemas.api import (
    ArtifactContentResponse,
    ChatHistoryEntry,
    SessionDetail,
    SessionSummary,
)
from backend.schemas.session import SessionData, SessionHistoryEntry, SessionStore


def get_session_or_404(
    sessions: SessionStore,
    session_id: str,
) -> SessionData:
    data = sessions.get(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return data


def scope_valid_from_session(session_data: SessionData) -> bool | None:
    node0 = session_data.get("node0_result") or {}
    return node0.get("result") == "PASS" if node0 else None


def build_session_summary(session_id: str, session_data: SessionData) -> SessionSummary:
    node1 = session_data.get("node1_result") or {}
    return SessionSummary(
        session_id=session_id,
        name=session_data.get("name", "Untitled Paper"),
        uploaded_at=session_data.get("uploaded_at"),
        scope_valid=scope_valid_from_session(session_data),
        model_type=node1.get("model_type"),
        node2_status=(session_data.get("node2_result") or {}).get("status"),
        node3_status=(session_data.get("node3_result") or {}).get("status"),
    )


def build_session_detail(session_id: str, session_data: SessionData) -> SessionDetail:
    node0 = session_data.get("node0_result") or {}
    node1 = session_data.get("node1_result") or {}
    return SessionDetail(
        session_id=session_id,
        name=session_data.get("name", "Untitled Paper"),
        gcs_uri=session_data.get("gcs_uri", ""),
        uploaded_at=session_data.get("uploaded_at"),
        scope_valid=scope_valid_from_session(session_data),
        scope_reason=node0.get("reason"),
        model_type=node1.get("model_type"),
        node2_status=(session_data.get("node2_result") or {}).get("status"),
        node3_status=(session_data.get("node3_result") or {}).get("status"),
        has_chat_history=bool(session_data.get("history")),
    )


def clean_history_entries(history: list[SessionHistoryEntry] | None) -> list[ChatHistoryEntry]:
    cleaned_history: list[ChatHistoryEntry] = []
    for entry in history or []:
        role = entry.get("role")
        text = entry.get("text")
        if role in {"user", "assistant"} and isinstance(text, str):
            cleaned_history.append(ChatHistoryEntry(role=role, text=text))
    return cleaned_history


def infer_language(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    return {
        ".py": "python",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".cu": "cpp",
        ".cuh": "cpp",
        ".cpp": "cpp",
        ".txt": "text",
        ".md": "markdown",
    }.get(suffix, "text")


def resolve_artifact_target(
    session_data: SessionData,
    artifact_group: str,
    file_name: str,
) -> Path:
    if artifact_group == "implementation":
        result = session_data.get("node2_result") or {}
    elif artifact_group == "acceleration":
        result = session_data.get("node3_result") or {}
    else:
        raise HTTPException(status_code=404, detail="Artifact group not found.")

    output_dir = result.get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=404, detail="Artifacts not available for this session.")

    root = Path(output_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=404, detail="Artifact directory not found.")

    target = (root / file_name).resolve()
    if target == root or root not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid artifact path.")
    return target


def build_artifact_content_response(
    session_id: str,
    artifact_group: str,
    file_name: str,
    content: str,
) -> ArtifactContentResponse:
    return ArtifactContentResponse(
        session_id=session_id,
        artifact_group=artifact_group,
        file_name=file_name,
        language=infer_language(file_name),
        content=content,
    )
