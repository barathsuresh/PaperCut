from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Callable

from backend.app_state import sessions, upload_node_artifacts
from backend.schemas.session import Node2StoredResult, Node3StoredResult, SessionData
from backend.services.pipeline_state import apply_blueprint_result, apply_node_output
from backend.session_store import save_session


def sse(event_type: str, payload: dict[str, Any]) -> str:
    return f"data: {json.dumps({'type': event_type, **payload})}\n\n"


def new_session_payload(gcs_uri: str) -> SessionData:
    return {
        "gcs_uri": gcs_uri,
        "name": "Untitled Paper",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "history": [],
    }


async def run_codegen_node_for_session(
    *,
    session_id: str,
    session_data: SessionData,
    input_payload: Any,
    runner: Callable[[Any, str | None], dict[str, Any]],
    session_key: str,
    artifact_group: str,
) -> Node2StoredResult | Node3StoredResult:
    result = await asyncio.to_thread(runner, input_payload, session_id)
    apply_node_output(session_data, session_key, result)
    await save_session(session_id, session_data)
    if result.get("status") == "completed":
        await upload_node_artifacts(session_id, result, artifact_group)
    return result


async def run_stream_codegen_node(
    *,
    session_id: str,
    input_payload: Any,
    runner: Callable[[Any, str | None], dict[str, Any]],
    session_key: str,
    artifact_group: str,
) -> Node2StoredResult | Node3StoredResult:
    session_data = sessions[session_id]
    result = await asyncio.to_thread(runner, input_payload, session_id)
    apply_node_output(session_data, session_key, result)
    asyncio.create_task(save_session(session_id, session_data))
    if result.get("status") == "completed":
        await upload_node_artifacts(session_id, result, artifact_group)
    return result


def persist_graph_results(
    session_data: SessionData,
    final_state: dict[str, Any],
) -> tuple[bool, str, Any, Node2StoredResult | None, Node3StoredResult | None]:
    scope_valid: bool = final_state.get("scope_valid", False)
    scope_reason: str = final_state.get("scope_reason", "")
    session_data["node0_result"] = {
        "result": "PASS" if scope_valid else "FAIL",
        "reason": scope_reason,
    }

    blueprint = final_state.get("blueprint")
    if blueprint:
        apply_blueprint_result(session_data, blueprint)

    scaffold = final_state.get("scaffold_code")
    if scaffold:
        apply_node_output(session_data, "node2_result", scaffold)

    cuda = final_state.get("cuda_blueprint")
    if cuda:
        apply_node_output(session_data, "node3_result", cuda)

    return scope_valid, scope_reason, blueprint, scaffold, cuda
