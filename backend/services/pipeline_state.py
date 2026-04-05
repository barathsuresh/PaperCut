from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from backend.schemas.api import PipelineResponse
from backend.schemas.contract import ArchitectureBlueprint
from backend.schemas.session import Node2StoredResult, Node3StoredResult, SessionData


def apply_node0_result(session_data: SessionData, result: Any) -> None:
    session_data["node0_result"] = {
        "result": result.result,
        "reason": result.reason,
    }


def apply_blueprint_result(
    session_data: SessionData,
    blueprint: ArchitectureBlueprint,
) -> None:
    session_data["node1_result"] = blueprint.model_dump()


def apply_node_output(
    session_data: SessionData,
    key: str,
    result: dict[str, Any],
) -> None:
    session_data[key] = result


def require_pipeline_prerequisite(
    session_data: SessionData,
    key: str,
    detail: str,
) -> None:
    if not session_data.get(key):
        raise HTTPException(status_code=400, detail=detail)


def node_error_status_code(result: Node2StoredResult | Node3StoredResult | dict[str, Any]) -> int:
    error_type = result.get("error_type", "")
    if error_type == "timeout":
        return 504
    if error_type == "auth":
        return 502
    return 500


def pipeline_outcome(
    scope_valid: bool,
    scope_reason: str,
    blueprint: ArchitectureBlueprint | None,
    scaffold: Node2StoredResult | None,
    cuda: Node3StoredResult | None,
) -> PipelineResponse:
    node2_status = scaffold.get("status", "error") if scaffold else "skipped"
    node2_error = scaffold.get("error") if scaffold else None
    node3_status = cuda.get("status", "error") if cuda else "skipped"
    node3_error = cuda.get("error") if cuda else None

    return PipelineResponse(
        session_id="",
        scope_valid=scope_valid,
        scope_reason=scope_reason,
        blueprint=blueprint,
        node2_status=node2_status,
        node2_files=scaffold.get("files") if scaffold else None,
        node3_status=node3_status,
        node3_stub_files=cuda.get("stub_files") if cuda else None,
        node3_bottlenecks=cuda.get("bottlenecks") if cuda else None,
        error=node2_error or node3_error,
    )


def pipeline_response(
    session_id: str,
    scope_valid: bool,
    scope_reason: str,
    blueprint: ArchitectureBlueprint | None,
    scaffold: Node2StoredResult | None,
    cuda: Node3StoredResult | None,
) -> PipelineResponse:
    response = pipeline_outcome(scope_valid, scope_reason, blueprint, scaffold, cuda)
    response.session_id = session_id
    return response
