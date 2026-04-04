from typing import Optional
from pydantic import BaseModel

from backend.schemas.contract import ArchitectureBlueprint


class UploadResponse(BaseModel):
    session_id: str
    gcs_uri: str
    message: str = "Upload successful"


class NodeRunRequest(BaseModel):
    session_id: str


class Node0Response(BaseModel):
    session_id: str
    result: str  # "PASS" or "FAIL"
    reason: str


class Node1Response(BaseModel):
    session_id: str
    blueprint: ArchitectureBlueprint


class Node2Response(BaseModel):
    session_id: str
    status: str
    output_dir: Optional[str] = None
    files: Optional[dict] = None
    error: Optional[str] = None


class Node3Response(BaseModel):
    session_id: str
    status: str
    output_dir: Optional[str] = None
    stub_files: Optional[list] = None
    bottlenecks: Optional[list] = None
    meta: Optional[dict] = None
    error: Optional[str] = None


class PipelineResponse(BaseModel):
    session_id: str
    scope_valid: bool
    scope_reason: Optional[str] = None
    blueprint: Optional[ArchitectureBlueprint] = None
    node2_status: str
    node2_files: Optional[dict] = None
    node3_status: str
    node3_stub_files: Optional[list] = None
    node3_bottlenecks: Optional[list] = None
    error: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
