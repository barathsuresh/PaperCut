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
