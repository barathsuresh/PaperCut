from typing import Any, Dict, Optional
from typing_extensions import TypedDict

from backend.schemas.contract import ArchitectureBlueprint
from backend.schemas.session import Node2StoredResult, Node3StoredResult


class AgentState(TypedDict, total=False):
    session_id: str
    pdf_gcs_uri: str
    scope_valid: bool
    scope_reason: str
    blueprint: Optional[ArchitectureBlueprint]
    scaffold_code: Optional[Node2StoredResult]
    cuda_blueprint: Optional[Node3StoredResult]
    error: Optional[str]
