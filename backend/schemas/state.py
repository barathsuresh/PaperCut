from typing import Any, Dict, Optional
from typing_extensions import TypedDict

from backend.schemas.contract import ArchitectureBlueprint


class AgentState(TypedDict, total=False):
    session_id: str
    pdf_gcs_uri: str
    scope_valid: bool
    scope_reason: str
    blueprint: Optional[ArchitectureBlueprint]
    scaffold_code: Optional[str]
    cuda_blueprint: Optional[Dict[str, Any]]
    error: Optional[str]
