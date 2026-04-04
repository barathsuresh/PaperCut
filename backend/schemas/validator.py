from pydantic import BaseModel


class ScopeValidationResult(BaseModel):
    result: str  # "PASS" or "FAIL"
    reason: str
