from pydantic import BaseModel, field_validator


class ScopeValidationResult(BaseModel):
    result: str  # "PASS" or "FAIL"
    reason: str

    @field_validator("result")
    @classmethod
    def normalise_result(cls, v: str) -> str:
        normalised = v.strip().upper()
        if normalised not in ("PASS", "FAIL"):
            raise ValueError(f"result must be 'PASS' or 'FAIL', got: {v!r}")
        return normalised
