from typing import List, Literal

from pydantic import BaseModel, ConfigDict, field_validator


class ArchitectureParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    vocab_size: int
    max_seq_len: int

    @field_validator("d_model", "n_heads", "n_layers", "d_ff", "vocab_size", "max_seq_len")
    @classmethod
    def must_be_positive(cls, v: int, info) -> int:
        if v < 1:
            raise ValueError(f"{info.field_name} must be a positive integer, got {v}")
        return v


class ArchitectureBlueprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: Literal["transformer", "cnn", "rnn", "gan", "vae", "diffusion"]
    architecture: ArchitectureParams
    objective: str
    key_operations: List[str]
    math_notes: str
