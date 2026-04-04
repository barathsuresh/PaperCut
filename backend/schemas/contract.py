from typing import List, Literal
from pydantic import BaseModel, ConfigDict


class ArchitectureParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    vocab_size: int
    max_seq_len: int


class ArchitectureBlueprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: Literal["transformer", "cnn", "rnn", "gan", "vae", "diffusion"]
    architecture: ArchitectureParams
    objective: str
    key_operations: List[str]
    math_notes: str
