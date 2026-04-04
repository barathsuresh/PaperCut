import json

from backend.nodes.node0_validator import run_node0
from backend.schemas.contract import ArchitectureBlueprint
from backend.tools.gemini_client import get_pro_model, pdf_part_from_gcs, strip_markdown_fences


class ScopeRejectedError(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


EXTRACTION_PROMPT = """\
Extract the ML architecture from this research paper and return ONLY a JSON object \
(no markdown fences, no extra text) matching this exact schema:

{
  "model_type": "<transformer | cnn | rnn | gan | vae | diffusion>",
  "architecture": {
    "d_model": <integer: embedding or hidden state dimension>,
    "n_heads": <integer: attention or parallel heads; use 1 if not applicable>,
    "n_layers": <integer: number of layers, blocks, or stages>,
    "d_ff": <integer: feed-forward or expansion dimension; estimate d_model*4 if not stated>,
    "vocab_size": <integer: vocabulary size; use 32000 if not stated>,
    "max_seq_len": <integer: maximum sequence length; use 2048 if not stated>
  },
  "objective": "<training objective or loss function>",
  "key_operations": ["<op1>", "<op2>"],
  "math_notes": "<dimension constraints and mathematical invariants>"
}

Rules:
- model_type must be the closest enum value; use "rnn" for SSMs and recurrent-style architectures
- All architecture fields must be positive integers — infer or estimate from context if not explicit
- key_operations must contain at least one entry\
"""


def run_node1(gcs_uri: str) -> ArchitectureBlueprint:
    # Internal scope check — node1 always validates before extracting
    scope = run_node0(gcs_uri)
    if scope.result != "PASS":
        raise ScopeRejectedError(scope.reason)

    model = get_pro_model()
    pdf = pdf_part_from_gcs(gcs_uri)
    response = model.generate_content([pdf, EXTRACTION_PROMPT])
    text = strip_markdown_fences(response.text)
    data = json.loads(text)
    return ArchitectureBlueprint(**data)
