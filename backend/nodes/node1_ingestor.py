import json
import logging

from pydantic import ValidationError

from backend.nodes.node0_validator import run_node0
from backend.schemas.contract import ArchitectureBlueprint
from backend.tools.gemini_client import get_pro_model, pdf_part_from_gcs, strip_markdown_fences
from backend.tools.model_response import ModelResponseError

logger = logging.getLogger(__name__)


class ScopeRejectedError(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


# Used when scope has NOT been pre-validated — model may return an error object
_EXTRACTION_PROMPT_GATED = """\
You are extracting ML architecture details from a research paper.

FIRST, verify this paper proposes an original ML/AI model with a concrete architecture \
(layers, dimensions, training objective). If it does NOT — for example it is a biology, \
medical, or social-science paper, or ML is only used as an off-the-shelf tool — return \
this exact JSON and nothing else:

{
  "error": true,
  "reason": "one sentence explaining why no ML architecture can be extracted"
}

If the paper DOES describe an original ML architecture, return ONLY this JSON \
(no markdown fences, no extra text):

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
- key_operations must contain at least one entry
- Do NOT hallucinate values for a non-ML paper — return the error object instead\
"""

# Used when node0 already ran and returned PASS — must produce a blueprint, no error escape
_EXTRACTION_PROMPT_FORCED = """\
This paper has already passed scope validation as an ML/AI research paper. \
Extract its architecture and return ONLY this JSON (no markdown fences, no extra text):

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


async def run_node1(gcs_uri: str, pre_validated: bool = False) -> ArchitectureBlueprint:
    logger.info(
        "Node 1 — extraction start | gcs_uri=%s | pre_validated=%s", gcs_uri, pre_validated
    )

    if not pre_validated:
        logger.debug("Node 1 — running internal Node 0 scope check (not pre-validated)")
        scope = await run_node0(gcs_uri)
        if scope.result != "PASS":
            logger.warning("Node 1 — internal scope check FAILED: %s", scope.reason)
            raise ScopeRejectedError(scope.reason)
        logger.info("Node 1 — internal scope check PASSED")

    prompt_name = "FORCED" if pre_validated else "GATED"
    logger.debug("Node 1 — using %s extraction prompt", prompt_name)

    model = get_pro_model()
    pdf = pdf_part_from_gcs(gcs_uri)
    prompt = _EXTRACTION_PROMPT_FORCED if pre_validated else _EXTRACTION_PROMPT_GATED
    response = await model.generate_content_async([pdf, prompt])

    raw_text = getattr(response, "text", None)
    logger.debug("Node 1 — raw model response: %s", raw_text)

    if raw_text is None:
        raise ModelResponseError(
            "node1: model returned no text (possible safety block or quota exhaustion)"
        )

    cleaned = strip_markdown_fences(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("Node 1 — JSON parse failed | preview=%r | error=%s", cleaned[:300], exc)
        raise ModelResponseError(f"node1: model response was not valid JSON — {exc}") from exc

    # Check error escape on both paths (model may still refuse on forced prompt)
    if data.get("error"):
        reason = data.get("reason", "Paper does not contain an extractable ML architecture.")
        if pre_validated:
            logger.error("Node 1 — forced path returned error escape: %s", reason)
        else:
            logger.warning("Node 1 — model returned error escape: %s", reason)
        raise ScopeRejectedError(reason)

    try:
        blueprint = ArchitectureBlueprint(**data)
    except ValidationError as exc:
        logger.warning("Node 1 — schema validation failed | data=%r | error=%s", data, exc)
        raise ModelResponseError(f"node1: architecture blueprint schema invalid — {exc}") from exc

    logger.info(
        "Node 1 — extraction complete | model_type=%s | n_layers=%s | d_model=%s",
        blueprint.model_type,
        blueprint.architecture.n_layers,
        blueprint.architecture.d_model,
    )
    return blueprint
