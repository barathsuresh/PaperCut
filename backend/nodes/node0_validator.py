import logging

from backend.schemas.validator import ScopeValidationResult
from backend.tools.gemini_client import get_flash_model, pdf_part_from_gcs
from backend.tools.model_response import ModelResponseError, parse_model_json

logger = logging.getLogger(__name__)

SCOPE_CHECK_PROMPT = """\
You are a strict classifier. Read this paper and decide if its PRIMARY contribution is \
a new machine learning or AI model, architecture, or learning algorithm.

PASS criteria (ALL must be true):
- The paper proposes or significantly extends an ML/AI model or architecture
- The paper includes training, loss functions, or learned parameters
- The core contribution is a neural network, learning system, or ML methodology

FAIL criteria (ANY one is enough to fail):
- The paper is from biology, medicine, chemistry, physics, social science, or other non-ML field
- ML/AI is only used as a tool or baseline, not the primary contribution
- The paper is a survey, review, or benchmark without a new model
- No neural network or learning algorithm is proposed

Respond ONLY with a JSON object (no markdown, no extra text):
{
  "result": "PASS" or "FAIL",
  "reason": "one sentence explaining the decision"
}

Be strict. When in doubt, return FAIL.\
"""


async def run_node0(gcs_uri: str) -> ScopeValidationResult:
    logger.info("Node 0 — scope validation start | gcs_uri=%s", gcs_uri)
    model = get_flash_model()
    pdf = pdf_part_from_gcs(gcs_uri)
    response = await model.generate_content_async([pdf, SCOPE_CHECK_PROMPT])
    logger.debug("Node 0 — raw model response: %s", getattr(response, "text", None))
    result = parse_model_json(response.text, ScopeValidationResult, "node0")
    logger.info("Node 0 — result=%s | reason=%s", result.result, result.reason)
    return result
