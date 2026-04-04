import json

from backend.schemas.validator import ScopeValidationResult
from backend.tools.gemini_client import get_flash_model, pdf_part_from_gcs, strip_markdown_fences

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
    model = get_flash_model()
    pdf = pdf_part_from_gcs(gcs_uri)
    response = await model.generate_content_async([pdf, SCOPE_CHECK_PROMPT])
    text = strip_markdown_fences(response.text)
    data = json.loads(text)
    return ScopeValidationResult(**data)
