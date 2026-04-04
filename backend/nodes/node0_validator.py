import json

from backend.schemas.validator import ScopeValidationResult
from backend.tools.gemini_client import get_flash_model, pdf_part_from_gcs, strip_markdown_fences

SCOPE_CHECK_PROMPT = """\
Analyze this research paper PDF and determine if it is a machine learning or AI research paper.

Respond ONLY with a JSON object in this exact format (no markdown fences, no extra text):
{
  "result": "PASS",
  "reason": "brief explanation"
}

Use "PASS" if the paper is about ML/AI research, "FAIL" otherwise.\
"""


def run_node0(gcs_uri: str) -> ScopeValidationResult:
    model = get_flash_model()
    pdf = pdf_part_from_gcs(gcs_uri)
    response = model.generate_content([pdf, SCOPE_CHECK_PROMPT])
    text = strip_markdown_fences(response.text)
    data = json.loads(text)
    return ScopeValidationResult(**data)
