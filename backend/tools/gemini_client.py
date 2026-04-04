import re

import vertexai
from vertexai.generative_models import GenerativeModel, Part

_initialized = False


def _ensure_init() -> None:
    global _initialized
    if not _initialized:
        from backend import config
        vertexai.init(project=config.GCP_PROJECT_ID, location=config.GCP_REGION)
        _initialized = True


def get_flash_model() -> GenerativeModel:
    _ensure_init()
    from backend import config
    return GenerativeModel(config.GEMINI_FLASH_MODEL)


def get_pro_model() -> GenerativeModel:
    _ensure_init()
    from backend import config
    return GenerativeModel(config.GEMINI_PRO_MODEL)


def pdf_part_from_gcs(gcs_uri: str) -> Part:
    return Part.from_uri(gcs_uri, mime_type="application/pdf")


def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()
