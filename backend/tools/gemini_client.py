import re
from typing import Any

from google import genai
from google.genai import types

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        from backend import config
        _client = genai.Client(
            vertexai=True,
            project=config.GCP_PROJECT_ID,
            location=config.GCP_REGION,
        )
    return _client


class _AsyncModelWrapper:
    """Keeps the same .generate_content_async() interface as the old vertexai
    GenerativeModel so all call sites (node0, node1, chat) stay unchanged."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    async def generate_content_async(self, contents: Any, stream: bool = False):
        client = _get_client()
        if stream:
            async def _gen():
                stream = await client.aio.models.generate_content_stream(
                    model=self._model_name, contents=contents
                )
                async for chunk in stream:
                    yield chunk
            return _gen()
        return await client.aio.models.generate_content(
            model=self._model_name, contents=contents
        )


def get_flash_model() -> _AsyncModelWrapper:
    from backend import config
    return _AsyncModelWrapper(config.GEMINI_FLASH_MODEL)


def get_pro_model() -> _AsyncModelWrapper:
    from backend import config
    return _AsyncModelWrapper(config.GEMINI_PRO_MODEL)


def pdf_part_from_gcs(gcs_uri: str) -> types.Part:
    return types.Part.from_uri(file_uri=gcs_uri, mime_type="application/pdf")


def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()
