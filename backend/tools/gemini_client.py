import asyncio
import logging
import re
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import ClientError as GeminiClientError

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF = [5, 10, 20]  # seconds per attempt

_client: genai.Client | None = None
_client_lock = asyncio.Lock()


async def _get_client() -> genai.Client:
    global _client
    if _client is None:
        async with _client_lock:
            if _client is None:  # double-checked inside the lock
                from backend import config
                logger.info(
                    "Initialising Gemini client | project=%s | region=%s",
                    config.GCP_PROJECT_ID,
                    config.GCP_REGION,
                )
                _client = genai.Client(
                    vertexai=True,
                    project=config.GCP_PROJECT_ID,
                    location=config.GCP_REGION,
                )
    return _client


class _AsyncModelWrapper:
    """Keeps the same .generate_content_async() interface so all call sites stay unchanged."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    async def generate_content_async(self, contents: Any, stream: bool = False):
        client = await _get_client()
        logger.debug("Gemini call | model=%s | stream=%s", self._model_name, stream)

        if stream:
            async def _gen():
                for attempt in range(_MAX_RETRIES + 1):
                    try:
                        response_stream = await client.aio.models.generate_content_stream(
                            model=self._model_name, contents=contents
                        )
                        async for chunk in response_stream:
                            yield chunk
                        return  # success
                    except GeminiClientError as e:
                        if e.code == 429 and attempt < _MAX_RETRIES:
                            wait = _RETRY_BACKOFF[attempt]
                            logger.warning(
                                "Gemini 429 on stream | model=%s | attempt=%d/%d | retrying in %ds",
                                self._model_name, attempt + 1, _MAX_RETRIES, wait,
                            )
                            await asyncio.sleep(wait)
                        else:
                            raise
            return _gen()

        # Non-streaming with retry
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await client.aio.models.generate_content(
                    model=self._model_name, contents=contents
                )
                logger.debug("Gemini response received | model=%s", self._model_name)
                return response
            except GeminiClientError as e:
                if e.code == 429 and attempt < _MAX_RETRIES:
                    wait = _RETRY_BACKOFF[attempt]
                    logger.warning(
                        "Gemini 429 | model=%s | attempt=%d/%d | retrying in %ds",
                        self._model_name, attempt + 1, _MAX_RETRIES, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise


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
    # Case-insensitive: handles ```json, ```JSON, ```
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()
