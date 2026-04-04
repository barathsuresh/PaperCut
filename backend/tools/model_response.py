"""
Shared helper for safely parsing LLM text responses into Pydantic schemas.
Centralises JSON decoding, fence stripping, None-text, and ValidationError
handling so every node gets consistent, descriptive error messages.
"""
from __future__ import annotations

import json
import logging
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ModelResponseError(ValueError):
    """Raised when the LLM response cannot be parsed into the expected schema."""


def parse_model_json(text: str | None, schema: Type[T], node_name: str) -> T:
    """
    Strip markdown fences, JSON-decode, and validate against *schema*.

    Args:
        text:      Raw text from model.generate_content_async() — may be None.
        schema:    Pydantic model class to validate into.
        node_name: Label used in error messages (e.g. "node0", "node1").

    Returns:
        A validated instance of *schema*.

    Raises:
        ModelResponseError: on None text, empty response, JSON parse failure,
                            or Pydantic validation failure.
    """
    if text is None:
        raise ModelResponseError(
            f"{node_name}: model returned no text "
            "(possible safety block or quota exhaustion)"
        )

    from backend.tools.gemini_client import strip_markdown_fences
    cleaned = strip_markdown_fences(text)

    if not cleaned:
        raise ModelResponseError(
            f"{node_name}: model returned an empty response after stripping fences"
        )

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning(
            "%s JSON parse failed | preview=%r | error=%s",
            node_name, cleaned[:300], exc,
        )
        raise ModelResponseError(
            f"{node_name}: model response was not valid JSON — {exc}"
        ) from exc

    try:
        return schema(**data)
    except ValidationError as exc:
        logger.warning(
            "%s schema validation failed | data=%r | error=%s",
            node_name, data, exc,
        )
        raise ModelResponseError(
            f"{node_name}: model response did not match expected schema — {exc}"
        ) from exc
