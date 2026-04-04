import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List

from backend.tools.gemini_client import get_pro_model

_SYSTEM_PREFIX = """\
You are an expert ML research assistant. You have been given the context of a processed research paper.
Use this context to answer the user's question accurately and helpfully.

Paper Context:
{context}
"""

# Keep at most this many past exchanges (user+assistant pairs) in the prompt
_MAX_HISTORY_PAIRS = 10


def _build_context(session_data: Dict[str, Any]) -> str:
    context_parts = []

    node1_result = session_data.get("node1_result")
    if node1_result:
        context_parts.append(
            f"Architecture Blueprint:\n{json.dumps(node1_result, indent=2)}"
        )

    node2_result = session_data.get("node2_result")
    if node2_result and node2_result.get("status") != "not_implemented":
        context_parts.append(f"Scaffold Code:\n{node2_result}")

    node3_result = session_data.get("node3_result")
    if node3_result and node3_result.get("status") != "not_implemented":
        context_parts.append(f"CUDA Blueprint:\n{json.dumps(node3_result, indent=2)}")

    return "\n\n".join(context_parts)


def _build_history_block(history: List[Dict[str, str]]) -> str:
    if not history:
        return ""
    # Trim to last N pairs
    pairs = history[- _MAX_HISTORY_PAIRS * 2 :]
    lines = ["Previous conversation:"]
    for entry in pairs:
        role = "User" if entry["role"] == "user" else "Assistant"
        lines.append(f"{role}: {entry['text']}")
    return "\n".join(lines)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def stream_chat(
    session_data: Dict[str, Any], message: str
) -> AsyncGenerator[str, None]:
    yield _sse({"type": "status", "text": "Thinking..."})
    await asyncio.sleep(0)

    context = _build_context(session_data)
    history = session_data.get("history", [])
    history_block = _build_history_block(history)

    yield _sse({"type": "status", "text": "Reading paper context..."})
    await asyncio.sleep(0)

    prompt_parts = [_SYSTEM_PREFIX.format(context=context)]
    if history_block:
        prompt_parts.append(history_block)
    prompt_parts.append(f"User: {message}")
    prompt = "\n\n".join(prompt_parts)

    model = get_pro_model()

    yield _sse({"type": "status", "text": "Generating response..."})
    await asyncio.sleep(0)

    async for chunk in await model.generate_content_async(prompt, stream=True):
        if chunk.text:
            yield _sse({"type": "token", "text": chunk.text})

    yield _sse({"type": "done", "text": ""})
