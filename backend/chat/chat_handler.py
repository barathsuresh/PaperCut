import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

from google.api_core.exceptions import GoogleAPICallError
from google.genai.errors import ClientError as GeminiClientError

from backend.tools.gemini_client import get_pro_model
from backend.tools.artifact_store import download_artifact

logger = logging.getLogger(__name__)

_SYSTEM_PREFIX = """\
You are an expert ML research assistant with full access to a processed research paper and its \
generated implementation. You can answer questions about the paper's theory, architecture, \
mathematical details, the generated PyTorch code, CUDA kernels, and how to train or extend the model.

Paper Context:
{context}
"""

# Keep at most this many past exchanges (user+assistant pairs) in the prompt
_MAX_HISTORY_PAIRS = 10

# Max characters per generated file included in context (avoids token explosion)
_MAX_FILE_CHARS = 3000

# PyTorch scaffold files to include in chat context (in order)
_SCAFFOLD_FILES = ("model.py", "train.py", "dataset.py", "config.yaml")


def _read_file_sync(path: Path) -> str:
    """Read a file and truncate to _MAX_FILE_CHARS."""
    try:
        text = path.read_text()
        if len(text) > _MAX_FILE_CHARS:
            text = text[:_MAX_FILE_CHARS] + f"\n... [truncated, {len(text)} chars total]"
        return text
    except Exception:
        return ""


def _truncate_text(text: str) -> str:
    if len(text) > _MAX_FILE_CHARS:
        return text[:_MAX_FILE_CHARS] + f"\n... [truncated, {len(text)} chars total]"
    return text


async def _load_artifact_text(
    session_id: str,
    artifact_group: str,
    file_name: str,
    output_dir: Path | None = None,
) -> str:
    content = await download_artifact(session_id, artifact_group, file_name)
    if content:
        return _truncate_text(content)

    if output_dir is None:
        return ""

    fpath = output_dir / file_name
    if fpath.exists():
        return await asyncio.to_thread(_read_file_sync, fpath)
    return ""


async def _build_context(session_id: str, session_data: Dict[str, Any]) -> str:
    context_parts = []

    # --- Node 1: architecture blueprint ---
    node1_result = session_data.get("node1_result")
    if node1_result:
        context_parts.append(
            f"Architecture Blueprint:\n{json.dumps(node1_result, indent=2)}"
        )

    # --- Node 2: actual generated PyTorch code ---
    node2_result = session_data.get("node2_result")
    if node2_result and node2_result.get("status") == "completed":
        output_dir = Path(node2_result["output_dir"]) if node2_result.get("output_dir") else None
        scaffold_parts = []
        for fname in _SCAFFOLD_FILES:
            content = await _load_artifact_text(session_id, "implementation", fname, output_dir)
            if content:
                scaffold_parts.append(f"### {fname}\n```\n{content}\n```")
        if scaffold_parts:
            context_parts.append("Generated PyTorch Scaffold:\n" + "\n\n".join(scaffold_parts))
            logger.debug("Chat context — included %d scaffold files", len(scaffold_parts))

    # --- Node 3: CUDA stubs + bottleneck analysis ---
    node3_result = session_data.get("node3_result")
    if node3_result and node3_result.get("status") == "completed":
        output_dir = Path(node3_result["output_dir"]) if node3_result.get("output_dir") else None
        cuda_parts = []

        # Bottleneck summary
        bottlenecks = node3_result.get("bottlenecks")
        if bottlenecks:
            cuda_parts.append(
                f"Top bottlenecks identified:\n{json.dumps(bottlenecks, indent=2)}"
            )

        # CUDA stub files
        for fname in node3_result.get("stub_files", []):
            content = await _load_artifact_text(session_id, "acceleration", fname, output_dir)
            if content:
                cuda_parts.append(f"### {fname}\n```cuda\n{content}\n```")

        if cuda_parts:
            context_parts.append("Generated CUDA Stubs:\n" + "\n\n".join(cuda_parts))
            logger.debug("Chat context — included CUDA stubs + bottlenecks")

    return "\n\n".join(context_parts)


def _build_history_block(history: List[Dict[str, str]]) -> str:
    if not history:
        return ""
    pairs = history[-_MAX_HISTORY_PAIRS * 2:]
    lines = ["Previous conversation:"]
    for entry in pairs:
        try:
            role = "User" if entry.get("role") == "user" else "Assistant"
            text = entry.get("text", "")
            if text:
                lines.append(f"{role}: {text}")
        except (AttributeError, TypeError):
            logger.warning("Skipping malformed history entry: %r", entry)
    return "\n".join(lines)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def stream_chat(
    session_id: str, session_data: Dict[str, Any], message: str
) -> AsyncGenerator[str, None]:
    history = session_data.get("history", [])
    context_keys = [k for k in ("node1_result", "node2_result", "node3_result") if session_data.get(k)]
    logger.info(
        "Chat stream start | history_pairs=%d | context_keys=%s | message_len=%d",
        len(history) // 2,
        context_keys,
        len(message),
    )

    yield _sse({"type": "status", "text": "Thinking..."})
    await asyncio.sleep(0)

    context = await _build_context(session_id, session_data)
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

    logger.debug("Chat — starting token stream")
    token_count = 0
    stream_error = False
    try:
        async for chunk in await model.generate_content_async(prompt, stream=True):
            if chunk.text:
                token_count += 1
                yield _sse({"type": "token", "text": chunk.text})
            elif chunk.text is None:
                # Safety block or empty chunk — check finish reason if available
                finish = getattr(chunk, "finish_reason", None) or getattr(
                    getattr(chunk, "candidates", [None])[0], "finish_reason", None
                    ) if getattr(chunk, "candidates", None) else None
                if finish and str(finish) not in ("STOP", "1", "FinishReason.STOP"):
                    logger.warning("Chat — response blocked | finish_reason=%s", finish)
                    yield _sse({
                        "type": "error",
                        "text": "Response blocked by content filter. Please rephrase your question.",
                    })
                    stream_error = True
                    break
    except GeminiClientError as e:
        stream_error = True
        if e.code == 429:
            logger.warning("Chat — Gemini rate limit (429)")
            yield _sse({"type": "error", "text": "Rate limit reached. Please wait a moment and try again."})
        else:
            logger.error("Chat — Gemini error | code=%s", e.code)
            yield _sse({"type": "error", "text": f"Model error ({e.code}). Please try again."})
    except GoogleAPICallError as e:
        stream_error = True
        logger.error("Chat — Vertex API error mid-stream: %s", e)
        yield _sse({"type": "error", "text": "Infrastructure error. Please try again."})
    except Exception as e:
        stream_error = True
        logger.error("Chat — unexpected error during stream: %s", e, exc_info=True)
        yield _sse({"type": "error", "text": "Unexpected error. Please try again."})

    if not stream_error:
        logger.info("Chat stream complete | chunks_yielded=%d", token_count)
    yield _sse({"type": "done", "text": ""})
