import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.app_state import persist, sessions
from backend.chat.chat_handler import stream_chat
from backend.schemas.api import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat")
async def chat(body: ChatRequest):
    session_id = body.session_id
    logger.info("POST /chat | session=%s | message_len=%d", session_id, len(body.message))
    if session_id not in sessions:
        logger.warning("Session not found | session=%s", session_id)
        raise HTTPException(status_code=404, detail="Session not found.")

    if not sessions[session_id].get("node1_result"):
        logger.warning("Chat attempted before node1 | session=%s", session_id)
        raise HTTPException(
            status_code=400,
            detail="No paper has been processed for this session. Run /upload and /run/node1 first.",
        )

    message = body.message

    async def _stream_and_persist() -> AsyncGenerator[str, None]:
        accumulated: list[str] = []
        had_error = False
        async for chunk in stream_chat(session_id, sessions[session_id], message):
            yield chunk
            try:
                data = json.loads(chunk[6:].strip())
                if data.get("type") == "token":
                    accumulated.append(data["text"])
                elif data.get("type") == "error":
                    had_error = True
                elif data.get("type") == "done" and accumulated and not had_error:
                    full_response = "".join(accumulated)
                    sessions[session_id].setdefault("history", []).extend([
                        {"role": "user", "text": message},
                        {"role": "assistant", "text": full_response},
                    ])
                    persist(session_id)
            except Exception as exc:
                logger.warning(
                    "SSE parse error in _stream_and_persist | session=%s | error=%s",
                    session_id, exc,
                )

    return StreamingResponse(
        _stream_and_persist(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
