import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from google.api_core.exceptions import GoogleAPICallError

from backend.chat.chat_handler import stream_chat
from backend.nodes.node0_validator import run_node0
from backend.nodes.node1_ingestor import ScopeRejectedError, run_node1
from backend.schemas.api import (
    ChatRequest,
    HealthResponse,
    Node0Response,
    Node1Response,
    NodeRunRequest,
    UploadResponse,
)
from backend.session_store import load_all_sessions, save_session
from backend.tools.pdf_loader import upload_pdf_bytes, upload_pdf_from_url

# In-memory session store: session_id -> session data dict
sessions: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load persisted sessions from GCS before accepting requests
    sessions.update(await load_all_sessions())
    yield


app = FastAPI(title="ArXiv Agent", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _persist(session_id: str) -> None:
    """Fire-and-forget GCS save — never blocks a response."""
    asyncio.create_task(save_session(session_id, sessions[session_id]))


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@app.post("/upload", response_model=UploadResponse)
async def upload(
    pdf_url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    if not pdf_url and not file:
        raise HTTPException(
            status_code=400, detail="Provide either pdf_url or a file upload."
        )

    session_id = str(uuid.uuid4())

    try:
        if pdf_url:
            gcs_uri = await upload_pdf_from_url(pdf_url, session_id)
        else:
            pdf_bytes = await file.read()
            gcs_uri = await upload_pdf_bytes(pdf_bytes, session_id)

        sessions[session_id] = {"gcs_uri": gcs_uri, "history": []}
        _persist(session_id)
        return UploadResponse(session_id=session_id, gcs_uri=gcs_uri)

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to fetch PDF from URL: {e.response.status_code} {e.response.reason_phrase}",
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=422, detail=f"Could not reach PDF URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run/node0", response_model=Node0Response)
async def run_node0_endpoint(body: NodeRunRequest):
    session_id = body.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        gcs_uri = sessions[session_id]["gcs_uri"]
        result = await run_node0(gcs_uri)
        sessions[session_id]["node0_result"] = {
            "result": result.result,
            "reason": result.reason,
        }
        _persist(session_id)
        return Node0Response(
            session_id=session_id, result=result.result, reason=result.reason
        )

    except GoogleAPICallError as e:
        raise HTTPException(status_code=502, detail=f"Vertex AI error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run/node1", response_model=Node1Response)
async def run_node1_endpoint(body: NodeRunRequest):
    session_id = body.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        gcs_uri = sessions[session_id]["gcs_uri"]
        node0 = sessions[session_id].get("node0_result")
        pre_validated = bool(node0 and node0.get("result") == "PASS")
        blueprint = await run_node1(gcs_uri, pre_validated=pre_validated)
        sessions[session_id]["node1_result"] = blueprint.model_dump()
        _persist(session_id)
        return Node1Response(session_id=session_id, blueprint=blueprint)

    except ScopeRejectedError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Paper rejected by scope check: {e.reason}",
        )
    except GoogleAPICallError as e:
        raise HTTPException(status_code=502, detail=f"Vertex AI error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(body: ChatRequest):
    session_id = body.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    if not sessions[session_id].get("node1_result"):
        raise HTTPException(
            status_code=400,
            detail="No paper has been processed for this session. Run /upload and /run/node1 first.",
        )

    message = body.message

    async def _stream_and_persist() -> AsyncGenerator[str, None]:
        accumulated: list[str] = []
        async for chunk in stream_chat(sessions[session_id], message):
            yield chunk
            try:
                data = json.loads(chunk[6:].strip())  # strip leading "data: "
                if data.get("type") == "token":
                    accumulated.append(data["text"])
                elif data.get("type") == "done" and accumulated:
                    full_response = "".join(accumulated)
                    sessions[session_id].setdefault("history", []).extend([
                        {"role": "user", "text": message},
                        {"role": "assistant", "text": full_response},
                    ])
                    _persist(session_id)
            except Exception:
                pass

    return StreamingResponse(
        _stream_and_persist(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
