import uuid
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from google.api_core.exceptions import GoogleAPICallError

from backend.chat.chat_handler import handle_chat
from backend.nodes.node0_validator import run_node0
from backend.nodes.node1_ingestor import ScopeRejectedError, run_node1
from backend.schemas.api import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    Node0Response,
    Node1Response,
    NodeRunRequest,
    UploadResponse,
)
from backend.tools.pdf_loader import upload_pdf_bytes, upload_pdf_from_url

app = FastAPI(title="ArXiv Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id -> session data dict
sessions: Dict[str, Dict[str, Any]] = {}


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
            gcs_uri = upload_pdf_from_url(pdf_url, session_id)
        else:
            pdf_bytes = await file.read()
            gcs_uri = upload_pdf_bytes(pdf_bytes, session_id)

        sessions[session_id] = {"gcs_uri": gcs_uri}
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
        result = run_node0(gcs_uri)
        sessions[session_id]["node0_result"] = {
            "result": result.result,
            "reason": result.reason,
        }
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
        blueprint = run_node1(gcs_uri)
        sessions[session_id]["node1_result"] = blueprint.model_dump()
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


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    session_id = body.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    if not sessions[session_id].get("node1_result"):
        raise HTTPException(
            status_code=400,
            detail="No paper has been processed for this session. Run /upload and /run/node1 first.",
        )

    try:
        response_text = handle_chat(sessions[session_id], body.message)
        return ChatResponse(session_id=session_id, response=response_text)

    except GoogleAPICallError as e:
        raise HTTPException(status_code=502, detail=f"Vertex AI error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
