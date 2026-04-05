import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from google.api_core.exceptions import GoogleAPICallError
from google.genai.errors import ClientError as GeminiClientError

from backend import config
from backend.chat.chat_handler import stream_chat
from backend.nodes.node0_validator import run_node0
from backend.nodes.node1_ingestor import ScopeRejectedError, run_node1
from backend.nodes.node2_client import run_node2
from backend.nodes.node3_client import run_node3
from backend.schemas.api import (
    ArtifactContentResponse,
    ChatHistoryEntry,
    ChatRequest,
    DeleteResponse,
    HealthResponse,
    Node0Response,
    Node1Response,
    Node2Response,
    Node3Response,
    NodeRunRequest,
    PipelineResponse,
    SessionDetail,
    SessionArtifactsResponse,
    SessionSummary,
    UploadResponse,
)
from backend.session_store import delete_session as delete_session_blob, load_all_sessions, save_session
from backend.tools.artifact_store import delete_all_artifacts, download_artifact, upload_artifacts
from backend.tools.gemini_client import get_flash_model, pdf_part_from_gcs
from backend.tools.model_response import ModelResponseError
from backend.tools.pdf_loader import upload_pdf_bytes, upload_pdf_from_url

# 50 MB limit for direct file uploads
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# In-memory session store: session_id -> session data dict
sessions: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate required config before accepting any traffic
    try:
        config.validate()
    except RuntimeError as e:
        logger.critical("Configuration error — server cannot start: %s", e)
        raise

    # Validate NAT config at startup so missing NVIDIA keys fail fast
    try:
        import nat.nat_config  # noqa: F401
        logger.info("NAT config validated OK")
    except RuntimeError as e:
        logger.warning("NAT config missing — Node 2/3 will fail at runtime: %s", e)

    logger.info("Server starting — loading sessions from GCS")
    sessions.update(await load_all_sessions())
    logger.info("Server ready | sessions_loaded=%d", len(sessions))
    yield
    logger.info("Server shutting down")


def _raise_gemini_error(e: GeminiClientError) -> None:
    """Map google-genai ClientError to the right HTTP status."""
    if e.code == 429:
        raise HTTPException(
            status_code=429,
            detail="Gemini rate limit hit. Wait a moment and try again.",
        )
    raise HTTPException(status_code=502, detail=f"Gemini API error: {e.message}")


app = FastAPI(title="ArXiv Agent", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all — prevents raw tracebacks from leaking to clients."""
    logger.error(
        "Unhandled exception | path=%s | method=%s | error=%s",
        request.url.path,
        request.method,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


def _persist(session_id: str) -> None:
    """Fire-and-forget GCS save — never blocks a response."""
    data = sessions.get(session_id)
    if data is None:
        logger.warning("_persist called for unknown session=%s — skipping", session_id)
        return
    asyncio.create_task(save_session(session_id, data))


def _resolve_artifact_root(session_data: Dict[str, Any], artifact_group: str) -> Path:
    if artifact_group == "implementation":
        result = session_data.get("node2_result") or {}
    elif artifact_group == "acceleration":
        result = session_data.get("node3_result") or {}
    else:
        raise HTTPException(status_code=404, detail="Artifact group not found.")

    output_dir = result.get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=404, detail="Artifacts not available for this session.")

    root = Path(output_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=404, detail="Artifact directory not found.")
    return root


def _infer_language(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    return {
        ".py": "python",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".cu": "cpp",
        ".cuh": "cpp",
        ".cpp": "cpp",
        ".txt": "text",
        ".md": "markdown",
    }.get(suffix, "text")


async def _fetch_and_store_title(session_id: str, gcs_uri: str) -> None:
    """
    Background task: extract the paper title via a single Gemini Flash call
    and update the session name. Silently no-ops on any failure.
    """
    try:
        model = get_flash_model()
        response = await model.generate_content_async(
            [pdf_part_from_gcs(gcs_uri), "Return only the paper title, nothing else. No quotes, no punctuation beyond the title itself."]
        )
        title = (response.text or "").strip().strip('"').strip("'")
        if title and session_id in sessions:
            sessions[session_id]["name"] = title[:160]
            await save_session(session_id, sessions[session_id])
            logger.info("Session title set | session=%s | title=%s", session_id, title[:80])
    except Exception as e:
        logger.warning("Could not fetch paper title | session=%s | %s", session_id, e)


async def _upload_node_artifacts(session_id: str, result: dict, artifact_group: str) -> None:
    """Background task: upload generated files to GCS after a node completes."""
    output_dir = result.get("output_dir")
    if not output_dir or result.get("status") != "completed":
        return
    local_dir = Path(output_dir)
    if not local_dir.exists():
        return
    uploaded = await upload_artifacts(session_id, local_dir, artifact_group)
    logger.info(
        "Artifact upload done | session=%s group=%s files=%s",
        session_id, artifact_group, uploaded,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@app.get("/sessions", response_model=List[SessionSummary])
async def list_sessions():
    """Return all sessions sorted newest-first, with name and pipeline status."""
    result = []
    for sid, data in sessions.items():
        node0 = data.get("node0_result") or {}
        node1 = data.get("node1_result") or {}
        scope_valid = node0.get("result") == "PASS" if node0 else None
        result.append(SessionSummary(
            session_id=sid,
            name=data.get("name", "Untitled Paper"),
            uploaded_at=data.get("uploaded_at"),
            scope_valid=scope_valid,
            model_type=node1.get("model_type"),
            node2_status=(data.get("node2_result") or {}).get("status"),
            node3_status=(data.get("node3_result") or {}).get("status"),
        ))
    result.sort(key=lambda s: s.uploaded_at or "", reverse=True)
    return result


@app.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str):
    """Return full metadata for a single session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    data = sessions[session_id]
    node0 = data.get("node0_result") or {}
    node1 = data.get("node1_result") or {}
    scope_valid = node0.get("result") == "PASS" if node0 else None
    return SessionDetail(
        session_id=session_id,
        name=data.get("name", "Untitled Paper"),
        gcs_uri=data.get("gcs_uri", ""),
        uploaded_at=data.get("uploaded_at"),
        scope_valid=scope_valid,
        scope_reason=node0.get("reason"),
        model_type=node1.get("model_type"),
        node2_status=(data.get("node2_result") or {}).get("status"),
        node3_status=(data.get("node3_result") or {}).get("status"),
        has_chat_history=bool(data.get("history")),
    )


@app.get("/sessions/{session_id}/history", response_model=List[ChatHistoryEntry])
async def get_session_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    history = sessions[session_id].get("history") or []
    cleaned_history = []
    for entry in history:
        role = entry.get("role")
        text = entry.get("text")
        if role in {"user", "assistant"} and isinstance(text, str):
            cleaned_history.append(ChatHistoryEntry(role=role, text=text))
    return cleaned_history


@app.get("/sessions/{session_id}/artifacts", response_model=SessionArtifactsResponse)
async def get_session_artifacts(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    data = sessions[session_id]
    node2 = data.get("node2_result") or {}
    node3 = data.get("node3_result") or {}
    return SessionArtifactsResponse(
        session_id=session_id,
        implementation_files=sorted((node2.get("files") or {}).keys()),
        acceleration_files=sorted(node3.get("stub_files") or []),
    )


@app.delete("/sessions/{session_id}", response_model=DeleteResponse)
async def delete_session_endpoint(session_id: str):
    """Delete a session: removes in-memory state, GCS session blob, and all artifact files."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    sessions.pop(session_id)
    # Both operations fire concurrently — session JSON + all paper artifacts
    await asyncio.gather(
        delete_session_blob(session_id),
        delete_all_artifacts(session_id),
        return_exceptions=True,
    )
    logger.info("Session deleted | session=%s", session_id)
    return DeleteResponse(session_id=session_id, deleted=True, message="Session deleted.")


@app.get(
    "/sessions/{session_id}/artifacts/{artifact_group}/{file_name:path}",
    response_model=ArtifactContentResponse,
)
async def get_artifact_content(session_id: str, artifact_group: str, file_name: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    content: Optional[str] = None

    # Try local filesystem first
    try:
        root = _resolve_artifact_root(sessions[session_id], artifact_group)
        target = (root / file_name).resolve()
        if target == root or root not in target.parents:
            raise HTTPException(status_code=400, detail="Invalid artifact path.")
        if target.exists() and target.is_file():
            try:
                content = await asyncio.to_thread(target.read_text)
            except UnicodeDecodeError:
                raise HTTPException(status_code=422, detail="Artifact file is not valid text.")
    except HTTPException as e:
        if e.status_code == 400:
            raise
        # 404 from _resolve_artifact_root — fall through to GCS

    # Fall back to GCS
    if content is None:
        content = await download_artifact(session_id, artifact_group, file_name)
        if content is None:
            raise HTTPException(status_code=404, detail="Artifact file not found.")

    return ArtifactContentResponse(
        session_id=session_id,
        artifact_group=artifact_group,
        file_name=file_name,
        language=_infer_language(file_name),
        content=content,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload(
    pdf_url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    if not pdf_url and not file:
        raise HTTPException(
            status_code=400, detail="Provide either pdf_url or a file upload."
        )

    # SSRF guard — only allow http/https URLs
    if pdf_url:
        try:
            parsed = httpx.URL(pdf_url)
            if parsed.scheme not in ("http", "https"):
                raise HTTPException(
                    status_code=422, detail="pdf_url must use http or https."
                )
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid pdf_url.")

    session_id = str(uuid.uuid4())
    source = "url" if pdf_url else "upload"
    logger.info("POST /upload | session=%s | source=%s", session_id, source)

    try:
        if pdf_url:
            gcs_uri = await upload_pdf_from_url(pdf_url, session_id)
        else:
            pdf_bytes = await file.read()
            if len(pdf_bytes) > _MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                )
            if not pdf_bytes[:4] == b"%PDF":
                raise HTTPException(
                    status_code=422, detail="Uploaded file does not appear to be a valid PDF."
                )
            gcs_uri = await upload_pdf_bytes(pdf_bytes, session_id)

        uploaded_at = datetime.now(timezone.utc).isoformat()
        sessions[session_id] = {
            "gcs_uri": gcs_uri,
            "name": "Untitled Paper",
            "uploaded_at": uploaded_at,
            "history": [],
        }
        await save_session(session_id, sessions[session_id])
        # Fetch the paper title in the background — doesn't block the response
        asyncio.create_task(_fetch_and_store_title(session_id, gcs_uri))
        logger.info("Upload complete | session=%s | gcs_uri=%s", session_id, gcs_uri)
        return UploadResponse(session_id=session_id, gcs_uri=gcs_uri)

    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to fetch PDF from URL: {e.response.status_code} {e.response.reason_phrase}",
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=422, detail=f"Could not reach PDF URL: {e}")
    except ValueError as e:
        # Raised by pdf_loader for size/content-type violations
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Unexpected upload error | session=%s | %s", session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Upload failed unexpectedly.")


@app.post("/run/node0", response_model=Node0Response)
async def run_node0_endpoint(body: NodeRunRequest):
    session_id = body.session_id
    logger.info("POST /run/node0 | session=%s", session_id)
    if session_id not in sessions:
        logger.warning("Session not found | session=%s", session_id)
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        gcs_uri = sessions[session_id]["gcs_uri"]
        result = await run_node0(gcs_uri)
        sessions[session_id]["node0_result"] = {
            "result": result.result,
            "reason": result.reason,
        }
        await save_session(session_id, sessions[session_id])
        logger.info("Node0 endpoint done | session=%s | result=%s", session_id, result.result)
        return Node0Response(
            session_id=session_id, result=result.result, reason=result.reason
        )

    except ModelResponseError as e:
        logger.error("Model response error in node0 | session=%s | %s", session_id, e)
        raise HTTPException(status_code=422, detail=str(e))
    except GeminiClientError as e:
        logger.error("Gemini error in node0 | session=%s | code=%s", session_id, e.code)
        _raise_gemini_error(e)
    except GoogleAPICallError as e:
        logger.error("Vertex AI error in node0 | session=%s | %s", session_id, e.message)
        raise HTTPException(status_code=502, detail=f"Vertex AI error: {e.message}")
    except Exception as e:
        logger.error("Unexpected error in node0 | session=%s | %s", session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Node 0 failed unexpectedly.")


@app.post("/run/node1", response_model=Node1Response)
async def run_node1_endpoint(body: NodeRunRequest):
    session_id = body.session_id
    logger.info("POST /run/node1 | session=%s", session_id)
    if session_id not in sessions:
        logger.warning("Session not found | session=%s", session_id)
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        gcs_uri = sessions[session_id]["gcs_uri"]
        node0 = sessions[session_id].get("node0_result")
        pre_validated = bool(node0 and node0.get("result") == "PASS")
        logger.info(
            "Node1 endpoint | session=%s | pre_validated=%s | node0_stored=%s",
            session_id,
            pre_validated,
            node0 is not None,
        )
        blueprint = await run_node1(gcs_uri, pre_validated=pre_validated)
        sessions[session_id]["node1_result"] = blueprint.model_dump()
        await save_session(session_id, sessions[session_id])
        logger.info(
            "Node1 endpoint done | session=%s | model_type=%s",
            session_id,
            blueprint.model_type,
        )
        return Node1Response(session_id=session_id, blueprint=blueprint)

    except ScopeRejectedError as e:
        logger.warning("Scope rejected | session=%s | reason=%s", session_id, e.reason)
        raise HTTPException(
            status_code=400,
            detail=f"Paper rejected by scope check: {e.reason}",
        )
    except ModelResponseError as e:
        logger.error("Model response error in node1 | session=%s | %s", session_id, e)
        raise HTTPException(status_code=422, detail=str(e))
    except GeminiClientError as e:
        logger.error("Gemini error in node1 | session=%s | code=%s", session_id, e.code)
        _raise_gemini_error(e)
    except GoogleAPICallError as e:
        logger.error("Vertex AI error in node1 | session=%s | %s", session_id, e.message)
        raise HTTPException(status_code=502, detail=f"Vertex AI error: {e.message}")
    except Exception as e:
        logger.error("Unexpected error in node1 | session=%s | %s", session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Node 1 failed unexpectedly.")


@app.post("/run/node2", response_model=Node2Response)
async def run_node2_endpoint(body: NodeRunRequest):
    session_id = body.session_id
    logger.info("POST /run/node2 | session=%s", session_id)
    if session_id not in sessions:
        logger.warning("Session not found | session=%s", session_id)
        raise HTTPException(status_code=404, detail="Session not found.")

    if not sessions[session_id].get("node1_result"):
        raise HTTPException(
            status_code=400,
            detail="Node 1 must be run before Node 2. Run /run/node1 first.",
        )

    try:
        result = await asyncio.to_thread(run_node2, sessions[session_id]["node1_result"])
        sessions[session_id]["node2_result"] = result
        await save_session(session_id, sessions[session_id])
        if result.get("status") == "completed":
            asyncio.create_task(_upload_node_artifacts(session_id, result, "implementation"))
        logger.info("Node2 endpoint done | session=%s | status=%s", session_id, result.get("status"))
        if result.get("status") == "error":
            error_type = result.get("error_type", "")
            status_code = 504 if error_type == "timeout" else 502 if error_type == "auth" else 500
            raise HTTPException(status_code=status_code, detail=result.get("error", "Node 2 failed."))
        return Node2Response(session_id=session_id, **result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error in node2 | session=%s | %s", session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Node 2 failed unexpectedly.")


@app.post("/run/node3", response_model=Node3Response)
async def run_node3_endpoint(body: NodeRunRequest):
    session_id = body.session_id
    logger.info("POST /run/node3 | session=%s", session_id)
    if session_id not in sessions:
        logger.warning("Session not found | session=%s", session_id)
        raise HTTPException(status_code=404, detail="Session not found.")

    if not sessions[session_id].get("node2_result"):
        raise HTTPException(
            status_code=400,
            detail="Node 2 must be run before Node 3. Run /run/node2 first.",
        )

    try:
        # Pass node2_result so Node 3 can locate the correct scaffold directory
        result = await asyncio.to_thread(run_node3, sessions[session_id]["node2_result"])
        sessions[session_id]["node3_result"] = result
        await save_session(session_id, sessions[session_id])
        if result.get("status") == "completed":
            asyncio.create_task(_upload_node_artifacts(session_id, result, "acceleration"))
        logger.info("Node3 endpoint done | session=%s | status=%s", session_id, result.get("status"))
        if result.get("status") == "error":
            error_type = result.get("error_type", "")
            status_code = 504 if error_type == "timeout" else 502 if error_type == "auth" else 500
            raise HTTPException(status_code=status_code, detail=result.get("error", "Node 3 failed."))
        return Node3Response(session_id=session_id, **result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error in node3 | session=%s | %s", session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Node 3 failed unexpectedly.")


def _sse(event_type: str, payload: dict) -> str:
    """Format a single SSE data line."""
    return f"data: {json.dumps({'type': event_type, **payload})}\n\n"


@app.post("/run/pipeline/stream")
async def run_pipeline_stream(body: NodeRunRequest):
    """
    Streaming variant of /run/pipeline.
    Yields SSE events as each node starts and finishes so the client
    can display live progress without polling.

    Event types:
      node_start  — a node has begun
      node_done   — a node finished successfully
      node_error  — a node failed (node2/3: non-fatal; node0/1: stream ends)
      done        — full PipelineResponse payload, stream complete
      error       — fatal error before or during graph execution
    """
    session_id = body.session_id
    logger.info("POST /run/pipeline/stream | session=%s", session_id)

    if session_id not in sessions:
        async def _not_found():
            yield _sse("error", {"message": "Session not found."})
        return StreamingResponse(_not_found(), media_type="text/event-stream")

    gcs_uri = sessions[session_id].get("gcs_uri")
    if not gcs_uri:
        async def _no_pdf():
            yield _sse("error", {"message": "No uploaded PDF for this session. Run /upload first."})
        return StreamingResponse(_no_pdf(), media_type="text/event-stream")

    async def _stream() -> AsyncGenerator[str, None]:
        scope_valid = False
        scope_reason = ""
        blueprint = None
        scaffold: dict = {}
        cuda: dict = {}

        # ── Node 0 ────────────────────────────────────────────────────────────
        yield _sse("node_start", {"node": "node0", "message": "Validating paper scope..."})
        try:
            result0 = await run_node0(gcs_uri)
            scope_valid = result0.result == "PASS"
            scope_reason = result0.reason
            sessions[session_id]["node0_result"] = {
                "result": result0.result,
                "reason": result0.reason,
            }
            asyncio.create_task(save_session(session_id, sessions[session_id]))

            if scope_valid:
                yield _sse("node_done", {
                    "node": "node0",
                    "result": "PASS",
                    "message": f"Scope valid: {scope_reason}",
                })
            else:
                yield _sse("node_error", {
                    "node": "node0",
                    "result": "FAIL",
                    "message": f"Paper out of scope: {scope_reason}",
                })
                yield _sse("done", {
                    "session_id": session_id,
                    "scope_valid": False,
                    "scope_reason": scope_reason,
                    "node2_status": "skipped",
                    "node3_status": "skipped",
                })
                return

        except Exception as e:
            logger.error("Stream pipeline node0 error | session=%s | %s", session_id, e, exc_info=True)
            yield _sse("node_error", {"node": "node0", "message": str(e)})
            yield _sse("error", {"message": f"Node 0 failed: {e}"})
            return

        # ── Node 1 ────────────────────────────────────────────────────────────
        yield _sse("node_start", {"node": "node1", "message": "Extracting architecture blueprint..."})
        try:
            blueprint = await run_node1(gcs_uri, pre_validated=True)
            sessions[session_id]["node1_result"] = blueprint.model_dump()
            asyncio.create_task(save_session(session_id, sessions[session_id]))
            yield _sse("node_done", {
                "node": "node1",
                "message": f"Blueprint extracted: {blueprint.model_type}",
                "model_type": blueprint.model_type,
            })

        except ScopeRejectedError as e:
            logger.warning("Stream pipeline node1 scope rejected | session=%s | %s", session_id, e.reason)
            yield _sse("node_error", {"node": "node1", "message": f"Scope rejected: {e.reason}"})
            yield _sse("done", {
                "session_id": session_id,
                "scope_valid": False,
                "scope_reason": e.reason,
                "node2_status": "skipped",
                "node3_status": "skipped",
            })
            return
        except Exception as e:
            logger.error("Stream pipeline node1 error | session=%s | %s", session_id, e, exc_info=True)
            yield _sse("node_error", {"node": "node1", "message": str(e)})
            yield _sse("error", {"message": f"Node 1 failed: {e}"})
            return

        # ── Node 2 ────────────────────────────────────────────────────────────
        yield _sse("node_start", {"node": "node2", "message": "Generating PyTorch scaffold..."})
        try:
            scaffold = await asyncio.to_thread(run_node2, blueprint.model_dump())
            sessions[session_id]["node2_result"] = scaffold
            asyncio.create_task(save_session(session_id, sessions[session_id]))

            if scaffold.get("status") == "error":
                yield _sse("node_error", {
                    "node": "node2",
                    "message": scaffold.get("error", "Node 2 failed."),
                })
            else:
                asyncio.create_task(_upload_node_artifacts(session_id, scaffold, "implementation"))
                files = list((scaffold.get("files") or {}).keys())
                yield _sse("node_done", {
                    "node": "node2",
                    "message": f"Scaffold generated: {len(files)} file(s)",
                    "files": files,
                })

        except Exception as e:
            logger.error("Stream pipeline node2 error | session=%s | %s", session_id, e, exc_info=True)
            scaffold = {"status": "error", "error": str(e)}
            yield _sse("node_error", {"node": "node2", "message": str(e)})

        # ── Node 3 ────────────────────────────────────────────────────────────
        yield _sse("node_start", {"node": "node3", "message": "Generating CUDA hardware blueprint..."})
        try:
            cuda = await asyncio.to_thread(run_node3, scaffold)
            sessions[session_id]["node3_result"] = cuda
            asyncio.create_task(save_session(session_id, sessions[session_id]))

            if cuda.get("status") == "error":
                yield _sse("node_error", {
                    "node": "node3",
                    "message": cuda.get("error", "Node 3 failed."),
                })
            else:
                asyncio.create_task(_upload_node_artifacts(session_id, cuda, "acceleration"))
                stubs = cuda.get("stub_files") or []
                bottlenecks = cuda.get("bottlenecks") or []
                yield _sse("node_done", {
                    "node": "node3",
                    "message": f"{len(stubs)} CUDA stub(s), {len(bottlenecks)} bottleneck(s) identified",
                    "stub_files": stubs,
                    "bottleneck_count": len(bottlenecks),
                })

        except Exception as e:
            logger.error("Stream pipeline node3 error | session=%s | %s", session_id, e, exc_info=True)
            cuda = {"status": "error", "error": str(e)}
            yield _sse("node_error", {"node": "node3", "message": str(e)})

        # ── Final result ──────────────────────────────────────────────────────
        node2_error = scaffold.get("error") if scaffold else None
        node3_error = cuda.get("error") if cuda else None
        yield _sse("done", {
            "session_id": session_id,
            "scope_valid": True,
            "scope_reason": scope_reason,
            "blueprint": blueprint.model_dump() if blueprint else None,
            "node2_status": scaffold.get("status", "error") if scaffold else "skipped",
            "node2_files": scaffold.get("files") if scaffold else None,
            "node3_status": cuda.get("status", "error") if cuda else "skipped",
            "node3_stub_files": cuda.get("stub_files") if cuda else None,
            "node3_bottlenecks": cuda.get("bottlenecks") if cuda else None,
            "error": node2_error or node3_error,
        })
        logger.info(
            "Stream pipeline complete | session=%s | node2=%s | node3=%s",
            session_id,
            scaffold.get("status", "error") if scaffold else "skipped",
            cuda.get("status", "error") if cuda else "skipped",
        )

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/run/pipeline", response_model=PipelineResponse)
async def run_pipeline_endpoint(body: NodeRunRequest):
    from backend.graph import pipeline
    from backend.schemas.state import AgentState

    session_id = body.session_id
    logger.info("POST /run/pipeline | session=%s", session_id)

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    gcs_uri = sessions[session_id].get("gcs_uri")
    if not gcs_uri:
        raise HTTPException(status_code=400, detail="Session has no uploaded PDF. Run /upload first.")

    try:
        initial_state: AgentState = {
            "session_id": session_id,
            "pdf_gcs_uri": gcs_uri,
        }

        logger.info("Pipeline invoking graph | session=%s", session_id)
        final_state = await pipeline.ainvoke(initial_state)

        # Persist all node results into the session
        scope_valid: bool = final_state.get("scope_valid", False)
        scope_reason: str = final_state.get("scope_reason", "")

        sessions[session_id]["node0_result"] = {
            "result": "PASS" if scope_valid else "FAIL",
            "reason": scope_reason,
        }

        blueprint = final_state.get("blueprint")
        if blueprint:
            sessions[session_id]["node1_result"] = blueprint.model_dump()

        scaffold = final_state.get("scaffold_code")
        if scaffold:
            sessions[session_id]["node2_result"] = scaffold

        cuda = final_state.get("cuda_blueprint")
        if cuda:
            sessions[session_id]["node3_result"] = cuda

        await save_session(session_id, sessions[session_id])
        logger.info(
            "Pipeline complete | session=%s | scope_valid=%s | node2=%s | node3=%s",
            session_id,
            scope_valid,
            scaffold.get("status") if scaffold else "skipped",
            cuda.get("status") if cuda else "skipped",
        )

        # Scope failed — pipeline stopped after node0 or node1
        if not scope_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Paper rejected by scope check: {scope_reason}",
            )

        # Node1 infrastructure failure (not a scope rejection)
        if final_state.get("error"):
            raise HTTPException(status_code=500, detail=final_state["error"])

        # Node2/3 failures are non-fatal — surface them in the response body
        node2_status = scaffold.get("status", "error") if scaffold else "skipped"
        node2_error = scaffold.get("error") if scaffold else None
        node3_status = cuda.get("status", "error") if cuda else "skipped"
        node3_error = cuda.get("error") if cuda else None

        if node2_status == "error":
            logger.warning(
                "Pipeline node2 failed | session=%s | error=%s", session_id, node2_error
            )
        if node3_status == "error":
            logger.warning(
                "Pipeline node3 failed | session=%s | error=%s", session_id, node3_error
            )

        return PipelineResponse(
            session_id=session_id,
            scope_valid=scope_valid,
            scope_reason=scope_reason,
            blueprint=blueprint,
            node2_status=node2_status,
            node2_files=scaffold.get("files") if scaffold else None,
            node3_status=node3_status,
            node3_stub_files=cuda.get("stub_files") if cuda else None,
            node3_bottlenecks=cuda.get("bottlenecks") if cuda else None,
            error=node2_error or node3_error,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Pipeline error | session=%s | %s", session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")


@app.post("/chat")
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
        async for chunk in stream_chat(sessions[session_id], message):
            yield chunk
            try:
                data = json.loads(chunk[6:].strip())  # strip "data: "
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
                    _persist(session_id)
            except Exception as exc:
                logger.warning(
                    "SSE parse error in _stream_and_persist | session=%s | error=%s",
                    session_id, exc,
                )

    return StreamingResponse(
        _stream_and_persist(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
