import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

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
from backend.pipeline import PipelineManager, sse_event_stream
from backend.nodes.node2_client import run_node2
from backend.nodes.node3_client import run_node3
from backend.schemas.api import (
    ChatRequest,
    HealthResponse,
    Node0Response,
    Node1Response,
    Node2Response,
    Node3Response,
    NodeRunRequest,
    RunAcceptedResponse,
    RunRequest,
    RunStatusResponse,
    PipelineResponse,
    UploadResponse,
)
from backend.session_store import load_all_sessions, save_session
from backend.tools.model_response import ModelResponseError
from backend.tools.pdf_loader import upload_pdf_bytes, upload_pdf_from_url

# 50 MB limit for direct file uploads
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# In-memory session store: session_id -> session data dict
sessions: Dict[str, Dict[str, Any]] = {}


async def _persist_for_pipeline(session_id: str) -> None:
    data = sessions.get(session_id)
    if data is None:
        logger.warning("_persist_for_pipeline called for unknown session=%s", session_id)
        return
    asyncio.create_task(save_session(session_id, data))
    await asyncio.sleep(0)


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
    allow_methods=["*"],
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


pipeline_manager = PipelineManager(sessions, persist_session=_persist_for_pipeline)


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

        sessions[session_id] = {"gcs_uri": gcs_uri, "history": []}
        await save_session(session_id, sessions[session_id])
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

        # Scope failed — pipeline stopped after node0
        if not scope_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Paper rejected by scope check: {scope_reason}",
            )

        # Surface any pipeline-level error stored in state
        if final_state.get("error"):
            raise HTTPException(status_code=500, detail=final_state["error"])

        return PipelineResponse(
            session_id=session_id,
            scope_valid=scope_valid,
            scope_reason=scope_reason,
            blueprint=blueprint,
            node2_status=scaffold.get("status", "error") if scaffold else "skipped",
            node2_files=scaffold.get("files") if scaffold else None,
            node3_status=cuda.get("status", "error") if cuda else "skipped",
            node3_stub_files=cuda.get("stub_files") if cuda else None,
            node3_bottlenecks=cuda.get("bottlenecks") if cuda else None,
            error=scaffold.get("error") or (cuda.get("error") if cuda else None),
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


@app.post("/run", response_model=RunAcceptedResponse)
async def run_pipeline(body: RunRequest):
    if not body.use_demo_cache and not body.pdf_url and not body.gcs_uri:
        raise HTTPException(
            status_code=400,
            detail="Provide pdf_url, gcs_uri, or set use_demo_cache=true.",
        )

    try:
        run = await pipeline_manager.start_run(
            pdf_url=body.pdf_url,
            gcs_uri=body.gcs_uri,
            session_id=body.session_id,
            use_demo_cache=body.use_demo_cache,
            demo_cache_key=body.demo_cache_key,
        )
        run_id = run["run_id"]
        return RunAcceptedResponse(
            run_id=run_id,
            session_id=run["session_id"],
            status=run["status"],
            current_node=run["current_node"],
            poll_url=f"/runs/{run_id}",
            stream_url=f"/runs/{run_id}/events",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str):
    try:
        run = await pipeline_manager.get_run(run_id)
        return RunStatusResponse(**run)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found.")


@app.get("/runs/{run_id}/events")
async def stream_run_events(run_id: str):
    try:
        await pipeline_manager.get_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found.")

    return StreamingResponse(
        sse_event_stream(pipeline_manager, run_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
