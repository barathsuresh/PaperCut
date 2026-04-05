import asyncio
import logging
import uuid
from typing import AsyncGenerator, Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from google.api_core.exceptions import GoogleAPICallError
from google.genai.errors import ClientError as GeminiClientError

from backend.app_state import (
    fetch_and_store_title,
    raise_gemini_error,
    sessions,
    upload_node_artifacts,
)
from backend.nodes.node0_validator import run_node0
from backend.nodes.node1_ingestor import ScopeRejectedError, run_node1
from backend.nodes.node2_client import run_node2
from backend.nodes.node3_client import run_node3
from backend.schemas.api import (
    Node0Response,
    Node1Response,
    Node2Response,
    Node3Response,
    NodeRunRequest,
    PipelineResponse,
    UploadResponse,
)
from backend.schemas.state import AgentState
from backend.services.pipeline_state import (
    apply_blueprint_result,
    apply_node0_result,
    node_error_status_code,
    pipeline_response,
    require_pipeline_prerequisite,
)
from backend.services.pipeline_runtime import (
    new_session_payload,
    persist_graph_results,
    run_codegen_node_for_session,
    run_stream_codegen_node,
    sse,
)
from backend.services.session_views import get_session_or_404
from backend.session_store import save_session
from backend.tools.model_response import ModelResponseError
from backend.tools.pdf_loader import GCSUnavailableError, upload_pdf_bytes, upload_pdf_from_url

logger = logging.getLogger(__name__)

router = APIRouter()

_MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@router.post("/upload", response_model=UploadResponse)
async def upload(
    pdf_url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    if not pdf_url and not file:
        raise HTTPException(status_code=400, detail="Provide either pdf_url or a file upload.")

    if pdf_url:
        try:
            parsed = httpx.URL(pdf_url)
            if parsed.scheme not in ("http", "https"):
                raise HTTPException(status_code=422, detail="pdf_url must use http or https.")
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

        sessions[session_id] = new_session_payload(gcs_uri)
        await save_session(session_id, sessions[session_id])
        asyncio.create_task(fetch_and_store_title(session_id, gcs_uri))
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
    except GCSUnavailableError as e:
        raise HTTPException(
            status_code=503,
            detail=f"{e} Check Google credentials, network access, and GCS reachability.",
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Unexpected upload error | session=%s | %s", session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Upload failed unexpectedly.")


@router.post("/run/node0", response_model=Node0Response)
async def run_node0_endpoint(body: NodeRunRequest):
    session_id = body.session_id
    logger.info("POST /run/node0 | session=%s", session_id)
    get_session_or_404(sessions, session_id)

    try:
        gcs_uri = sessions[session_id]["gcs_uri"]
        result = await run_node0(gcs_uri)
        apply_node0_result(sessions[session_id], result)
        await save_session(session_id, sessions[session_id])
        logger.info("Node0 endpoint done | session=%s | result=%s", session_id, result.result)
        return Node0Response(session_id=session_id, result=result.result, reason=result.reason)
    except ModelResponseError as e:
        logger.error("Model response error in node0 | session=%s | %s", session_id, e)
        raise HTTPException(status_code=422, detail=str(e))
    except GeminiClientError as e:
        logger.error("Gemini error in node0 | session=%s | code=%s", session_id, e.code)
        raise_gemini_error(e)
    except GoogleAPICallError as e:
        logger.error("Vertex AI error in node0 | session=%s | %s", session_id, e.message)
        raise HTTPException(status_code=502, detail=f"Vertex AI error: {e.message}")
    except Exception as e:
        logger.error("Unexpected error in node0 | session=%s | %s", session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Node 0 failed unexpectedly.")


@router.post("/run/node1", response_model=Node1Response)
async def run_node1_endpoint(body: NodeRunRequest):
    session_id = body.session_id
    logger.info("POST /run/node1 | session=%s", session_id)
    get_session_or_404(sessions, session_id)

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
        apply_blueprint_result(sessions[session_id], blueprint)
        await save_session(session_id, sessions[session_id])
        logger.info("Node1 endpoint done | session=%s | model_type=%s", session_id, blueprint.model_type)
        return Node1Response(session_id=session_id, blueprint=blueprint)
    except ScopeRejectedError as e:
        logger.warning("Scope rejected | session=%s | reason=%s", session_id, e.reason)
        raise HTTPException(status_code=400, detail=f"Paper rejected by scope check: {e.reason}")
    except ModelResponseError as e:
        logger.error("Model response error in node1 | session=%s | %s", session_id, e)
        raise HTTPException(status_code=422, detail=str(e))
    except GeminiClientError as e:
        logger.error("Gemini error in node1 | session=%s | code=%s", session_id, e.code)
        raise_gemini_error(e)
    except GoogleAPICallError as e:
        logger.error("Vertex AI error in node1 | session=%s | %s", session_id, e.message)
        raise HTTPException(status_code=502, detail=f"Vertex AI error: {e.message}")
    except Exception as e:
        logger.error("Unexpected error in node1 | session=%s | %s", session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Node 1 failed unexpectedly.")


@router.post("/run/node2", response_model=Node2Response)
async def run_node2_endpoint(body: NodeRunRequest):
    session_id = body.session_id
    logger.info("POST /run/node2 | session=%s", session_id)
    session_data = get_session_or_404(sessions, session_id)
    require_pipeline_prerequisite(
        session_data, "node1_result", "Node 1 must be run before Node 2. Run /run/node1 first."
    )

    try:
        result = await run_codegen_node_for_session(
            session_id=session_id,
            session_data=session_data,
            input_payload=session_data["node1_result"],
            runner=run_node2,
            session_key="node2_result",
            artifact_group="implementation",
        )
        logger.info("Node2 endpoint done | session=%s | status=%s", session_id, result.get("status"))
        if result.get("status") == "error":
            raise HTTPException(
                status_code=node_error_status_code(result),
                detail=result.get("error", "Node 2 failed."),
            )
        return Node2Response(session_id=session_id, **result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error in node2 | session=%s | %s", session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Node 2 failed unexpectedly.")


@router.post("/run/node3", response_model=Node3Response)
async def run_node3_endpoint(body: NodeRunRequest):
    session_id = body.session_id
    logger.info("POST /run/node3 | session=%s", session_id)
    session_data = get_session_or_404(sessions, session_id)
    require_pipeline_prerequisite(
        session_data, "node2_result", "Node 2 must be run before Node 3. Run /run/node2 first."
    )

    try:
        result = await run_codegen_node_for_session(
            session_id=session_id,
            session_data=session_data,
            input_payload=session_data["node2_result"],
            runner=run_node3,
            session_key="node3_result",
            artifact_group="acceleration",
        )
        logger.info("Node3 endpoint done | session=%s | status=%s", session_id, result.get("status"))
        if result.get("status") == "error":
            raise HTTPException(
                status_code=node_error_status_code(result),
                detail=result.get("error", "Node 3 failed."),
            )
        return Node3Response(session_id=session_id, **result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error in node3 | session=%s | %s", session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Node 3 failed unexpectedly.")


@router.post("/run/pipeline/stream")
async def run_pipeline_stream(body: NodeRunRequest):
    session_id = body.session_id
    logger.info("POST /run/pipeline/stream | session=%s", session_id)

    if session_id not in sessions:
        async def _not_found():
            yield sse("error", {"message": "Session not found."})
        return StreamingResponse(_not_found(), media_type="text/event-stream")

    gcs_uri = sessions[session_id].get("gcs_uri")
    if not gcs_uri:
        async def _no_pdf():
            yield sse("error", {"message": "No uploaded PDF for this session. Run /upload first."})
        return StreamingResponse(_no_pdf(), media_type="text/event-stream")

    async def _stream() -> AsyncGenerator[str, None]:
        scope_valid = False
        scope_reason = ""
        blueprint = None
        scaffold: dict = {}
        cuda: dict = {}

        yield sse("node_start", {"node": "node0", "message": "Validating paper scope..."})
        try:
            result0 = await run_node0(gcs_uri)
            scope_valid = result0.result == "PASS"
            scope_reason = result0.reason
            apply_node0_result(sessions[session_id], result0)
            asyncio.create_task(save_session(session_id, sessions[session_id]))

            if scope_valid:
                yield sse("node_done", {"node": "node0", "result": "PASS", "message": f"Scope valid: {scope_reason}"})
            else:
                yield sse("node_error", {"node": "node0", "result": "FAIL", "message": f"Paper out of scope: {scope_reason}"})
                yield sse("done", {"session_id": session_id, "scope_valid": False, "scope_reason": scope_reason, "node2_status": "skipped", "node3_status": "skipped"})
                return
        except Exception as e:
            logger.error("Stream pipeline node0 error | session=%s | %s", session_id, e, exc_info=True)
            yield sse("node_error", {"node": "node0", "message": str(e)})
            yield sse("error", {"message": f"Node 0 failed: {e}"})
            return

        yield sse("node_start", {"node": "node1", "message": "Extracting architecture blueprint..."})
        try:
            blueprint = await run_node1(gcs_uri, pre_validated=True)
            apply_blueprint_result(sessions[session_id], blueprint)
            asyncio.create_task(save_session(session_id, sessions[session_id]))
            yield sse("node_done", {"node": "node1", "message": f"Blueprint extracted: {blueprint.model_type}", "model_type": blueprint.model_type})
        except ScopeRejectedError as e:
            logger.warning("Stream pipeline node1 scope rejected | session=%s | %s", session_id, e.reason)
            yield sse("node_error", {"node": "node1", "message": f"Scope rejected: {e.reason}"})
            yield sse("done", {"session_id": session_id, "scope_valid": False, "scope_reason": e.reason, "node2_status": "skipped", "node3_status": "skipped"})
            return
        except Exception as e:
            logger.error("Stream pipeline node1 error | session=%s | %s", session_id, e, exc_info=True)
            yield sse("node_error", {"node": "node1", "message": str(e)})
            yield sse("error", {"message": f"Node 1 failed: {e}"})
            return

        yield sse("node_start", {"node": "node2", "message": "Generating PyTorch scaffold..."})
        try:
            scaffold = await run_stream_codegen_node(
                session_id=session_id,
                input_payload=blueprint.model_dump(),
                runner=run_node2,
                session_key="node2_result",
                artifact_group="implementation",
            )
            if scaffold.get("status") == "error":
                yield sse("node_error", {"node": "node2", "message": scaffold.get("error", "Node 2 failed.")})
            else:
                files = list((scaffold.get("files") or {}).keys())
                yield sse("node_done", {"node": "node2", "message": f"Scaffold generated: {len(files)} file(s)", "files": files})
        except Exception as e:
            logger.error("Stream pipeline node2 error | session=%s | %s", session_id, e, exc_info=True)
            scaffold = {"status": "error", "error": str(e)}
            yield sse("node_error", {"node": "node2", "message": str(e)})

        yield sse("node_start", {"node": "node3", "message": "Generating CUDA hardware blueprint..."})
        try:
            cuda = await run_stream_codegen_node(
                session_id=session_id,
                input_payload=scaffold,
                runner=run_node3,
                session_key="node3_result",
                artifact_group="acceleration",
            )
            if cuda.get("status") == "error":
                yield sse("node_error", {"node": "node3", "message": cuda.get("error", "Node 3 failed.")})
            else:
                stubs = cuda.get("stub_files") or []
                bottlenecks = cuda.get("bottlenecks") or []
                yield sse("node_done", {"node": "node3", "message": f"{len(stubs)} CUDA stub(s), {len(bottlenecks)} bottleneck(s) identified", "stub_files": stubs, "bottleneck_count": len(bottlenecks)})
        except Exception as e:
            logger.error("Stream pipeline node3 error | session=%s | %s", session_id, e, exc_info=True)
            cuda = {"status": "error", "error": str(e)}
            yield sse("node_error", {"node": "node3", "message": str(e)})

        response = pipeline_response(session_id, True, scope_reason, blueprint, scaffold, cuda)
        yield sse("done", response.model_dump())
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


@router.post("/run/pipeline", response_model=PipelineResponse)
async def run_pipeline_endpoint(body: NodeRunRequest):
    from backend.graph import pipeline

    session_id = body.session_id
    logger.info("POST /run/pipeline | session=%s", session_id)

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    gcs_uri = sessions[session_id].get("gcs_uri")
    if not gcs_uri:
        raise HTTPException(status_code=400, detail="Session has no uploaded PDF. Run /upload first.")

    try:
        initial_state: AgentState = {"session_id": session_id, "pdf_gcs_uri": gcs_uri}
        logger.info("Pipeline invoking graph | session=%s", session_id)
        final_state = await pipeline.ainvoke(initial_state)

        scope_valid, scope_reason, blueprint, scaffold, cuda = persist_graph_results(
            sessions[session_id], final_state
        )

        await save_session(session_id, sessions[session_id])
        logger.info(
            "Pipeline complete | session=%s | scope_valid=%s | node2=%s | node3=%s",
            session_id,
            scope_valid,
            scaffold.get("status") if scaffold else "skipped",
            cuda.get("status") if cuda else "skipped",
        )

        if not scope_valid:
            raise HTTPException(status_code=400, detail=f"Paper rejected by scope check: {scope_reason}")
        if final_state.get("error"):
            raise HTTPException(status_code=500, detail=final_state["error"])

        response = pipeline_response(session_id, scope_valid, scope_reason, blueprint, scaffold, cuda)
        if response.node2_status == "error":
            logger.warning("Pipeline node2 failed | session=%s | error=%s", session_id, response.error)
        if response.node3_status == "error":
            logger.warning("Pipeline node3 failed | session=%s | error=%s", session_id, response.error)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Pipeline error | session=%s | %s", session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")
