import asyncio
import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional

from backend import config
from backend.demo_cache import load_demo_cache
from backend.nodes.node0_validator import run_node0
from backend.nodes.node1_ingestor import run_node1
from backend.nodes.node2_client import run_node2
from backend.nodes.node3_client import run_node3
from backend.schemas.contract import ArchitectureBlueprint
from backend.tools.pdf_loader import upload_pdf_from_url


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineManager:
    def __init__(
        self,
        sessions: Dict[str, Dict[str, Any]],
        persist_session: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> None:
        self.sessions = sessions
        self.persist_session = persist_session
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    async def start_run(
        self,
        *,
        pdf_url: Optional[str],
        gcs_uri: Optional[str],
        session_id: Optional[str],
        use_demo_cache: bool,
        demo_cache_key: str,
    ) -> Dict[str, Any]:
        run_id = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())
        created_at = utc_now_iso()

        run_state = {
            "run_id": run_id,
            "session_id": session_id,
            "status": "queued",
            "current_node": None,
            "created_at": created_at,
            "updated_at": created_at,
            "events": [],
            "outputs": {},
            "error": None,
            "input": {
                "pdf_url": pdf_url,
                "gcs_uri": gcs_uri,
                "use_demo_cache": use_demo_cache,
                "demo_cache_key": demo_cache_key,
            },
        }

        with self._lock:
            self._runs[run_id] = run_state

        thread = threading.Thread(
            target=lambda: asyncio.run(
                self._execute_run(
                    run_id=run_id,
                    pdf_url=pdf_url,
                    gcs_uri=gcs_uri,
                    session_id=session_id,
                    use_demo_cache=use_demo_cache,
                    demo_cache_key=demo_cache_key,
                )
            ),
            daemon=True,
        )
        thread.start()
        return await self.get_run(run_id)

    async def get_run(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(run_id)
            return {
                **run,
                "events": [event.copy() for event in run["events"]],
                "outputs": dict(run["outputs"]),
                "input": dict(run["input"]),
            }

    async def _execute_run(
        self,
        *,
        run_id: str,
        pdf_url: Optional[str],
        gcs_uri: Optional[str],
        session_id: str,
        use_demo_cache: bool,
        demo_cache_key: str,
    ) -> None:
        try:
            await self._set_status(run_id, "running", "pipeline", "Pipeline execution started.")
            self.sessions.setdefault(session_id, {"history": []})

            if use_demo_cache:
                await self._replay_demo_cache(run_id, session_id, demo_cache_key)
                await self._set_status(run_id, "completed", "pipeline", "Demo cache replay complete.")
                return

            requires_bucket = bool(pdf_url)
            config.validate_runtime_config(requires_bucket=requires_bucket)

            if pdf_url:
                await self._log_event(run_id, "upload", "running", f"Uploading PDF from {pdf_url}")
                gcs_uri = await upload_pdf_from_url(pdf_url, session_id)
            elif gcs_uri:
                await self._log_event(run_id, "upload", "completed", f"Using existing GCS URI {gcs_uri}")
            else:
                raise ValueError("Provide either pdf_url, gcs_uri, or use_demo_cache=true.")

            self.sessions[session_id]["gcs_uri"] = gcs_uri
            await self._store_output(run_id, "gcs_uri", gcs_uri)
            await self._persist_session(session_id)

            await self._log_event(run_id, "node0", "running", "Running scope validation.")
            node0_result = await run_node0(gcs_uri)
            node0_payload = node0_result.model_dump()
            self.sessions[session_id]["node0_result"] = node0_payload
            await self._store_output(run_id, "node0", node0_payload)
            await self._persist_session(session_id)
            await self._log_event(run_id, "node0", "completed", f"Scope result: {node0_result.result}")

            if node0_result.result != "PASS":
                await self._set_status(run_id, "failed", "node0", node0_result.reason)
                return

            await self._log_event(run_id, "node1", "running", "Extracting architecture blueprint.")
            blueprint = await run_node1(gcs_uri, pre_validated=True)
            blueprint_payload = blueprint.model_dump()
            self.sessions[session_id]["node1_result"] = blueprint_payload
            await self._store_output(run_id, "node1", blueprint_payload)
            await self._persist_session(session_id)
            await self._log_event(run_id, "node1", "completed", "Blueprint extracted.")

            await self._log_event(run_id, "node2", "running", "Generating PyTorch scaffold.")
            node2_result = run_node2(blueprint_payload)
            self.sessions[session_id]["node2_result"] = node2_result
            await self._store_output(run_id, "node2", node2_result)
            await self._persist_session(session_id)
            await self._log_event(run_id, "node2", "completed", "Node 2 finished.")

            await self._log_event(run_id, "node3", "running", "Generating CUDA blueprint.")
            node3_result = run_node3(blueprint_payload)
            self.sessions[session_id]["node3_result"] = node3_result
            await self._store_output(run_id, "node3", node3_result)
            await self._persist_session(session_id)
            await self._log_event(run_id, "node3", "completed", "Node 3 finished.")

            await self._set_status(run_id, "completed", "pipeline", "Pipeline execution finished.")
        except Exception as exc:
            await self._set_status(run_id, "failed", await self._current_node(run_id), str(exc))

    async def _replay_demo_cache(self, run_id: str, session_id: str, cache_key: str) -> None:
        cache = load_demo_cache(cache_key)
        manifest = cache["manifest"]

        self.sessions[session_id]["demo_cache"] = cache_key
        await self._store_output(run_id, "demo_cache", manifest)

        node0_payload = cache["node0_result"]
        blueprint_payload = cache["blueprint"]
        node2_payload = cache["node2_result"]
        node3_payload = cache["node3_result"]

        await self._log_event(run_id, "node0", "running", "Replaying cached scope validation.")
        await asyncio.sleep(0.05)
        self.sessions[session_id]["node0_result"] = node0_payload
        await self._store_output(run_id, "node0", node0_payload)
        await self._persist_session(session_id)
        await self._log_event(run_id, "node0", "completed", f"Scope result: {node0_payload['result']}")

        await self._log_event(run_id, "node1", "running", "Replaying cached research contract.")
        await asyncio.sleep(0.05)
        blueprint = ArchitectureBlueprint(**blueprint_payload)
        self.sessions[session_id]["node1_result"] = blueprint.model_dump()
        await self._store_output(run_id, "node1", blueprint.model_dump())
        await self._persist_session(session_id)
        await self._log_event(run_id, "node1", "completed", "Blueprint loaded from cache.")

        await self._log_event(run_id, "node2", "running", "Replaying cached PyTorch scaffold output.")
        await asyncio.sleep(0.05)
        self.sessions[session_id]["node2_result"] = node2_payload
        await self._store_output(run_id, "node2", node2_payload)
        await self._persist_session(session_id)
        await self._log_event(run_id, "node2", "completed", "PyTorch scaffold cache loaded.")

        await self._log_event(run_id, "node3", "running", "Replaying cached CUDA blueprint output.")
        await asyncio.sleep(0.05)
        self.sessions[session_id]["node3_result"] = node3_payload
        await self._store_output(run_id, "node3", node3_payload)
        await self._persist_session(session_id)
        await self._log_event(run_id, "node3", "completed", "CUDA blueprint cache loaded.")

    async def _current_node(self, run_id: str) -> Optional[str]:
        with self._lock:
            run = self._runs.get(run_id)
            return None if run is None else run["current_node"]

    async def _set_status(self, run_id: str, status: str, node: Optional[str], message: str) -> None:
        with self._lock:
            run = self._runs[run_id]
            run["status"] = status
            run["current_node"] = node
            run["updated_at"] = utc_now_iso()
            if status == "failed":
                run["error"] = message
            run["events"].append(
                {
                    "sequence": len(run["events"]) + 1,
                    "timestamp": run["updated_at"],
                    "node": node,
                    "status": status,
                    "message": message,
                }
            )

    async def _log_event(self, run_id: str, node: str, status: str, message: str) -> None:
        with self._lock:
            run = self._runs[run_id]
            run["current_node"] = node
            run["updated_at"] = utc_now_iso()
            run["events"].append(
                {
                    "sequence": len(run["events"]) + 1,
                    "timestamp": run["updated_at"],
                    "node": node,
                    "status": status,
                    "message": message,
                }
            )

    async def _store_output(self, run_id: str, key: str, value: Any) -> None:
        with self._lock:
            run = self._runs[run_id]
            run["outputs"][key] = value
            run["updated_at"] = utc_now_iso()

    async def _persist_session(self, session_id: str) -> None:
        if self.persist_session is not None:
            await self.persist_session(session_id)


async def sse_event_stream(manager: PipelineManager, run_id: str):
    next_index = 0
    while True:
        run = await manager.get_run(run_id)
        events = run["events"]

        while next_index < len(events):
            event = events[next_index]
            next_index += 1
            yield f"event: status\ndata: {json.dumps(event)}\n\n"

        if run["status"] in {"completed", "failed"}:
            break

        await asyncio.sleep(0.25)
