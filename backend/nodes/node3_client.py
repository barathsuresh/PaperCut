"""
Node 3 client — bridges the LangGraph graph to the real Hardware Blueprint Generator.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_OUTPUTS_ROOT = Path(__file__).resolve().parent.parent.parent / "outputs" / "sessions"


def _node3_output_dir(session_id: str | None, scaffold_dir: Path | None) -> Path:
    if session_id:
        return _OUTPUTS_ROOT / session_id / "hardware_blueprint"
    if scaffold_dir is not None:
        return scaffold_dir.parent / "hardware_blueprint"
    return Path(__file__).resolve().parent.parent.parent / "outputs" / "hardware_blueprint"


def run_node3(node2_result: Dict[str, Any], session_id: str | None = None) -> Dict[str, Any]:
    """
    Generate annotated CUDA C++ stubs from the Node 2 PyTorch scaffold.

    Accepts node2_result so it can pass the correct scaffold_dir to the
    real implementation rather than relying on a hardcoded default path.

    Returns a dict written to cuda_blueprint in the graph state.
    """
    try:
        from nat import make_nat_caller_reason
        from nat.nat_client import NATAuthError, NATError, NATTimeoutError
        from nodes.node3_hardware_blueprint import run_node3 as _real_run_node3

        # Use the output_dir written by Node 2; fall back to default if absent
        scaffold_dir: Path | None = None
        raw_dir = node2_result.get("output_dir") if node2_result else None
        if raw_dir:
            scaffold_dir = Path(raw_dir)
            if not scaffold_dir.exists():
                logger.warning(
                    "Node 3 — scaffold_dir does not exist: %s — using default", scaffold_dir
                )
                scaffold_dir = None

        caller = make_nat_caller_reason()

        # Retry up to 2 times when Nemotron returns null content
        _MAX_NODE3_RETRIES = 2
        last_nat_error: Exception | None = None
        output_dir = None
        for attempt in range(_MAX_NODE3_RETRIES + 1):
            try:
                output_dir = _real_run_node3(
                    scaffold_dir=scaffold_dir,
                    output_dir=_node3_output_dir(session_id, scaffold_dir),
                    nat_caller=caller,
                )
                break
            except NATError as e:
                last_nat_error = e
                if attempt < _MAX_NODE3_RETRIES:
                    logger.warning(
                        "Node 3 NATError attempt %d/%d — retrying: %s",
                        attempt + 1, _MAX_NODE3_RETRIES, e,
                    )
                else:
                    raise last_nat_error

        stub_files = [f.name for f in output_dir.glob("*.cu")]

        bottleneck_path = output_dir / "bottleneck_analysis.json"
        bottlenecks: list = []
        if bottleneck_path.exists():
            try:
                bottlenecks = json.loads(bottleneck_path.read_text())
            except json.JSONDecodeError as exc:
                logger.warning("Node 3 — could not parse bottleneck_analysis.json: %s", exc)

        meta_path = output_dir / "hardware_blueprint_meta.json"
        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError as exc:
                logger.warning("Node 3 — could not parse hardware_blueprint_meta.json: %s", exc)

        logger.info("Node 3 completed — %d CUDA stubs in %s", len(stub_files), output_dir)
        return {
            "status": "completed",
            "output_dir": str(output_dir),
            "stub_files": stub_files,
            "bottlenecks": bottlenecks,
            "meta": meta,
        }

    except NATTimeoutError as e:
        logger.error("Node 3 NAT timeout: %s", e)
        return {"status": "error", "error": str(e), "error_type": "timeout"}
    except NATAuthError as e:
        logger.error("Node 3 NAT auth error: %s", e)
        return {"status": "error", "error": str(e), "error_type": "auth"}
    except NATError as e:
        logger.error("Node 3 NAT error: %s", e)
        return {"status": "error", "error": str(e), "error_type": "nat"}
    except Exception as e:
        logger.error("Node 3 failed: %s", e, exc_info=True)
        return {"status": "error", "error": str(e)}
