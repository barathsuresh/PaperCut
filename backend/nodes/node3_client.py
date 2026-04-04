"""
Node 3 client — bridges the LangGraph graph to the real Hardware Blueprint Generator.

Translates between the graph's dict-in/dict-out interface and
nodes.node3_hardware_blueprint.run_node3 which expects a scaffold directory
and returns an output directory Path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def run_node3(research_contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate annotated CUDA C++ stubs from the Node 2 PyTorch scaffold.

    Called by backend.graph._node3 with blueprint.model_dump().
    The real run_node3 reads scaffold files from outputs/pytorch_scaffold/
    (the default written by Node 2), so we pass scaffold_dir=None to use
    that default.

    Returns a dict written to cuda_blueprint in the graph state.
    """
    try:
        from nodes.node3_hardware_blueprint import run_node3 as _real_run_node3
        from nat import make_nat_caller_reason

        caller = make_nat_caller_reason()
        output_dir = _real_run_node3(
            scaffold_dir=None,   # defaults to outputs/pytorch_scaffold/
            nat_caller=caller,
        )

        # Read back results for the graph state
        stub_files = [f.name for f in output_dir.glob("*.cu")]

        bottleneck_path = output_dir / "bottleneck_analysis.json"
        bottlenecks = (
            json.loads(bottleneck_path.read_text())
            if bottleneck_path.exists()
            else []
        )

        meta_path = output_dir / "hardware_blueprint_meta.json"
        meta = (
            json.loads(meta_path.read_text())
            if meta_path.exists()
            else {}
        )

        logger.info(
            "Node 3 completed — %d CUDA stubs in %s",
            len(stub_files), output_dir,
        )

        return {
            "status": "completed",
            "output_dir": str(output_dir),
            "stub_files": stub_files,
            "bottlenecks": bottlenecks,
            "meta": meta,
        }

    except Exception as e:
        logger.error("Node 3 failed: %s", e)
        return {
            "status": "error",
            "error": str(e),
        }
