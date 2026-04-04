"""
Node 2 client — bridges the LangGraph graph to the real PyTorch Architect.

Translates between the graph's dict-in/dict-out interface and
nodes.node2_pytorch_architect.run_node2 which expects a JSON file path
and returns an output directory Path.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def run_node2(research_contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a PyTorch scaffold from an architecture blueprint dict.

    Called by backend.graph._node2 with blueprint.model_dump().
    Returns a dict whose "status" key is read by the graph layer.
    """
    try:
        from nodes.node2_pytorch_architect import run_node2 as _real_run_node2
        from nat import make_nat_caller_code

        # run_node2 reads a JSON file, so write the dict to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            json.dump(research_contract, tmp)
            tmp_path = Path(tmp.name)

        caller = make_nat_caller_code()
        output_dir = _real_run_node2(
            blueprint_path=tmp_path,
            nat_caller=caller,
        )

        # Read back generated files for the graph state
        files_generated = {
            f.name: f.stat().st_size
            for f in output_dir.iterdir()
            if f.is_file()
        }

        logger.info("Node 2 completed — %d files in %s", len(files_generated), output_dir)

        return {
            "status": "completed",
            "output_dir": str(output_dir),
            "files": files_generated,
        }

    except Exception as e:
        logger.error("Node 2 failed: %s", e)
        return {
            "status": "error",
            "error": str(e),
        }
    finally:
        # Clean up temp file if it was created
        try:
            tmp_path.unlink(missing_ok=True)
        except NameError:
            pass
