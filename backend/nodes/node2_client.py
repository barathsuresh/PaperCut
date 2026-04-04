"""
Node 2 client — bridges the LangGraph graph to the real PyTorch Architect.
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

    Returns a dict whose "status" key is "completed" or "error".
    """
    tmp_path: Path | None = None
    try:
        from nat import make_nat_caller_code
        from nat.nat_client import NATAuthError, NATError, NATTimeoutError
        from nodes.node2_pytorch_architect import run_node2 as _real_run_node2

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(research_contract, tmp)
            tmp_path = Path(tmp.name)

        caller = make_nat_caller_code()
        output_dir = _real_run_node2(blueprint_path=tmp_path, nat_caller=caller)

        if not output_dir.exists():
            raise FileNotFoundError(
                f"Node 2 output directory not found after generation: {output_dir}"
            )

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

    except NATTimeoutError as e:
        logger.error("Node 2 NAT timeout: %s", e)
        return {"status": "error", "error": str(e), "error_type": "timeout"}
    except NATAuthError as e:
        logger.error("Node 2 NAT auth error: %s", e)
        return {"status": "error", "error": str(e), "error_type": "auth"}
    except NATError as e:
        logger.error("Node 2 NAT error: %s", e)
        return {"status": "error", "error": str(e), "error_type": "nat"}
    except Exception as e:
        logger.error("Node 2 failed: %s", e, exc_info=True)
        return {"status": "error", "error": str(e)}
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
