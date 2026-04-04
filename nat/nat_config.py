"""
NAT configuration — reads all values from .env via environment variables.
No hardcoded strings. Load .env before importing this module.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path, override=True)


def _require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Missing required env var: {key}. Check .env file.")
    return val


NVIDIA_API_BASE: str = _require("NVIDIA_API_BASE")
NVIDIA_API_KEY: str = _require("NVIDIA_API_KEY")
NAT_MODEL_CODE: str = _require("NAT_MODEL_CODE")
NAT_MODEL_REASON: str = _require("NAT_MODEL_REASON")
NAT_TIMEOUT: int = int(os.environ.get("NAT_TIMEOUT", "120"))
