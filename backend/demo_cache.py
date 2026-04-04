import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_CACHE_ROOT = PROJECT_ROOT / "demo_cache"


class DemoCacheError(FileNotFoundError):
    pass


def demo_cache_path(cache_key: str) -> Path:
    return DEMO_CACHE_ROOT / cache_key


def load_demo_cache(cache_key: str) -> Dict[str, Any]:
    cache_dir = demo_cache_path(cache_key)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise DemoCacheError(
            f"Demo cache '{cache_key}' was not found at {manifest_path}."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "cache_key": cache_key,
        "cache_dir": str(cache_dir),
        "manifest": manifest,
        "node0_result": _load_json(cache_dir / "node0_result.json"),
        "blueprint": _load_json(cache_dir / "blueprint.json"),
        "node2_result": _load_json(cache_dir / "node2_result.json"),
        "node3_result": _load_json(cache_dir / "node3_result.json"),
    }


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise DemoCacheError(f"Expected demo cache file at {path} was not found.")
    return json.loads(path.read_text(encoding="utf-8"))
