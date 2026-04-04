import argparse
import json
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import config
from backend.demo_cache import load_demo_cache


RESET = "\033[0m"
COLORS = {
    "queued": "\033[90m",
    "running": "\033[94m",
    "completed": "\033[92m",
    "failed": "\033[91m",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal demo runner for ArXiv Agent.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--pdf-url", default=None)
    parser.add_argument("--gcs-uri", default=None)
    parser.add_argument("--demo-cache-key", default="attention_is_all_you_need")
    parser.add_argument("--use-demo-cache", action="store_true")
    parser.add_argument("--show-env", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.show_env:
        print(json.dumps(config.runtime_environment(), indent=2))
        return 0

    if args.use_demo_cache:
        replay_local_cache(args.demo_cache_key)
        return 0

    payload = {
        "pdf_url": args.pdf_url,
        "gcs_uri": args.gcs_uri,
        "use_demo_cache": False,
        "demo_cache_key": args.demo_cache_key,
    }
    return run_against_api(args.api_base, payload)


def replay_local_cache(cache_key: str) -> None:
    cache = load_demo_cache(cache_key)
    print(f"Demo cache: {cache_key}")
    print(f"Paper: {cache['manifest']['paper_title']}")
    print_event("node0", "completed", cache["node0_result"]["reason"])
    print_event("node1", "completed", "Blueprint loaded from cache.")
    print_event("node2", "completed", "PyTorch scaffold cache loaded.")
    print_event("node3", "completed", "CUDA blueprint cache loaded.")
    print(json.dumps(cache["node3_result"], indent=2))


def run_against_api(api_base: str, payload: dict) -> int:
    with httpx.Client(base_url=api_base, timeout=30.0) as client:
        response = client.post("/run", json=payload)
        response.raise_for_status()
        run = response.json()
        run_id = run["run_id"]
        print(f"Run ID: {run_id}")

        seen = 0
        while True:
            status_response = client.get(f"/runs/{run_id}")
            status_response.raise_for_status()
            state = status_response.json()
            events = state["events"]

            while seen < len(events):
                event = events[seen]
                seen += 1
                print_event(event["node"] or "pipeline", event["status"], event["message"], event["timestamp"])

            if state["status"] in {"completed", "failed"}:
                print(json.dumps(state["outputs"], indent=2))
                return 0 if state["status"] == "completed" else 1

            time.sleep(0.5)


def print_event(node: str, status: str, message: str, timestamp: str | None = None) -> None:
    ts = timestamp or time.strftime("%Y-%m-%dT%H:%M:%S")
    color = COLORS.get(status, "")
    print(f"{color}[{ts}] {node:<8} {status:<10} {message}{RESET}")


if __name__ == "__main__":
    raise SystemExit(main())
