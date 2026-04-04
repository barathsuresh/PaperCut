# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run server (from repo root)
uvicorn backend.main:app --reload

# Run all tests
pytest

# Run a single test file
pytest backend/tests/test_node0.py -v
```

## Architecture

**4-node LangGraph pipeline** triggered per session:

```
Node 0 (Gemini Flash) → scope validate → PASS/FAIL
    ↓ PASS only
Node 1 (Gemini Pro)   → extract ResearchContract JSON
    ↓
Node 2                → PyTorch scaffold (stub)
    ↓
Node 3                → CUDA blueprint (stub)
```

The graph lives in `backend/graph.py`. Individual nodes are also called directly from FastAPI endpoints in `backend/main.py` — the graph and the endpoints share the same node functions from `backend/nodes/`.

**Session store**: in-memory `dict` in `main.py`, keyed by `session_id` (UUID). Stores `gcs_uri`, `node0_result`, `node1_result` per session. Sessions are lost on server restart.

**Key data models** (`backend/schemas/`):
- `ResearchContract` — structured paper extraction output (Pydantic)
- `AgentState` — LangGraph TypedDict
- `ScopeValidationResult` — Node 0 output

**Gemini calls** go through `backend/tools/gemini_client.py` using the `vertexai` SDK (not `google-generativeai`). Model strings: `gemini-1.5-pro-001`, `gemini-1.5-flash-001`. `vertexai.init()` is called lazily on first use.

All Gemini JSON responses must be passed through `strip_markdown_fences()` before `json.loads()`.

## GCP Infrastructure

- **Project ID**: `arxivagent-492308`
- **Storage Bucket**: `arxiv-agent-papers` (region: `us-central1`)
- **Credentials**: `gcp-key.json` at repo root (gitignored)
- PDFs stored at `gs://{bucket}/papers/{session_id}/paper.pdf`

Environment loaded from `.env` via `python-dotenv` in `backend/config.py`.

## Stubs

`backend/nodes/node2_client.py` and `node3_client.py` return `{"status": "not_implemented"}` — teammates fill these in.
