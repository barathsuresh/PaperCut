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
Node 0 (Gemini Flash)      → scope validate → PASS/FAIL
    ↓ PASS only
Node 1 (Gemini Pro)        → extract ResearchContract JSON
    ↓
Node 2 (Qwen2.5-Coder 32B) → PyTorch scaffold (model.py, train.py, dataset.py, config.yaml)
    ↓
Node 3 (Nemotron Super 49B)→ CUDA annotated stubs + bottleneck analysis
```

The graph lives in `backend/graph.py`. The real node implementations are in `nodes/` (repo root, not `backend/nodes/`). `backend/nodes/node2_client.py` and `node3_client.py` are thin bridges that import from `nodes/` and wrap errors.

Node 2 and Node 3 use **NVIDIA NIM API** (OpenAI-compatible) via the `nat/` package:
- `nat/nat_client.py` — HTTP caller with timeout/retry logic; raises `NATError`, `NATTimeoutError`, `NATAuthError`
- `nat/nat_config.py` — reads `NVIDIA_API_KEY`, `NVIDIA_API_BASE`, `NAT_MODEL_CODE`, `NAT_MODEL_REASON` from `.env`
- `nat/__init__.py` — exports `make_nat_caller_code()` and `make_nat_caller_reason()` factory functions

Node 3 retries up to 2 times on `NATError` (Nemotron occasionally returns null content).

**Session store**: in-memory `dict` in `backend/app_state.py`, keyed by `session_id` (UUID). Sessions are loaded from GCS on startup and persist across restarts.

**Key data models** (`backend/schemas/`):
- `ResearchContract` — structured paper extraction output (Pydantic)
- `AgentState` — LangGraph TypedDict
- `ScopeValidationResult` — Node 0 output
- `SessionData` — TypedDict stored in app_state and GCS

**Service layer** (`backend/services/`):
- `pipeline_runtime.py` — shared helpers for running codegen nodes, uploading artefacts, SSE formatting
- `pipeline_state.py` — applies node results into SessionData
- `session_views.py` — shapes SessionData into API response dicts

**Chat** (`backend/chat/chat_handler.py`):
- Builds context from blueprint + scaffold files (model.py, train.py, dataset.py, config.yaml)
- Files read from local `outputs/` first, falls back to GCS via `download_artifact()`
- Truncates each file to `_MAX_FILE_CHARS` (3000) to avoid token explosion
- Keeps last `_MAX_HISTORY_PAIRS` (10) exchanges in prompt

**Gemini calls** go through `backend/tools/gemini_client.py` using the `google-genai` SDK. Model strings configurable via `.env` (`GEMINI_FLASH_MODEL`, `GEMINI_PRO_MODEL`), defaulting to `gemini-2.5-flash` / `gemini-2.5-pro`.

All Gemini JSON responses must be passed through `strip_markdown_fences()` before `json.loads()`.

## Directory layout (non-obvious parts)

```
nodes/                        # Real Node 2 & 3 implementations (not backend/nodes/)
  node2_pytorch_architect.py  # Qwen2.5-Coder scaffold generator
  node3_hardware_blueprint.py # Nemotron CUDA stub generator
nat/                          # NVIDIA NIM API client package
  nat_client.py
  nat_config.py
contracts/                    # JSON schemas + sample contracts
  architecture_blueprint_schema.json
outputs/                      # Local artefact output root (gitignored)
  sessions/{session_id}/
    pytorch_scaffold/
    hardware_blueprint/
backend/chat/                 # Chat handler (separate from routes/)
  chat_handler.py
```

## GCP Infrastructure

- **Project ID**: `arxivagent-492308`
- **Storage Bucket**: `arxiv-agent-papers` (region: `us-central1`)
- **Credentials**: `gcp-key.json` at repo root (gitignored)
- PDFs stored at `gs://{bucket}/papers/{session_id}/paper.pdf`
- Artefacts stored at `gs://{bucket}/papers/{session_id}/scaffold/` and `.../hardware/`

Environment loaded from `.env` via `python-dotenv` in `backend/config.py`.

## Required `.env` keys

```
GCP_PROJECT_ID
GCP_BUCKET_NAME
GOOGLE_APPLICATION_CREDENTIALS   # default: ./gcp-key.json
GEMINI_FLASH_MODEL                # default: gemini-2.5-flash
GEMINI_PRO_MODEL                  # default: gemini-2.5-pro
NVIDIA_API_KEY
NVIDIA_API_BASE
NAT_MODEL_CODE                    # e.g. qwen/qwen2.5-coder-32b-instruct
NAT_MODEL_REASON                  # e.g. nvidia/nemotron-super-49b-v1
NAT_TIMEOUT                       # seconds, optional
```
