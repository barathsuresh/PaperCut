# PaperCut — ArXiv Agent

**Cut through the paper.** PaperCut analyses ML research papers, extracts a structured research contract, generates a PyTorch implementation scaffold and CUDA optimisation blueprint, then lets you explore the paper through a streaming chat interface.

---

## Features

- **Scope validation** — Gemini Flash quickly checks whether a paper is an ML research paper before spending tokens on deeper analysis
- **Structured extraction** — Gemini Pro extracts a `ResearchContract` (model type, architecture, training recipe, datasets, etc.)
- **Code generation** — Node 2 generates a PyTorch scaffold; Node 3 produces a CUDA/hardware blueprint
- **Streaming pipeline** — live progress updates via Server-Sent Events as each node completes
- **Streaming chat** — RAG-style Q&A about the paper with token-level streaming
- **GCS persistence** — PDFs and generated artefacts are stored in Google Cloud Storage; sessions survive server restarts
- **VSCode-style code viewer** — file tree + syntax-highlighted code panel with line-wrap and font-size controls, resizable
- **ArXiv URL support** — paste an `/abs/` URL and PaperCut converts it to the PDF automatically
- **Global drag-and-drop** — drop a PDF anywhere on the page to start a new analysis
- **Session management** — rename, delete, and search across sessions; inline double-click rename in sidebar
- **Toast notifications** — non-blocking success/error/info toasts throughout the UI

---

## Pipeline

```
Node 0  Gemini Flash  →  scope validate  →  PASS / FAIL
           ↓ PASS only
Node 1  Gemini Pro    →  extract ResearchContract JSON
           ↓
Node 2                →  PyTorch scaffold
           ↓
Node 3                →  CUDA / hardware blueprint
```

| Node | Model | Role | Status |
|------|-------|------|--------|
| Node 0 | Gemini 2.5 Flash | Scope validator — strict PASS/FAIL for ML papers | Working |
| Node 1 | Gemini 2.5 Pro | Architecture ingestor — extracts typed JSON blueprint | Working |
| Node 2 | — | PyTorch scaffold generator | Teammate stub |
| Node 3 | — | CUDA hardware blueprint generator | Teammate stub |

The graph is compiled with **LangGraph** in `backend/graph.py`. FastAPI routes in `backend/routes/` call node functions directly for per-step endpoints and stream the full pipeline via SSE.

---

## Directory layout

```
ArXiv_Agent/
├── backend/
│   ├── main.py                  # FastAPI app, CORS, lifespan
│   ├── config.py                # Env vars, model strings
│   ├── graph.py                 # LangGraph pipeline
│   ├── app_state.py             # In-memory session store
│   ├── session_store.py         # GCS session persistence
│   ├── nodes/
│   │   ├── node0_validator.py   # Gemini Flash scope check
│   │   ├── node1_ingestor.py    # Gemini Pro extraction
│   │   ├── node2_client.py      # PyTorch scaffold (teammate)
│   │   └── node3_client.py      # CUDA blueprint (teammate)
│   ├── routes/
│   │   ├── sessions.py          # CRUD + artifact endpoints
│   │   ├── pipeline.py          # /run/pipeline/stream SSE
│   │   ├── chat.py              # /chat/stream SSE
│   │   └── health.py            # GET /health
│   ├── schemas/                 # Pydantic models
│   └── tools/
│       ├── gemini_client.py     # Vertex AI SDK wrapper
│       └── artifact_store.py    # GCS artifact upload/download
└── frontend/
    ├── src/
    │   ├── App.jsx              # Root state, routing
    │   ├── api/client.js        # All fetch/SSE calls
    │   ├── components/
    │   │   ├── Sidebar.jsx      # Session list, search, rename
    │   │   ├── UploadPanel.jsx  # PDF upload + ArXiv URL
    │   │   ├── PipelineProgress.jsx
    │   │   ├── ChatView.jsx     # Messages, starter questions
    │   │   ├── ChatMessage.jsx  # Markdown, copy button
    │   │   ├── CodePanel.jsx    # File tree + syntax viewer
    │   │   ├── MessageInput.jsx
    │   │   ├── SplashScreen.jsx
    │   │   └── Toast.jsx
    │   ├── hooks/useToast.js
    │   └── index.css
    └── package.json
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| Node.js | 18+ |
| Google Cloud SDK | any recent |
| GCP project | with Vertex AI + GCS enabled |

---

## Setup

### 1. Clone and create the virtual environment

```bash
git clone <repo-url>
cd ArXiv_Agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file at the repo root:

```env
GCP_PROJECT_ID=your-project-id
GCP_BUCKET_NAME=your-gcs-bucket
GCP_REGION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=./gcp-key.json

# Optional — defaults shown
GEMINI_FLASH_MODEL=gemini-2.5-flash
GEMINI_PRO_MODEL=gemini-2.5-pro
```

Place your GCP service-account key at `gcp-key.json` (gitignored). The key needs **Storage Object Admin** and **Vertex AI User** roles.

### 3. Create the GCS bucket

```bash
gcloud storage buckets create gs://your-gcs-bucket --location=us-central1
```

### 4. Install frontend dependencies

```bash
cd frontend
npm install
```

---

## Running

### Backend

```bash
# from repo root
uvicorn backend.main:app --reload
```

Server starts on `http://localhost:8000`. Interactive API docs at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend
npm run dev
```

Vite dev server starts on `http://localhost:5173`.

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/sessions` | List all sessions |
| `POST` | `/sessions/upload` | Upload PDF, create session |
| `GET` | `/sessions/{id}` | Get session detail |
| `PATCH` | `/sessions/{id}/name` | Rename session |
| `DELETE` | `/sessions/{id}` | Delete session + GCS artefacts |
| `GET` | `/sessions/{id}/history` | Chat message history |
| `GET` | `/sessions/{id}/artifacts` | List generated files |
| `GET` | `/sessions/{id}/artifacts/{group}/{file}` | Download file content |
| `POST` | `/run/pipeline/stream` | Run full pipeline (SSE) |
| `POST` | `/run/node0` | Run scope validation only |
| `POST` | `/run/node1` | Run extraction only |
| `POST` | `/run/node2` | Run scaffold generation only |
| `POST` | `/run/node3` | Run CUDA blueprint only |
| `POST` | `/chat/stream` | Chat with paper context (SSE) |

### SSE event shapes

**Pipeline stream** (`/run/pipeline/stream`):
```
data: {"type":"node_start","node":0}
data: {"type":"node_done","node":0,"scope_valid":true}
data: {"type":"node_done","node":1,"model_type":"transformer"}
data: {"type":"done","scope_valid":true}
data: {"type":"error","message":"..."}
```

**Chat stream** (`/chat/stream`):
```
data: {"type":"status","text":"Thinking..."}
data: {"type":"token","text":"<chunk>"}
data: {"type":"done"}
```

---

## GCS layout

```
gs://{bucket}/
└── papers/
    └── {session_id}/
        ├── paper.pdf
        ├── session.json
        ├── scaffold/        # Node 2 output files
        └── hardware/        # Node 3 output files
```

---

## Scope validation

Two-layer validation prevents wasting Pro tokens on non-ML papers:

- **Layer 1 — Node 0 (Flash):** Strict prompt requiring the paper's *primary contribution* to be a new ML/AI model or architecture. Papers that merely apply ML as a tool, surveys, and non-ML papers all fail.
- **Layer 2 — Node 1 (Pro):** If Node 0 was not pre-run, the extraction model self-validates. If Node 0 already passed, extraction is forced — no second-guessing — preventing nondeterministic scope disagreement.

---

## Key data models

**`ResearchContract`** (`backend/schemas/contract.py`) — structured extraction from Node 1. Contains model type, architecture details, training recipe, datasets, evaluation metrics, and novelty claims.

**`AgentState`** (`backend/schemas/state.py`) — LangGraph `TypedDict` threaded through all nodes. Holds `pdf_gcs_uri`, `scope_valid`, `scope_reason`, `blueprint`, `scaffold_code`, `cuda_blueprint`, `session_id`, and `error`.

**`ScopeValidationResult`** (`backend/schemas/validator.py`) — Node 0 output: `result` (`PASS`/`FAIL`) and `reason` string.

---

## Testing

```bash
# Run all tests
pytest

# Run a specific test file
pytest backend/tests/test_node0.py -v
```

---

## Notes

- Sessions are loaded from GCS on server start, so they survive restarts.
- Node 2 and Node 3 are teammate-owned stubs — they return `{"status": "not_implemented"}` until implemented.
- All Gemini JSON responses are passed through `strip_markdown_fences()` before `json.loads()` to handle code-fenced model responses.
- Vertex AI is initialised lazily on first call via `vertexai.init()` in `gemini_client.py`.
- All I/O is non-blocking: `generate_content_async` for Gemini, `asyncio.to_thread` for GCS and Node 2/3, fire-and-forget `asyncio.create_task` for background artefact uploads.
