# ArXiv Agent

An autonomous ML research-to-code scaffold assistant. Given an ArXiv PDF, it validates scope, extracts a structured research contract, and now includes a demo-friendly backend pipeline wrapper with cached replay support.
An autonomous ML research-to-code scaffold assistant. Given an ArXiv PDF URL or file upload, it validates ML scope, extracts a structured architecture blueprint, persists sessions to GCS, and supports multi-turn streaming chat grounded in the paper.

---

| Node | Model | Role |
|------|-------|------|
| Node 0 | Gemini 1.5 Flash | Scope validator, PASS or FAIL |
| Node 1 | Gemini 1.5 Pro | Research ingestor, extracts JSON contract |
| Node 2 | - | PyTorch scaffold generator (stub client today) |
| Node 3 | - | CUDA hardware blueprint generator (stub client today) |

A `/chat` endpoint allows context-aware Q&A about processed papers using the pipeline output stored in the session. A `/run` endpoint launches the full Node 0 -> Node 3 flow and exposes polling plus SSE status updates.
## What is implemented (Phase 1)

### Pipeline
| Node | Model | Role | Status |
|------|-------|------|--------|
| Node 0 | Gemini 2.5 Flash | Scope validator — strict PASS/FAIL for ML papers | Working |
| Node 1 | Gemini 2.5 Pro | Architecture ingestor — extracts typed JSON blueprint | Working |
| Node 2 | — | PyTorch scaffold generator | Stub |
| Node 3 | — | CUDA hardware blueprint generator | Stub |

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/upload` | Accept PDF via `pdf_url` (form) or multipart file upload; stores to GCS |
| POST | `/run/node0` | Scope validation; result stored in session |
| POST | `/run/node1` | Architecture extraction; honours stored node0 result to avoid re-validation |
| POST | `/chat` | SSE streaming chat grounded in session blueprint + conversation history |

### Scope validation (two-layer)
- **Layer 1 — Node 0 (Flash):** Strict prompt that requires the paper's *primary contribution* to be a new ML/AI model or architecture. Applies ML as a tool, surveys, and non-ML fields all fail.
- **Layer 2 — Node 1 (Pro):** If Node 0 was not pre-run, the extraction model also self-validates. If Node 0 already passed, extraction is forced (no second-guess). This prevents nondeterministic scope disagreement between the two calls.

### Blueprint schema
Node 1 returns a typed `ArchitectureBlueprint` (Pydantic, `additionalProperties: false`):
```json
{
  "model_type": "transformer | cnn | rnn | gan | vae | diffusion",
  "architecture": {
    "d_model": 512, "n_heads": 8, "n_layers": 6,
    "d_ff": 2048, "vocab_size": 37000, "max_seq_len": 2048
  },
  "objective": "cross-entropy with label smoothing",
  "key_operations": ["Multi-Head Attention", "Position-wise FFN"],
  "math_notes": "d_model must be divisible by n_heads"
}
```

### Chat (SSE streaming)
`POST /chat` returns `text/event-stream`. Events:
```
data: {"type":"status","text":"Thinking..."}
data: {"type":"status","text":"Reading paper context..."}
data: {"type":"status","text":"Generating response..."}
data: {"type":"token","text":"<chunk>"}   ← many, streamed live from Gemini
data: {"type":"done","text":""}
```
Each completed exchange (user + assistant) is appended to `session["history"]` and included in the next prompt (last 10 pairs), enabling multi-turn conversation.

### Session persistence
- Sessions are keyed by UUID and stored in-memory for active requests.
- After every state-changing call (`/upload`, `/run/node0`, `/run/node1`) the session is **awaited** to GCS (`sessions/{session_id}.json`) before the response is returned — ensuring durability across server restarts.
- Chat history is persisted fire-and-forget after each exchange.
- On startup, all sessions are reloaded from GCS automatically.

### Error handling
| Scenario | HTTP status |
|---|---|
| Invalid/unreachable PDF URL | 422 |
| Non-ML paper rejected by scope check | 400 |
| Vertex AI model error (wrong model, quota) | 502 |
| Session not found | 404 |
| Unexpected server error | 500 |

### Async
All I/O is non-blocking: `generate_content_async` for Gemini, `httpx.AsyncClient` for PDF download, `asyncio.to_thread` for GCS uploads.

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure environment**
```bash
cp .env.example .env
# Fill in GCP_PROJECT_ID, GCP_BUCKET_NAME, GCP_REGION, GOOGLE_APPLICATION_CREDENTIALS
```

Place your GCP service account key at `gcp-key.json` in the project root.

**3. Run the server**
```bash
uvicorn backend.main:app --reload
```

API at `http://localhost:8000` — interactive docs at `http://localhost:8000/docs`.

---

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/upload` | Upload PDF via URL (`pdf_url` form field) or file (`file` form field) |
| POST | `/run/node0` | Scope validation only |
| POST | `/run/node1` | Paper ingestion only |
| POST | `/run` | Launch the full pipeline in the background |
| GET | `/runs/{run_id}` | Poll status, events, and outputs |
| GET | `/runs/{run_id}/events` | Stream node status updates via SSE |
| POST | `/chat` | Context-aware chat about the processed paper |

### Typical flow
```text
POST /run           -> { run_id, poll_url, stream_url }
GET /runs/{run_id}  -> { status, events, outputs }
POST /chat          -> { response: "..." }
```

## Demo

Live run:
```bash
python demo/run_demo.py --pdf-url https://arxiv.org/pdf/1706.03762.pdf
```

Offline cached rehearsal:
```bash
python demo/run_demo.py --use-demo-cache
```

Inspect environment values:
```bash
python demo/run_demo.py --show-env
## Typical flow

```
POST /upload        → { session_id, gcs_uri }
POST /run/node0     → { result: "PASS", reason: "..." }
POST /run/node1     → { blueprint: { model_type, architecture, ... } }
POST /chat          → SSE stream  (resumable across restarts)
```

`/chat` is resumable — send the same `session_id` after a server restart and history is recovered from GCS.

---

## Tests

```bash
pytest          # or: .venv/bin/python -m pytest -q
```

6 unit tests covering node0 (pass/fail/fence-stripping) and node1 (extraction, fence-stripping, scope rejection).
