# ArXiv Agent

An autonomous ML research-to-code scaffold assistant. Given an ArXiv PDF, it validates scope, extracts a structured research contract, and now includes a demo-friendly backend pipeline wrapper with cached replay support.

## Pipeline

| Node | Model | Role |
|------|-------|------|
| Node 0 | Gemini 1.5 Flash | Scope validator, PASS or FAIL |
| Node 1 | Gemini 1.5 Pro | Research ingestor, extracts JSON contract |
| Node 2 | - | PyTorch scaffold generator (stub client today) |
| Node 3 | - | CUDA hardware blueprint generator (stub client today) |

A `/chat` endpoint allows context-aware Q&A about processed papers using the pipeline output stored in the session. A `/run` endpoint launches the full Node 0 -> Node 3 flow and exposes polling plus SSE status updates.

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure environment**
```bash
cp .env.example .env
# Fill in your GCP values
```

Place your GCP service account key at `gcp-key.json` in the project root.

**3. Run the server**
```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs are available at `http://localhost:8000/docs`.

## API

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
```

## Tests

```bash
pytest
```
