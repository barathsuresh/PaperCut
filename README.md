# ArXiv Agent

An autonomous ML research-to-code scaffold assistant. Given an ArXiv PDF, it validates scope, extracts a structured research contract, and (in future phases) generates PyTorch scaffolding and a CUDA hardware blueprint.

## Pipeline

| Node | Model | Role |
|------|-------|------|
| Node 0 | Gemini 1.5 Flash | Scope validator — PASS/FAIL |
| Node 1 | Gemini 1.5 Pro | Research ingestor — extracts JSON contract |
| Node 2 | — | PyTorch scaffold generator *(stub)* |
| Node 3 | — | CUDA hardware blueprint generator *(stub)* |

A `/chat` endpoint allows context-aware Q&A about processed papers using the pipeline output stored in the session.

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

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/upload` | Upload PDF via URL (`pdf_url` form field) or file (`file` form field) |
| POST | `/run/node0` | Scope validation — returns `PASS`/`FAIL` |
| POST | `/run/node1` | Paper ingestion — returns JSON contract |
| POST | `/chat` | Context-aware chat about the processed paper |

### Typical flow
```
POST /upload        → { session_id, gcs_uri }
POST /run/node0     → { result: "PASS", reason: "..." }
POST /run/node1     → { contract: { title, authors, ... } }
POST /chat          → { response: "..." }
```

## Tests

```bash
pytest
```
