# Agentic MedAssist

An advanced, asynchronous AI agent workspace for medical/technical document processing,
multi-modal RAG retrieval pipelines, and automated reasoning workflows.  Built with a
LangGraph-powered agent graph that routes clinical queries through triage, planning,
local Qdrant retrieval, optional web search, evidence synthesis, self-criticism, and
safety guardrails before returning a structured answer.

---

## Tech Stack

| Layer | Libraries / Components |
|-------|------------------------|
| **Language / Runtime** | Python 3.11+, async/await throughout |
| **API** | FastAPI, Uvicorn, Pydantic v2 + pydantic-settings |
| **Agent Orchestration** | LangGraph (StateGraph) with triage / plan / search / synthesise / critic / escalate nodes |
| **LLM** | Groq (llama-3.3-70b-versatile) with structured fallback answers |
| **Vector Store** | Qdrant (client with async API + optional on-disk local mode) |
| **Embeddings** | ColPali Engine for multi-modal PDF page embeddings; deterministic fallback for tests |
| **Safety** | Dosage range guardrails, contraindication rule engine |
| **Observability** | Structured JSON audit log (Postgres via asyncpg) + structlog |
| **Database** | PostgreSQL for audit & auth (optional Supabase integration) |
| **Health Records** | ABDM / ABHA proxy endpoints for Ayushman Bharat Digital Mission flows |
| **Evaluation** | RAGAs evaluation harness with MacCrobat reference dataset |
| **Frontend** | Next.js 15 (TypeScript) chat UI (see `frontend/`) |
| **Testing** | pytest, pytest-asyncio, respx, FastAPI TestClient |
| **Deployment** | Docker compose with backend, frontend, Qdrant, and Postgres services |

---

## Repository Structure

```text
agentic-medassist/
├── backend/                 # Dockerfile for the FastAPI backend
├── configs/safety/          # YAML rules for dosage limits & contraindications
├── data/
│   ├── evaluation/          # Reference QA datasets (maccrobat.jsonl)
│   ├── processed/           # Post-ingestion manifest
│   └── raw/                 # Source PDFs (Gale Encyclopedia, etc.)
├── evals/                   # RAGAs evaluation scripts + reports
├── frontend/                # Next.js 15 chat UI
├── qdrant_db/               # On-disk local Qdrant store (not versioned)
├── src/
│   ├── agents/              # Legacy bridge: orchestrate_query() adapter
│   ├── api/
│   │   ├── v1/abha.py       # ABHA create & login proxy to ABDM sandbox
│   │   └── v1/chat.py       # Chat endpoint wired to LangGraph
│   │   └── main.py          # Legacy /api/v1/agent/query + /health endpoints
│   ├── core/config.py       # Legacy Settings (backwards-compatible)
│   ├── main.py              # *Primary* FastAPI entrypoint: /v1/chat, /v1/abha, /healthz
│   ├── observability/       # Structured audit logger (Postgres-backed)
│   ├── orchestration/
│   │   ├── main_graph.py    # Compiled LangGraph orchestrator (run_query())
│   │   ├── state.py         # Typed GraphState (TypedDict with reducer channels)
│   │   └── nodes/           # triage, plan, search_local, search_web, synthesise, critic, escalate, output
│   ├── pipeline/
│   │   ├── ingestion_manager.py  # Loads pre-built Qdrant collection + manifest (load_store())
│   │   ├── qdrant_store.py       # Async wrapper: init_collection, upsert_vectors, search_similar
│   │   └── query_embedder.py     # Deterministic L2-normed pseudo-embedder + interface
│   ├── retrieval/           # Hybrid RRF fusion (bm25 + maxsim + graph)
│   │                        # plus get_async_qdrant() singleton factory
│   ├── safety/              # Dosage guard & contraindication rule engine
│   └── settings.py          # Unified AppSettings class (new code should use this)
├── tests/
│   ├── conftest.py          # Test-wide env defaults
│   ├── e2e/test_clinician_flow.py  # End-to-end graph + chat endpoint exercise
│   └── unit/                # Embedding, ingestion manager, nodes, APIs, safety tests
├── .env.example             # Reference for required environment variables
├── docker-compose.yml       # Backend / Frontend / Qdrant / Postgres stack
├── pyproject.toml           # Build + pytest config (asyncio_mode = auto)
├── requirements.txt         # Python dependency manifest
└── README.md                # This file
```

---

## Quick Start (Local Development)

### 1. Bootstrap the Python environment

```powershell
# Windows (PowerShell 5)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` → `.env` at the project root and fill in the secrets you need.
Minimal working local config (no external services required for mock retrieval):

```env
ENVIRONMENT=dev
QDRANT_URL=
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/postgres
GROQ_API_KEY=your_groq_key_here
```

### 3. Start the backend API

```bash
# Primary entrypoint (LangGraph chat + ABHA routes)
uvicorn src.main:app --host 0.0.0.0 --port 8080 --reload

# Legacy entrypoint (backwards-compatible /api/v1/agent/query)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Run the frontend

```bash
cd frontend
npm install
npm run dev   # http://localhost:3000
```

### 5. Send a test chat request

```bash
curl -X POST http://localhost:8080/v1/chat \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is the dosage guidance for amoxicillin in children?",
       "user_id": "clinician-42"
     }'
```

Legacy endpoint:

```bash
curl -X POST http://localhost:8000/api/v1/agent/query \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What are the diagnostic criteria for asthma?",
       "session_id": "sess-demo-001"
     }'
```

### 6. Run the test suite

```bash
pytest tests/ -v
```

Expect **62 passing tests** (e2e + unit) when the environment is configured per
`tests/conftest.py` defaults.

---

## Docker Compose (Full Stack)

```bash
docker compose up --build
```

Starts four services on a dedicated bridge network:

| Service | Port | Purpose |
|---------|------|---------|
| **backend**  | `8080` | FastAPI API (uvicorn `src.main:app`) |
| **frontend** | `3000` | Next.js chat UI |
| **qdrant**   | `6333` (HTTP) / `6334` (gRPC) | Vector database, persisted in named volume `qdrant_storage` |
| **db**       | `5432` | PostgreSQL 16 Alpine, persisted in `postgres_data` volume |

The `backend` container receives `QDRANT_URL=http://qdrant:6333` and
`POSTGRES_DSN=postgresql://postgres:postgres@db:5432/postgres` automatically
via `docker-compose.yml`.

---

## Environment Variables

| Variable | Default | Required? | Description |
|----------|---------|-----------|-------------|
| `ENVIRONMENT` | `dev` | no | Runtime label: `dev`, `staging`, or `prod` |
| `LOG_LEVEL` | `INFO` | no | structlog / standard library log level |
| **Vector DB** | | | |
| `QDRANT_URL` | *(empty → on-disk `./qdrant_db/`)* | for remote Qdrant | Full service URL, e.g. `http://qdrant:6333` or Cloud cluster |
| `QDRANT_API_KEY` | *(empty)* | for Qdrant Cloud | API key credential (kept secret via `SecretStr`) |
| **Postgres / Audit** | | | |
| `POSTGRES_DSN` | *(empty)* | if audit persistence is wanted | `postgresql://user:pass@host:port/dbname`.  Accepts SSL params. |
| `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` | *(empty)* | Supabase deployments only | Alternative Postgres + GoTrue backend |
| **LLM Providers** | | | |
| `GROQ_API_KEY` | *(empty)* | recommended | Used for real Groq LLM calls in the synthesiser; falls back to deterministic template answers when unset |
| `DEEPSEEK_API_KEY` | *(empty)* | optional | Reserved for future tool-selection models |
| **Observability** | | | |
| `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` | *(empty)* | optional | Langfuse tracing integration |
| **ABDM / ABHA** | | | |
| `ABDM_SANDBOX_BASE_URL` | *(empty)* | for ABHA flows | Base URL of the Ayushman Bharat Digital Mission sandbox environment |
| **Legacy Back-Compat** | | | |
| `DEBUG` | `False` | no | FastAPI debug flag (legacy `src/api/main.py` only) |
| `PORT` | `8000` | no | Legacy API listener port |
| `DATABASE_URL` | *(empty)* | no | Alias for `POSTGRES_DSN` used by the original `Settings` class |
| `HUGGINGFACE_TOKEN` | *(empty)* | when ColPali model is private | HF auth for gated multi-modal models |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | no | Legacy split host+port config; prefer `QDRANT_URL` in new work |
| `COLPALI_MODEL_NAME` | `vidore/colpali-v1.2` | no | Model identifier for multi-modal embeddings |

All secrets are wrapped in `SecretStr` / equivalent so they never appear in
reprs, logs, or crash dumps.

---

## LangGraph Agent Pipeline

The compiled graph in `src/orchestration/main_graph.py` executes the following
sequence for every incoming chat request:

```
START → triage → plan → [COND: plan router → search_local]
  → search_local (Qdrant + embedder, fallback to keyword evidence)
  → search_web  (mock / future real web tool)
  → synthesise  (Groq → clinical answer with citations + safety check)
  → critic      (verdict: accept / insufficient / contradicted / unsafe)
  → [COND: critic router →
       accept        → output → END
       insufficient  → search_local (up to max_iterations) → re-synthesise
       unsafe / max  → escalate → END]
```

Key design properties:

- **Async-first:** every node, embedder, vector DB call, and audit write is
  `async def` and runs under `ainvoke`.
- **Graceful degradation:** Qdrant or LLM unavailable → nodes fall back to
  deterministic keyword evidence / template answers rather than 500-ing.
- **Audit every hop:** each node appends to `state["audit_trail"]` with
  latency, status, error, and details; the chat endpoint additionally writes
  a row to Postgres via `append_audit()`.
- **Safety gating:** contraindication and dosage rules execute inside the
  synthesiser / critic path and can flip the critic verdict to `unsafe`
  triggering the `escalate` node with a human-review message.

---

## Adding a New PDF to the Vector Store

1. Drop the PDF into `data/raw/`.
2. Run your ingestion pipeline (ColPali embed → chunk → upsert to Qdrant)
   against the same collection named in `ingestion_manifest.json`.
   Default collection name is **`medassist_pages`** (see `src/settings.py` +
   `src/pipeline/ingestion_manager.py`).
3. Update `ingestion_manifest.json` with the new file in `processed_files`
   and set `"status": "success"`.
4. Restart the backend.  `IngestionManager` picks up the manifest on startup
   and `search_local_node` will route real queries to Qdrant when it's reachable.

During development the pipeline works end-to-end without Qdrant – look for
`retrieval_mode: "fallback*"` in the audit trail to confirm you are exercising
the deterministic code path.

---

## Running the RAGAs Evaluation Suite

```bash
python -m evals.evaluate_ragas
```

Reports are written under `evals/reports/<timestamp>.json` and include
faithfulness, answer relevancy, context precision, and context recall scores
computed against the MacCrobat evaluation set.

---

## Testing & Linting

```bash
# Unit + E2E (62 tests, asyncio_mode=auto already configured in pyproject.toml)
pytest tests/ -v

# Coverage report (optional)
pip install pytest-cov
pytest tests/ --cov=src --cov-report=term-missing:skip-covered
```

Diagnostics: Trae's built-in `GetDiagnostics` is the recommended way to check
for import / type issues in the editor before pushing changes.

---

## Upgrading / Contributing Principles

- **Async every I/O:** no blocking DB, HTTP, or model calls.
- **Pydantic v2 everywhere:** input schemas for endpoints, node payloads, and
  safety rule outputs all use validated models (or strict TypedDicts for the
  reducer channels in LangGraph state).
- **Fail loudly in tests, gracefully in production:** all production nodes
  catch expected exceptions, degrade to a sensible fallback, and record the
  error in the audit trail.
- **Prefer `src.settings.AppSettings` in new code** over the legacy
  `src.core.config.Settings` singleton – the latter is kept for backwards
  compatibility with the original `/api/v1/agent/query` endpoint.
