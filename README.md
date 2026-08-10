# 🩺 Medical RAG Assistant

A clinical-grade, evidence-grounded **Medical Retrieval-Augmented Generation (RAG)** system built on the *Gale Encyclopedia of Medicine*.

Powered by **FastAPI**, **LangGraph**, **ChromaDB**, **BM25**, and **Groq (`llama-3.1-8b-instant`)**, featuring a modern glassmorphic web UI with interactive citations.

---

## 🌟 Key Features

- **Evidence-Grounded Generation**: Strictly answers based on encyclopedia sources with inline chunk citations (e.g. `[gale-p1933-c1899]`). Zero outside hallucinations.
- **Hybrid Retrieval (Dense + Sparse)**: Combines **ChromaDB** vector similarity (`sentence-transformers/all-MiniLM-L6-v2`) with **BM25Okapi** lexical search using **Reciprocal Rank Fusion (RRF, k=60)**.
- **Smart Symptom-Disease Disambiguation**: Applies heuristic penalties to prevent attributing symptoms of unrelated conditions to the queried disease.
- **Query Rewriter Agent**: Uses conversation history (last 8 turns) to resolve pronouns and context into standalone search queries.
- **Medical-Grade Glassmorphic UI**: Single Page App with Tailwind CSS, **Lenis Smooth Scroll**, interactive citation badges, and prompt starters.
- **CPU-Friendly Architecture**: Lightweight embeddings designed for standard CPU execution paired with ultra-fast Groq cloud inference.

---

## 🏗️ Architecture Pipeline

```text
User Question + History
         │
         ▼
[Node 1: Rewrite Query] ──► Resolves conversational context via Groq
         │
         ▼
[Node 2: Hybrid Retrieval]
   ├── Dense (ChromaDB Cosine) ─┐
   └── Sparse (BM25Okapi)     ──┴──► RRF Fusion (1.5x Dense Boost) + Symptom Penalty
         │
         ▼
[Node 3: Grounded Generate] ──► Strict Evidence Grounding + Citation Insertion (Groq)
         │
         ▼
Clean Markdown Response with Interactive Citation Badges
```

---

## 🚀 Quick Start (Local Setup)

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/PushkarKanjani/medical-rag-assistant.git
cd medical-rag-assistant

# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate on Linux/macOS
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
Create a `.env` file in the project root:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

### 4. Run PDF Ingestion Pipeline
Ensure your source PDF is in `data/pdf/The_Gale_Encyclopedia_of_Medicine_3rd_Edition.pdf`:
```bash
python -m app.ingest
```
*This extracts ~4,000 pages, cleans noisy headers, builds ~1500 char chunks, computes embeddings, populates ChromaDB at `data/indexes/chroma`, and saves BM25 index to `data/indexes/bm25_index.pkl`.*

### 5. Launch Application
```bash
# Start FastAPI backend with Live Web UI
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## ☁️ Deployment Guide (Render.com)

This repository includes a `render.yaml` Blueprint to automate deployment on Render.

### Option A: Deploy via Render Blueprint (Recommended)
1. Push your repository to GitHub.
2. Log in to [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** → **Blueprint**.
4. Connect your `medical-rag-assistant` GitHub repository.
5. Render will automatically read `render.yaml`.
6. Under **Environment Variables**, add:
   - `GROQ_API_KEY`: Your secret Groq API key (`gsk_...`).
7. Click **Apply** to deploy!

### Option B: Deploy as a Standard Web Service
1. On Render, click **New +** → **Web Service**.
2. Connect your GitHub repository.
3. Configure the following settings:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Free` or `Starter`
4. Add Environment Variable:
   - `GROQ_API_KEY`: `gsk_...`
5. Click **Deploy Web Service**.

> **Note on Free-Tier Ephemeral Storage**: Render free-tier containers have ephemeral storage. When deploying without persistent disks, you can trigger ingestion on-demand via the `POST /api/ingest` endpoint once a PDF is accessible.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web Chat UI (static/index.html) |
| `GET` | `/health` | Server and vector index readiness check |
| `POST` | `/api/chat` | Main LangGraph RAG chat inference |
| `POST` | `/api/ingest` | Triggers background PDF ingestion |
| `GET` | `/docs` | Interactive Swagger API documentation |

---

## 🔒 Security & Privacy
- Sensitive files (`.env`, `data/indexes/`, large PDFs, and Jupyter checkpoints) are strictly excluded via `.gitignore`.
- Citations are transparently referenced to verifiable encyclopedia page records.

---

## 📄 License
This project is licensed under the MIT License.
