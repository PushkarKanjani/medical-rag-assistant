"""
app/agent.py
────────────
Medical RAG Agent with Query Rewriting, Hybrid Dense+BM25 Retrieval,
Reciprocal Rank Fusion (RRF), Symptom-Disease Penalty, Evidence Grounding,
and LangGraph Orchestration using Groq (llama-3.1-8b-instant).

Ported from Cell 3 & Cell 4 of Medical_RAG_Core_Engine.ipynb.
Includes lazy initialization for fast server startup on ephemeral/cloud hosts.

Run via:
    python -m app.agent
"""

from __future__ import annotations

import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import chromadb
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, StateGraph
import torch

# ---- Load Environment & Paths ----
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

INDEXES_DIR = BASE_DIR / "data" / "indexes"
CHROMA_PATH = INDEXES_DIR / "chroma"
BM25_PATH = INDEXES_DIR / "bm25_index.pkl"

COLLECTION_NAME = "medical_encyclopedia"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GROQ_MODEL = "llama-3.1-8b-instant"

# ---- Lazy Client & Resource Cache ----
_groq_client: Groq | None = None
_chroma_collection = None
_embed_model = None
_bm25 = None
_chunk_records = None
_rag_app = None


def reset_resource_cache():
    """Resets the cached Chroma and BM25 objects so newly ingested indexes are loaded."""
    global _chroma_collection, _bm25, _chunk_records
    _chroma_collection = None
    _bm25 = None
    _chunk_records = None


def get_groq_client() -> Groq | None:
    """Returns or lazily creates the Groq API client from environment variables."""
    global _groq_client
    if _groq_client is not None:
        return _groq_client

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        return None

    _groq_client = Groq(api_key=api_key)
    return _groq_client


def get_resources():
    """Lazily initializes and returns ChromaDB collection, embedding model, BM25 index, and chunk records."""
    global _chroma_collection, _embed_model, _bm25, _chunk_records

    if _chroma_collection is None:
        if not CHROMA_PATH.exists():
            raise FileNotFoundError(
                f"Chroma index not found at {CHROMA_PATH}. Run 'python -m app.ingest' or trigger /api/ingest first."
            )
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _chroma_collection = chroma_client.get_collection(COLLECTION_NAME)

    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

    if _bm25 is None or _chunk_records is None:
        if not BM25_PATH.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {BM25_PATH}. Run 'python -m app.ingest' or trigger /api/ingest first."
            )
        with open(BM25_PATH, "rb") as f:
            _bm25_data = pickle.load(f)
        _bm25 = _bm25_data["bm25"]
        _chunk_records = _bm25_data["chunks"]

    return _chroma_collection, _embed_model, _bm25, _chunk_records


# ---- Agent State ----
class AgentState(TypedDict):
    question: str
    history: List[Dict[str, str]]
    search_query: str
    evidence: List[Dict[str, Any]]
    answer: str


# ---- Node 1: Query Rewriting ----
def rewrite_node(state: AgentState) -> AgentState:
    """Rewrites follow-up questions into self-contained standalone search queries."""
    question = state["question"]
    history = state.get("history", [])[-6:]

    if not history:
        state["search_query"] = question
        return state

    history_str = "\n".join(f"{h['role']}: {h['content']}" for h in history)

    system_prompt = (
        "You rewrite follow-up medical questions into fully self-contained search queries. "
        "Use the conversation history to resolve pronouns and implicit references "
        "(e.g. 'it', 'that condition'). Output ONLY the rewritten query, nothing else. "
        "If the question is already self-contained, return it unchanged."
    )
    user_prompt = (
        f"Conversation history:\n{history_str}\n\n"
        f"Follow-up question: {question}\n\n"
        "Rewritten standalone query:"
    )

    client = get_groq_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=100,
            )
            rewritten = resp.choices[0].message.content.strip().strip('"')
            state["search_query"] = rewritten if rewritten else question
        except Exception as e:
            print(f"[rewrite_node] Groq call failed, falling back to raw question: {e}")
            state["search_query"] = question
    else:
        state["search_query"] = question

    print(f"[rewrite_node] '{question}' -> '{state['search_query']}'")
    return state


# ---- Node 2: Hybrid Retrieval (Dense + BM25, RRF, Symptom Penalty) ----
RRF_K = 60
DENSE_BOOST = 1.5  # Boost dense to prioritize semantic meaning over keywords
TOP_N_PER_METHOD = 20
TOP_K_FINAL = 5


def _extract_disease_phrase(query: str) -> str:
    """Heuristic: strip symptom-related stopwords to isolate the likely disease term."""
    stop = r'\b(symptom|symptoms|of|for|the|what|are|is|a|an|does|do|show|cause|causes|treatment|treat|cure|how|to)\b'
    disease = re.sub(stop, '', query, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', disease).strip()


def _symptom_penalty_factor(query: str, text: str) -> float:
    """
    If query is about symptoms, check whether a DIFFERENT disease-like phrase
    (Title Case, multi-word) appears in the chunk BEFORE the queried disease term.
    If so, this chunk is likely describing the wrong condition's symptoms -> penalize.
    """
    if "symptom" not in query.lower():
        return 1.0

    disease = _extract_disease_phrase(query)
    if not disease:
        return 1.0

    disease_pos = text.lower().find(disease.lower())
    if disease_pos == -1:
        return 1.0

    other_disease_matches = re.finditer(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})\b', text[:disease_pos])
    for m in other_disease_matches:
        candidate = m.group(1).lower()
        # Check if candidate is actually the target disease
        if candidate not in disease.lower() and disease.lower() not in candidate:
            return 0.4
    return 1.0


def retrieve_node(state: AgentState) -> AgentState:
    """Performs hybrid retrieval using Chroma dense vectors and BM25 sparse scores with RRF fusion."""
    collection, embed_model, bm25, chunk_records = get_resources()
    query = state["search_query"]

    # --- Dense Retrieval (Chroma) ---
    query_embedding = embed_model.encode([query]).tolist()
    dense_results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_N_PER_METHOD,
        include=["documents", "metadatas", "distances"],
    )
    dense_ids = dense_results["ids"][0]
    dense_docs = dense_results["documents"][0]
    dense_meta = dense_results["metadatas"][0]
    dense_rank_map = {cid: rank for rank, cid in enumerate(dense_ids)}
    dense_lookup = {
        cid: {"text": doc, "page": meta["page"]}
        for cid, doc, meta in zip(dense_ids, dense_docs, dense_meta)
    }

    # --- Sparse Retrieval (BM25) ---
    tokenized_query = re.findall(r'\w+', query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)

    scored_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:TOP_N_PER_METHOD]

    sparse_ids = [chunk_records[i]["id"] for i in scored_indices]
    sparse_rank_map = {cid: rank for rank, cid in enumerate(sparse_ids)}

    sparse_lookup = {}
    for i in scored_indices:
        c = chunk_records[i]
        sparse_lookup[c["id"]] = {"text": c["text"], "page": c["page"]}

    # --- Reciprocal Rank Fusion (Dense Boosted 1.5x) ---
    all_ids = set(dense_ids) | set(sparse_ids)
    fused_scores = {}
    for cid in all_ids:
        score = 0.0
        if cid in dense_rank_map:
            score += DENSE_BOOST * (1.0 / (RRF_K + dense_rank_map[cid] + 1))
        if cid in sparse_rank_map:
            score += 1.0 / (RRF_K + sparse_rank_map[cid] + 1)
        fused_scores[cid] = score

    # --- Apply Symptom-Disease Penalty ---
    final_scores = {}
    text_lookup = {}
    for cid, score in fused_scores.items():
        info = dense_lookup.get(cid) or sparse_lookup.get(cid)
        text_lookup[cid] = info
        penalty = _symptom_penalty_factor(query, info["text"])
        final_scores[cid] = score * penalty

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_FINAL]

    evidence = []
    print(f"\n[retrieve_node] Query: '{query}' — top {len(ranked)} results:")
    for cid, score in ranked:
        info = text_lookup[cid]
        snippet = info["text"][:120].replace("\n", " ")
        print(f"  Page {info['page']:>4} | score={score:.5f} | {snippet}...")
        evidence.append({"id": cid, "text": info["text"], "page": info["page"], "score": score})

    state["evidence"] = evidence
    return state


# ---- Node 3: Grounded Answer Generation ----
SYSTEM_PROMPT = """You are a careful, evidence-grounded medical information assistant.

STRICT RULES:
1. Answer ONLY using the provided evidence. Never use outside knowledge to fill gaps.
2. Carefully distinguish the MAIN condition being asked about from OTHER related conditions
   mentioned in the evidence. Do NOT attribute symptoms or facts belonging to a different
   condition (e.g. Kidney Failure) to the queried condition (e.g. Hypertension).
3. If the evidence contains phrases like "silent killer" or states the condition has
   "no symptoms" / "often asymptomatic", state this prominently and early in your answer.
4. Cite every factual claim with the source chunk id in square brackets, e.g. [gale-p42-c3].
   Do not make claims without a citation.
5. If the evidence does not contain enough information to answer, say so explicitly rather
   than guessing.

Respond in a natural, conversational tone — you're a knowledgeable, careful assistant,
not a robotic search engine."""


def generate_node(state: AgentState) -> AgentState:
    """Generates a strictly grounded response with inline citations using Groq."""
    question = state["question"]
    evidence = state["evidence"]

    if not evidence:
        state["answer"] = "I could not find any relevant information in the medical encyclopedia for your question."
        return state

    # Truncate evidence to prevent prompt overflow
    evidence_block = "\n\n".join(
        f"[{e['id']}] (page {e['page']}):\n{e['text'][:1000]}..." for e in evidence
    )

    user_prompt = f"Evidence:\n{evidence_block}\n\nQuestion: {question}\n\nAnswer (with citations):"
    full_prompt_for_size = SYSTEM_PROMPT + user_prompt
    print(f"[generate_node] Prompt size: {len(full_prompt_for_size):,} chars (~{len(full_prompt_for_size) // 4:,} tokens est.)")

    client = get_groq_client()
    if not client:
        state["answer"] = (
            "Error: GROQ_API_KEY is not configured on this server. Please set your GROQ_API_KEY in environment variables to generate answers."
        )
        return state

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        state["answer"] = resp.choices[0].message.content.strip()
    except Exception as e:
        error_msg = f"Sorry, I hit an error generating the answer: {e}"
        print(f"\n❌ GENERATE ERROR: {error_msg}")
        state["answer"] = error_msg

    return state


# ---- Build & Compile LangGraph ----
def create_rag_graph():
    """Builds and compiles the StateGraph workflow."""
    workflow = StateGraph(AgentState)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


def get_rag_app():
    """Lazy loader for the compiled LangGraph workflow."""
    global _rag_app
    if _rag_app is None:
        _rag_app = create_rag_graph()
    return _rag_app


# Backward compatibility alias
rag_app = get_rag_app()


# ---- Interactive Chat Loop ----
def run_chat():
    """Interactive terminal chat loop maintaining last 8 turns of conversation history."""
    chat_history: List[Dict[str, str]] = []
    print("=" * 65)
    print("🩺 Medical RAG Assistant — Gale Encyclopedia of Medicine")
    print("Type your medical question below. Type 'exit' or Ctrl+C to quit.")
    print("=" * 65)

    app_instance = get_rag_app()
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break

            initial_state: AgentState = {
                "question": user_input,
                "history": chat_history[-8:],
                "search_query": "",
                "evidence": [],
                "answer": "",
            }

            result = app_instance.invoke(initial_state)
            answer = result.get("answer", "No answer generated.")

            if not answer:
                answer = "The system failed to generate a response. Please try again."

            print("\n" + "-" * 65)
            print(f"Assistant:\n{answer}")
            print("-" * 65)

            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": answer})
            chat_history = chat_history[-8:]

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Error] {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    run_chat()
