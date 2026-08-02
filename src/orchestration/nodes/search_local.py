from __future__ import annotations

import time

from src.orchestration.state import GraphState
from src.pipeline.bm25_retriever import bm25_search


async def search_local_node(state: GraphState) -> dict:
    """Retrieve relevant PDF pages via BM25 and return as candidate evidence."""
    start = time.perf_counter()
    query = state.get("query", "")

    evidence: list[dict] = []
    retrieval_mode = "fallback"
    error: str | None = None

    try:
        hits = bm25_search(query, top_k=5)
        if hits:
            evidence = hits
            retrieval_mode = "bm25"
        else:
            evidence = _fallback_evidence(query)
            retrieval_mode = "fallback_no_hits"
    except Exception as exc:
        error = str(exc)
        evidence = _fallback_evidence(query)
        retrieval_mode = "fallback_error"

    latency_ms = int((time.perf_counter() - start) * 1000)

    subgraph_facts = [
        {"source": ev.get("channel", "local"), "fact": ev.get("text", "")}
        for ev in evidence
    ]

    return {
        "candidate_evidence": evidence,
        "subgraph_facts": subgraph_facts,
        "iteration": state.get("iteration", 0) + 1,
        "audit_trail": [
            {
                "node": "search_local",
                "status": "completed" if not error else "degraded",
                "latency_ms": latency_ms,
                "details": {
                    "results": len(evidence),
                    "retrieval_mode": retrieval_mode,
                    **({} if not error else {"error": error}),
                },
            }
        ],
    }


def _fallback_evidence(query: str) -> list[dict]:
    """Deterministic keyword fallback when BM25 corpus is unavailable."""
    q = query.lower()
    if "fever" in q or "rash" in q:
        src, title, auth = "guideline://cdc-fever-rash", "CDC Guidelines on Acute Febrile Exanthems", "guideline"
    elif "amoxicillin" in q or "dose" in q:
        src, title, auth = "guideline://aap-dosing", "AAP Pediatric Antimicrobial Dosing Reference", "guideline"
    elif "warfarin" in q:
        src, title, auth = "label://fda-warfarin", "FDA Prescribing Information for Warfarin", "label"
    elif "hypertension" in q or "blood pressure" in q:
        src, title, auth = "guideline://acc-aha-htn", "ACC/AHA Hypertension Guidelines", "guideline"
    else:
        src, title, auth = "textbook://harrisons", f"Clinical Reference: {query}", "textbook"

    return [{
        "chunk_id": "fallback-1",
        "text": title,
        "full_text": title,
        "page_number": 1,
        "bbox": (0.1, 0.1, 0.9, 0.9),
        "source_uri": src,
        "score": 0.50,
        "channel": "local",
        "authority_level": auth,
    }]