from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict


class CriticReport(TypedDict, total=False):
    verdict: Literal["accept", "unsafe", "insufficient", "contradicted"]
    rationale: str
    confidence: float


class AuditEntry(TypedDict, total=False):
    node: str
    status: str
    latency_ms: int
    error: str
    details: dict


class GraphState(TypedDict, total=False):
    query: str
    user_id: str
    abha_id: str | None
    intent: Literal[
        "triage",
        "guideline_search",
        "drug_interaction",
        "abdm_flow",
        "general_qna",
        "escalation",
    ]
    risk_flags: list[str]
    language: str
    sub_questions: list[dict]
    candidate_evidence: Annotated[list[dict], operator.add]
    subgraph_facts: list[dict]
    web_results: list[dict]
    candidate_answer: str
    citations: list[dict]
    critic_report: CriticReport
    iteration: int
    max_iterations: int
    confidence_vector: dict
    final_answer: str
    audit_trail: Annotated[list[AuditEntry], operator.add]