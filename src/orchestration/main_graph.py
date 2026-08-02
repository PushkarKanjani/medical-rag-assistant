from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.orchestration.nodes import (
    critic_node,
    escalate_node,
    output_node,
    plan_node,
    search_local_node,
    search_web_node,
    synthesise_node,
    triage_node,
)
from src.orchestration.state import GraphState


def _route_from_plan(_: GraphState) -> str:
    return "search_local"


def _route_from_critic(state: GraphState) -> str:
    report = state.get("critic_report", {})
    verdict = report.get("verdict", "insufficient")

    if verdict == "accept":
        return "output"
    if verdict == "unsafe":
        return "escalate"
    if verdict in {"insufficient", "contradicted"}:
        if state.get("iteration", 0) < state.get("max_iterations", 3):
            return "search_local"
        return "escalate"
    return "escalate"


builder = StateGraph(GraphState)
builder.add_node("triage", triage_node)
builder.add_node("plan", plan_node)
builder.add_node("search_local", search_local_node)
builder.add_node("search_web", search_web_node)
builder.add_node("synthesise", synthesise_node)
builder.add_node("critic", critic_node)
builder.add_node("escalate", escalate_node)
builder.add_node("output", output_node)

builder.add_edge(START, "triage")
builder.add_edge("triage", "plan")
builder.add_conditional_edges("plan", _route_from_plan, {"search_local": "search_local"})
builder.add_edge("search_local", "search_web")
builder.add_edge("search_web", "synthesise")
builder.add_edge("synthesise", "critic")
builder.add_conditional_edges(
    "critic",
    _route_from_critic,
    {
        "output": "output",
        "escalate": "escalate",
        "search_local": "search_local",
    },
)
builder.add_edge("output", END)
builder.add_edge("escalate", END)

compiled = builder.compile()


async def run_query(
    query: str,
    user_id: str,
    abha_id: str | None = None,
    max_iterations: int = 3,
) -> dict:
    initial_state: GraphState = {
        "query": query,
        "user_id": user_id,
        "abha_id": abha_id,
        "max_iterations": max_iterations,
        "iteration": 0,
    }
    return await compiled.ainvoke(initial_state)