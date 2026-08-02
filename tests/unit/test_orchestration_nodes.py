from __future__ import annotations

import pytest

from src.orchestration.nodes.triage import triage_node
from src.orchestration.nodes.plan import plan_node
from src.orchestration.nodes.critic import critic_node
from src.orchestration.nodes.output import output_node
from src.orchestration.nodes.escalate import escalate_node
from src.orchestration.nodes.search_web import search_web_node
from src.orchestration.nodes.search_local import _fallback_evidence
from src.orchestration.state import GraphState


async_mark = pytest.mark.asyncio


@async_mark
class TestTriageNode:
    async def test_triage_abha_intent(self) -> None:
        state: GraphState = {"query": "create my ABHA health id"}
        result = await triage_node(state)
        assert result["intent"] == "abdm_flow"
        assert "audit_trail" in result

    async def test_triage_drug_interaction(self) -> None:
        state: GraphState = {"query": "aspirin dose 500 mg"}
        result = await triage_node(state)
        assert result["intent"] == "drug_interaction"

    async def test_triage_guideline(self) -> None:
        state: GraphState = {"query": "protocol for diabetes management recommendation"}
        result = await triage_node(state)
        assert result["intent"] == "guideline_search"

    async def test_triage_general(self) -> None:
        state: GraphState = {"query": "what causes headaches"}
        result = await triage_node(state)
        assert result["intent"] == "general_qna"

    async def test_triage_empty_query_escalation(self) -> None:
        state: GraphState = {"query": "  "}
        result = await triage_node(state)
        assert result["intent"] == "escalation"


@async_mark
class TestPlanNode:
    async def test_plan_simple_query(self) -> None:
        state: GraphState = {"query": "what is hypertension"}
        result = await plan_node(state)
        assert len(result["sub_questions"]) >= 1
        assert result["max_iterations"] == 3

    async def test_plan_drug_interaction_adds_subquestion(self) -> None:
        state: GraphState = {"query": "warfarin dosage", "intent": "drug_interaction"}
        result = await plan_node(state)
        assert len(result["sub_questions"]) >= 2

    async def test_plan_respects_custom_max_iterations(self) -> None:
        state: GraphState = {"query": "test", "max_iterations": 1}
        result = await plan_node(state)
        assert result["max_iterations"] == 1


@async_mark
class TestCriticNode:
    async def test_critic_accepts_non_empty_answer(self) -> None:
        state: GraphState = {"candidate_answer": "Some clinical answer text"}
        result = await critic_node(state)
        assert result["critic_report"]["verdict"] == "accept"

    async def test_critic_insufficient_without_answer(self) -> None:
        state: GraphState = {"candidate_answer": ""}
        result = await critic_node(state)
        assert result["critic_report"]["verdict"] == "insufficient"

    async def test_critic_includes_audit_trail(self) -> None:
        state: GraphState = {"candidate_answer": "answer"}
        result = await critic_node(state)
        assert result["audit_trail"][0]["node"] == "critic"


@async_mark
class TestOutputNode:
    async def test_output_uses_candidate_answer(self) -> None:
        state: GraphState = {
            "candidate_answer": "### Final clinical guidance",
            "final_answer": "other",
        }
        result = await output_node(state)
        assert result["final_answer"] == "### Final clinical guidance"

    async def test_output_falls_back_to_final_answer(self) -> None:
        state: GraphState = {"final_answer": "Escalated"}
        result = await output_node(state)
        assert result["final_answer"] == "Escalated"

    async def test_output_placeholder_when_no_answer(self) -> None:
        state: GraphState = {}
        result = await output_node(state)
        assert "No answer generated." in result["final_answer"]


@async_mark
class TestEscalateNode:
    async def test_escalate_sets_message(self) -> None:
        state: GraphState = {"critic_report": {"verdict": "unsafe"}}
        result = await escalate_node(state)
        assert "human review" in result["final_answer"]
        audit = result["audit_trail"][0]
        assert audit["node"] == "escalate"
        assert audit["details"]["reason"] == "unsafe"


@async_mark
class TestSearchWebNode:
    async def test_search_web_returns_results(self) -> None:
        state: GraphState = {"query": "diabetes guidelines 2024"}
        result = await search_web_node(state)
        assert len(result["web_results"]) >= 1
        assert "url" in result["web_results"][0]


class TestFallbackEvidence:
    def test_fallback_fever_rash(self) -> None:
        ev = _fallback_evidence("child fever rash")
        assert ev[0]["authority_level"] == "guideline"
        assert "cdc" in ev[0]["source_uri"]

    def test_fallback_amoxicillin(self) -> None:
        ev = _fallback_evidence("amoxicillin dose")
        assert "aap" in ev[0]["source_uri"]

    def test_fallback_warfarin(self) -> None:
        ev = _fallback_evidence("warfarin interactions")
        assert ev[0]["authority_level"] == "label"

    def test_fallback_hypertension(self) -> None:
        ev = _fallback_evidence("high blood pressure bp")
        assert "acc-aha" in ev[0]["source_uri"]

    def test_fallback_default(self) -> None:
        ev = _fallback_evidence("migraine causes")
        assert "harrisons" in ev[0]["source_uri"]

    def test_fallback_structure(self) -> None:
        ev = _fallback_evidence("test")
        item = ev[0]
        for key in ("chunk_id", "text", "page_number", "bbox", "source_uri", "score", "channel"):
            assert key in item
