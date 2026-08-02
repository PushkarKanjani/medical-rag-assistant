from __future__ import annotations

import pytest

from src.safety.contraindication import (
    ContraCheckResult,
    Conflict,
    _normalize_rxnorm,
    check_contraindications,
)
from src.safety.dosage_guard import (
    DosageCheckResult,
    DosageViolationError,
    check_dosage_ranges,
)
from src.retrieval.fusion import Evidence


class TestNormalizeRxnorm:
    def test_match_warfarin(self) -> None:
        assert _normalize_rxnorm("prescribe warfarin") == "warfarin"

    def test_match_aspirin_mixed_case(self) -> None:
        assert _normalize_rxnorm("ASPIRIN 100mg") == "aspirin"

    def test_no_match(self) -> None:
        assert _normalize_rxnorm("lisinopril 20mg") is None

    def test_match_metformin(self) -> None:
        assert _normalize_rxnorm("metformin extended release") == "metformin"


@pytest.mark.asyncio
class TestCheckContraindications:
    async def test_no_conflicts_without_configs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        evidence = [Evidence(
            chunk_id="1",
            text="No content",
            page_number=1,
            bbox=(0, 0, 1, 1),
            source_uri="test://",
            score=0.9,
            channel="local",
        )]
        result = await check_contraindications("warfarin", evidence)
        assert isinstance(result, ContraCheckResult)
        assert result.ok is True
        assert isinstance(result.conflicts, list)

    async def test_empty_evidence(self) -> None:
        result = await check_contraindications("aspirin", [])
        assert result.ok is True
        assert result.conflicts == []


class TestCheckDosageRanges:
    def test_no_patterns_in_text(self) -> None:
        result = check_dosage_ranges("No dosage information present.")
        assert isinstance(result, DosageCheckResult)
        assert result.ok is True
        assert result.violations == []

    def test_pattern_matches_without_limits_no_violations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.safety.dosage_guard._load_dosage_limits", lambda: {})
        result = check_dosage_ranges("Give Amoxicillin 500 mg orally")
        assert result.ok is True
        assert len(result.violations) == 0

    def test_dose_under_max_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "src.safety.dosage_guard._load_dosage_limits",
            lambda: {"amoxicillin": {"max_mg_per_day": 4000}},
        )
        result = check_dosage_ranges("Amoxicillin 500 mg twice daily")
        assert result.ok is True

    def test_dose_exceeds_max_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "src.safety.dosage_guard._load_dosage_limits",
            lambda: {"amoxicillin": {"max_mg_per_day": 4000}},
        )
        with pytest.raises(DosageViolationError):
            check_dosage_ranges("Amoxicillin 5000 mg overdose")

    def test_result_schema_contract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.safety.dosage_guard._load_dosage_limits", lambda: {})
        result = check_dosage_ranges("plain text")
        assert result.model_dump() == {"ok": True, "violations": []}
