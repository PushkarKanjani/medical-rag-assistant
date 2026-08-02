from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from src.retrieval.fusion import Evidence


class Conflict(BaseModel):
    condition_icd10: str
    drug_rxnorm: str
    severity: str
    reason: str


class ContraCheckResult(BaseModel):
    ok: bool
    conflicts: list[Conflict]


_CONTRA_PATH = Path(__file__).resolve().parents[2] / "configs" / "safety" / "contraindications.yaml"
_RXNORM_LOOKUP = {
    "warfarin": "warfarin",
    "aspirin": "aspirin",
    "metformin": "metformin",
    "ibuprofen": "ibuprofen",
    "heparin": "heparin",
    "clopidogrel": "clopidogrel",
}


def _load_contraindications() -> list[dict]:
    if not _CONTRA_PATH.exists():
        return []
    loaded = yaml.safe_load(_CONTRA_PATH.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, list) else []


def _normalize_rxnorm(text: str) -> str | None:
    lowered = text.lower()
    for name, rxnorm in _RXNORM_LOOKUP.items():
        if name in lowered:
            return rxnorm
    return None


async def check_contraindications(query: str, evidence: list[Evidence]) -> ContraCheckResult:
    rules = _load_contraindications()
    query_rxnorm = _normalize_rxnorm(query)
    evidence_text = " ".join(item.text for item in evidence)
    evidence_rxnorm = _normalize_rxnorm(evidence_text)

    conflicts: list[Conflict] = []

    for rule in rules:
        if not isinstance(rule, dict):
            continue

        condition_icd10 = str(rule.get("condition_icd10", "")).strip()
        drug_rxnorm = str(rule.get("drug_rxnorm", "")).strip()
        severity = str(rule.get("severity", "relative")).strip().lower()

        if not condition_icd10 or not drug_rxnorm:
            continue

        query_matches = condition_icd10.lower() in query.lower() or drug_rxnorm.lower() == (query_rxnorm or "")
        evidence_matches = condition_icd10.lower() in evidence_text.lower() or drug_rxnorm.lower() == (evidence_rxnorm or "")

        if query_matches or evidence_matches:
            conflict = Conflict(
                condition_icd10=condition_icd10,
                drug_rxnorm=drug_rxnorm,
                severity="absolute" if severity == "absolute" else "relative",
                reason="contraindication_match",
            )
            conflicts.append(conflict)

    return ContraCheckResult(ok=not any(conflict.severity == "absolute" for conflict in conflicts), conflicts=conflicts)