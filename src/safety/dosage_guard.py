from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel


class Violation(BaseModel):
    drug: str
    dose_mg: float
    limit_mg_per_day: float | None = None
    reason: str


class DosageCheckResult(BaseModel):
    ok: bool
    violations: list[Violation]


class DosageViolationError(ValueError):
    pass


_DOSAGE_PATH = Path(__file__).resolve().parents[2] / "configs" / "safety" / "dosage_maximums.yaml"
_DOSAGE_PATTERN = re.compile(r"\b([A-Za-z][A-Za-z0-9_\-]*)\s+(\d+(?:\.\d+)?)\s*mg\b", re.IGNORECASE)


def _load_dosage_limits() -> dict[str, dict]:
    if not _DOSAGE_PATH.exists():
        return {}
    loaded = yaml.safe_load(_DOSAGE_PATH.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def check_dosage_ranges(answer: str) -> DosageCheckResult:
    limits = _load_dosage_limits()
    violations: list[Violation] = []

    for drug, dose_text in _DOSAGE_PATTERN.findall(answer):
        normalized_drug = drug.lower()
        dose_mg = float(dose_text)
        limit_entry = limits.get(normalized_drug) or limits.get(drug) or {}
        max_mg_per_day = limit_entry.get("max_mg_per_day")

        if max_mg_per_day is not None and dose_mg > float(max_mg_per_day):
            violation = Violation(
                drug=drug,
                dose_mg=dose_mg,
                limit_mg_per_day=float(max_mg_per_day),
                reason="dose_exceeds_maximum",
            )
            violations.append(violation)
            raise DosageViolationError(f"{drug} dose {dose_mg} mg exceeds maximum {max_mg_per_day} mg/day")

        min_age = limit_entry.get("min_age")
        if min_age is not None:
            continue

    return DosageCheckResult(ok=not violations, violations=violations)