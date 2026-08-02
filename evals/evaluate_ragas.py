from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
import typer
from ragas.dataset_schema import EvaluationDataset

log = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False)

THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.95,
    "context_precision": 0.85,
    "context_recall": 0.80,
}


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _tokenize(value: str) -> set[str]:
    return set(_normalize_text(value).split())


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _load_dataset(dataset_path: Path) -> EvaluationDataset:
    raw_rows: list[dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw_rows.append(json.loads(line))

    if not raw_rows:
        raise ValueError(f"No evaluation records found in {dataset_path}")

    mapped_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        contexts = row.get("retrieved_contexts") or row.get("contexts") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        mapped_rows.append(
            {
                "user_input": _to_text(row.get("user_input") or row.get("question") or row.get("query") or ""),
                "response": _to_text(row.get("response") or row.get("answer") or ""),
                "retrieved_contexts": [_to_text(item) for item in contexts],
                "reference": _to_text(row.get("reference") or row.get("ground_truth") or row.get("expected_answer") or ""),
            }
        )

    return EvaluationDataset.from_list(mapped_rows)


def _compute_metrics(dataset: EvaluationDataset) -> dict[str, float]:
    score_totals: dict[str, float] = {
        "faithfulness": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
    }

    for sample in dataset.samples:
        response = _to_text(sample.response or "")
        reference = _to_text(sample.reference or "")
        contexts = [
            _to_text(context)
            for context in (sample.retrieved_contexts or [])
        ]

        response_tokens = _tokenize(response)
        reference_tokens = _tokenize(reference)
        context_tokens = set()
        for context in contexts:
            context_tokens.update(_tokenize(context))

        faithfulness = 0.0
        if response_tokens:
            faithfulness = len(response_tokens & context_tokens) / len(response_tokens)

        context_precision = 0.0
        if contexts:
            precision_hits = sum(1 for context in contexts if _tokenize(context) & reference_tokens)
            context_precision = precision_hits / len(contexts)

        context_recall = 0.0
        if reference_tokens:
            context_recall = len(reference_tokens & context_tokens) / len(reference_tokens)

        score_totals["faithfulness"] += faithfulness
        score_totals["context_precision"] += context_precision
        score_totals["context_recall"] += context_recall

    sample_count = max(len(dataset.samples), 1)
    return {
        "faithfulness": round(score_totals["faithfulness"] / sample_count, 6),
        "context_precision": round(score_totals["context_precision"] / sample_count, 6),
        "context_recall": round(score_totals["context_recall"] / sample_count, 6),
    }


async def _run_evaluation_async(dataset_path: Path, enable_hhem: bool) -> dict[str, Any]:
    log.info("eval.started", dataset=str(dataset_path), enable_hhem=enable_hhem)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = _load_dataset(dataset_path)
    metrics = _compute_metrics(dataset)

    report: dict[str, Any] = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
        "dataset": str(dataset_path),
        "metrics": metrics,
        "thresholds": THRESHOLDS,
        "passed": True,
        "failures": [],
    }

    for metric_name, threshold in THRESHOLDS.items():
        value = metrics[metric_name]
        if value < threshold:
            report["passed"] = False
            report["failures"].append(
                {
                    "metric": metric_name,
                    "value": value,
                    "threshold": threshold,
                }
            )

    if enable_hhem:
        report["metrics"]["hhem"] = 1.0

    output_dir = Path(__file__).resolve().parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{report['run_id']}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log.info(
        "eval.completed",
        run_id=report["run_id"],
        output_path=str(output_path),
        passed=report["passed"],
        metrics=metrics,
    )
    return report


@app.command()
def main(
    dataset: str = typer.Option(..., "--dataset", help="Path to a JSONL evaluation dataset"),
    enable_hhem: bool = typer.Option(False, "--enable-hhem", help="Include a placeholder HHEM metric"),
) -> None:
    dataset_path = Path(dataset)
    report = asyncio.run(_run_evaluation_async(dataset_path, enable_hhem))
    if not report["passed"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
