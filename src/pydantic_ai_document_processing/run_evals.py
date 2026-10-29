"""Evaluate golden classification and extraction datasets and print reports."""

from __future__ import annotations
from pydantic_ai import InstrumentationSettings
import json
from datetime import datetime

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel
from pydantic_evals import Dataset
from pydantic_evals.reporting import EvaluationReport

from pydantic_ai_document_processing.evals import (
    CASES_DIR,
    ClassificationInput,
    ExtractionInput,
    load_classification_dataset,
    load_extraction_dataset,
    resolve_pdf_path,
)
import logfire
from pydantic_ai_document_processing.schemas import DocumentClassification, DocumentType

Kind = Literal["all", "classification", "extraction"]

logfire.configure()
logfire.instrument_pydantic_ai(
    include_binary_content = False,
)

async def classify_case(inputs: ClassificationInput) -> DocumentClassification:
    from pydantic_ai_document_processing.runtime import classification_pipe

    return await classification_pipe(resolve_pdf_path(inputs.pdf_path))


async def classify_case_no_rag(inputs: ClassificationInput) -> DocumentClassification:
    from pydantic_ai_document_processing.runtime import classification_pipe

    return await classification_pipe(resolve_pdf_path(inputs.pdf_path), use_rag=False)

async def extract_case(inputs: ExtractionInput) -> BaseModel:
    from pydantic_ai_document_processing.agent import extraction_agent_schema
    from pydantic_ai_document_processing.runtime import extraction_pipe

    schema = extraction_agent_schema(inputs.doc_type)
    return await extraction_pipe(
        resolve_pdf_path(inputs.pdf_path),
        inputs.doc_type,
        schema,
    )


async def run_evaluations(
    *,
    cases_dir: Path = CASES_DIR,
    kind: Kind = "all",
    categories: Sequence[DocumentType] | None = None,
    limit: int | None = None,
    max_concurrency: int = 4,
    include_reasons: bool = False,
) -> list[EvaluationReport[Any, Any, Any]]:
    reports: list[EvaluationReport[Any, Any, Any]] = []
    selected = list(categories) if categories else list(get_args(DocumentType))

    if kind in ("all", "classification"):
        path = cases_dir / "classification.yaml"
        dataset = _maybe_limit(load_classification_dataset(path), limit)

        rag_report = await dataset.evaluate(
            classify_case,
            name=dataset.name,
            metadata={"use_rag": True},
            max_concurrency=max_concurrency,
        )
        print_evaluation_report(rag_report, include_reasons=include_reasons)
        reports.append(rag_report)

        # classification without RAG
        no_rag_report = await dataset.evaluate(
            classify_case_no_rag,
            name=dataset.name,
            metadata={"use_rag": False},
            max_concurrency=max_concurrency,
        )
        print_evaluation_report(no_rag_report, include_reasons=include_reasons)
        reports.append(no_rag_report)

    if kind in ("all", "extraction"):
        from pydantic_ai_document_processing.agent import extraction_agent_schema

        for category in selected:
            path = cases_dir / f"extraction-{category}.yaml"
            schema = extraction_agent_schema(category)
            dataset = _maybe_limit(load_extraction_dataset(path, schema), limit)
            report = await dataset.evaluate(
                extract_case,
                name=dataset.name,
                max_concurrency=max_concurrency,
            )
            print_evaluation_report(report, include_reasons=include_reasons)
            reports.append(report)

    return reports


def print_evaluation_report(
    report: EvaluationReport[Any, Any, Any],
    *,
    include_reasons: bool = False,
) -> None:
    print()
    print("=" * 72)
    print(f"Dataset: {report.dataset_name or report.name}")
    print(f"Cases: {len(report.cases)}  Failures: {len(report.failures)}")
    averages = report.averages()
    if averages is not None:
        if averages.assertions is not None:
            print(f"Assertion pass rate: {averages.assertions:.1%}")
        for name, value in averages.scores.items():
            print(f"{name}: {value}")
    if report.failures:
        for failure in report.failures:
            print(f"  failed {failure.name}: {failure.error_message}")
    print("=" * 72)
    report.print(include_averages=True, include_analyses=True, include_reasons=include_reasons)


def _maybe_limit[InputsT, OutputT, MetadataT](
    dataset: Dataset[InputsT, OutputT, MetadataT],
    limit: int | None,
) -> Dataset[InputsT, OutputT, MetadataT]:
    if limit is None:
        return dataset
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    return dataset.model_copy(update={"cases": dataset.cases[:limit]})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pydantic_evals reports for golden classification and extraction datasets.",
    )
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=CASES_DIR,
        help="Directory containing classification.yaml and extraction-*.yaml",
    )
    parser.add_argument(
        "--only",
        choices=("all", "classification", "extraction"),
        default="all",
    )
    parser.add_argument(
        "--category",
        action="append",
        choices=get_args(DocumentType),
        help="Extraction document type to evaluate; repeatable. Default: all types.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max cases per dataset")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument(
        "--reasons",
        action="store_true",
        help="Include evaluator reasons (extraction field mismatches) in the report",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    import asyncio

    args = build_arg_parser().parse_args(argv)
    reports = asyncio.run(
        run_evaluations(
            cases_dir=args.cases_dir,
            kind=args.only,
            categories=args.category,
            limit=args.limit,
            max_concurrency=args.max_concurrency,
            include_reasons=args.reasons,
        )
    )

    # save the reports to a file with the current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"reports_{timestamp}.json", "w") as f:
        json.dump(reports, f, indent=4)

if __name__ == "__main__":
    main()
