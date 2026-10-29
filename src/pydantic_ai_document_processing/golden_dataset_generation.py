"""Build typed pydantic_evals datasets from validation PDFs and JSON sidecars."""

from __future__ import annotations

import json
import random
import types
from pathlib import Path
from typing import Any, Annotated, Union, get_args, get_origin

from pydantic import BaseModel, TypeAdapter
from pydantic_evals import Case

from pydantic_ai_document_processing.evals import (
    CASES_DIR,
    PACKAGE_ROOT,
    ClassificationCase,
    ClassificationDataset,
    ClassificationInput,
    ClassificationMetrics,
    ExtractionInput,
    ExtractionMetrics,
    ExtractionMetricsReport,
    classification_confusion_matrix,
    to_dataset_pdf_path,
    typed_extraction_dataset,
)
from pydantic_ai_document_processing.schemas import DocumentClassification, DocumentType

_DATA_ROOT = PACKAGE_ROOT / "data" / "company-document-dataset-v2" / "validation"

SAMPLE_SIZE = 20
SAMPLE_SEED = 42


def generate_golden_datasets(
    *,
    sample_size: int = SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
    cases_dir: Path = CASES_DIR,
    data_root: Path = _DATA_ROOT,
) -> None:
    from pydantic_ai_document_processing.agent import extraction_agent_schema

    rng = random.Random(seed)
    cases_dir.mkdir(parents=True, exist_ok=True)
    classification_cases: list[ClassificationCase] = []

    for category in get_args(DocumentType):
        pdfs = _sample_pdfs(data_root / category, sample_size, rng)
        classification_cases.extend(_classification_cases(category, pdfs))
        schema = extraction_agent_schema(category)
        extraction_cases = _extraction_cases(category, schema, pdfs)
        _save_extraction_dataset(category, schema, extraction_cases, cases_dir)

    _save_classification_dataset(classification_cases, cases_dir)


def convert_extraction_json_to_pydantic[T: BaseModel](
    extraction_json: dict[str, Any],
    doc_schema: type[T],
) -> T:
    """Map a document JSON pair onto the extraction schema.

    Sidecar files are HuggingFace rows: gold extraction lives in
    ``extracted_data`` (often a JSON string) and includes keys the Pydantic
    models do not declare (e.g. ``parties.bill_to.contact``, ``refs.salesperson``).
    Unknown keys are folded into the nearest ``extra`` field instead of being
    dropped or assumed to match 1:1.
    """
    payload = _unwrap_extracted_data(extraction_json)
    adapted, leftover = _adapt_model(doc_schema, payload)
    if leftover:
        extra = adapted.setdefault("extra", {})
        if not isinstance(extra, dict):
            extra = {"_existing_extra": extra}
            adapted["extra"] = extra
        adapted["extra"] = _deep_merge(leftover, extra)

    return doc_schema.model_validate(adapted)


def _sample_pdfs(category_path: Path, sample_size: int, rng: random.Random) -> list[Path]:
    documents = list(category_path.glob("*.pdf"))
    if len(documents) < sample_size:
        raise ValueError(
            f"Not enough documents in {category_path}: {len(documents)} < {sample_size}"
        )
    return rng.sample(documents, sample_size)


def _classification_cases(
    category: DocumentType,
    pdfs: list[Path],
) -> list[ClassificationCase]:
    return [
        Case(
            name=f"{category}/{pdf_path.stem}",
            inputs=ClassificationInput(pdf_path=to_dataset_pdf_path(pdf_path)),
            expected_output=DocumentClassification(document_type=category),
        )
        for pdf_path in pdfs
    ]


def _extraction_cases[T: BaseModel](
    category: DocumentType,
    schema: type[T],
    pdfs: list[Path],
) -> list[Case[ExtractionInput, T, dict[str, Any]]]:
    cases: list[Case[ExtractionInput, T, dict[str, Any]]] = []
    for pdf_path in pdfs:
        sidecar_path = pdf_path.with_suffix(".json")
        if not sidecar_path.exists():
            raise FileNotFoundError(f"Missing JSON pair for {pdf_path}")
        sidecar = json.loads(sidecar_path.read_text())
        cases.append(
            Case(
                name=f"{category}/{pdf_path.stem}",
                inputs=ExtractionInput(
                    pdf_path=to_dataset_pdf_path(pdf_path),
                    doc_type=category,
                ),
                expected_output=convert_extraction_json_to_pydantic(sidecar, schema),
            )
        )
    return cases


def _save_extraction_dataset[T: BaseModel](
    category: DocumentType,
    schema: type[T],
    cases: list[Case[ExtractionInput, T, dict[str, Any]]],
    cases_dir: Path,
) -> None:
    dataset = typed_extraction_dataset(schema)(
        name=f"extraction-{category}",
        cases=cases,
        evaluators=[ExtractionMetrics()],
        report_evaluators=[ExtractionMetricsReport()],
    )
    dataset.to_file(
        cases_dir / f"extraction-{category}.yaml",
        custom_evaluator_types=[ExtractionMetrics],
        custom_report_evaluator_types=[ExtractionMetricsReport],
    )


def _save_classification_dataset(
    cases: list[ClassificationCase],
    cases_dir: Path,
) -> None:
    ClassificationDataset(
        name="classification",
        cases=cases,
        evaluators=[ClassificationMetrics()],
        report_evaluators=[classification_confusion_matrix()],
    ).to_file(
        cases_dir / "classification.yaml",
        custom_evaluator_types=[ClassificationMetrics],
    )


def _unwrap_extracted_data(extraction_json: dict[str, Any]) -> dict[str, Any]:
    if "extracted_data" not in extraction_json:
        if not isinstance(extraction_json, dict):
            raise TypeError("extraction_json must be a dict")
        return extraction_json

    extracted = extraction_json["extracted_data"]
    if isinstance(extracted, str):
        extracted = json.loads(extracted)
    if not isinstance(extracted, dict):
        raise TypeError("extracted_data must decode to an object")
    return extracted


def _concrete_args(annotation: Any) -> tuple[Any, ...]:
    origin = get_origin(annotation)
    if origin is Annotated:
        return _concrete_args(get_args(annotation)[0])
    if origin is Union or origin is types.UnionType:
        return tuple(arg for arg in get_args(annotation) if arg is not type(None))
    return (annotation,)


def _is_pydantic_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _adapt_model(
    model_cls: type[BaseModel],
    data: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(data, dict):
        validated = TypeAdapter(model_cls).validate_python(data)
        return validated.model_dump(exclude_unset=True), {}

    fields = model_cls.model_fields
    adapted: dict[str, Any] = {}
    unknown: dict[str, Any] = {}
    nested_unmapped: dict[str, Any] = {}

    for key, value in data.items():
        if key == "extra":
            continue
        if key not in fields:
            unknown[key] = value
            continue
        coerced, leftover = _adapt_value(fields[key].annotation, value)
        adapted[key] = coerced
        if leftover:
            nested_unmapped[key] = leftover

    unmapped = _deep_merge(unknown, nested_unmapped)
    source_extra = data.get("extra")
    if isinstance(source_extra, dict):
        unmapped = _deep_merge(unmapped, source_extra)
    elif source_extra is not None:
        unmapped = _deep_merge(unmapped, {"_source_extra": source_extra})

    if "extra" in fields:
        if unmapped:
            adapted["extra"] = unmapped
        return adapted, {}
    return adapted, unmapped


def _adapt_value(annotation: Any, value: Any) -> tuple[Any, dict[str, Any]]:
    if value is None:
        return None, {}

    for arg in _concrete_args(annotation):
        origin = get_origin(arg)
        if origin is list:
            inner = get_args(arg)[0] if get_args(arg) else Any
            if not isinstance(value, list):
                continue
            items: list[Any] = []
            leftover: dict[str, Any] = {}
            for index, item in enumerate(value):
                coerced, item_left = _adapt_value(inner, item)
                items.append(coerced)
                if item_left:
                    leftover[str(index)] = item_left
            return items, leftover
        if origin is dict:
            return value, {}
        if _is_pydantic_model(arg) and isinstance(value, dict):
            return _adapt_model(arg, value)

    return TypeAdapter(annotation).validate_python(value), {}


if __name__ == "__main__":
    generate_golden_datasets()
