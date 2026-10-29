from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic_evals import Case
from pydantic_evals.dataset import Dataset

from pydantic_ai_document_processing.evals import (
    ClassificationDataset,
    ClassificationInput,
    ClassificationMetrics,
    ExtractionInput,
    ExtractionMetrics,
    ExtractionMetricsReport,
    classification_confusion_matrix,
    load_extraction_dataset,
    typed_extraction_dataset,
)
from pydantic_ai_document_processing.golden_dataset_generation import (
    convert_extraction_json_to_pydantic,
)
from pydantic_ai_document_processing.schemas import DocumentClassification, Invoice, PackingSlip


def test_convert_folds_unknown_keys_into_invoice_extra() -> None:
    payload = {
        "extracted_data": json.dumps(
            {
                "doc_type": "invoice",
                "number": "INV-1",
                "parties": {"bill_to": {"name": "Acme", "contact": "Jane"}},
            }
        )
    }

    invoice = convert_extraction_json_to_pydantic(payload, Invoice)

    assert isinstance(invoice, Invoice)
    assert invoice.number == "INV-1"
    assert invoice.parties is not None
    assert invoice.parties.bill_to is not None
    assert invoice.parties.bill_to.name == "Acme"
    assert invoice.extra is not None
    assert invoice.extra["parties"]["bill_to"]["contact"] == "Jane"


def test_extraction_dataset_schema_is_bound_to_invoice() -> None:
    schema = typed_extraction_dataset(Invoice).model_json_schema_with_evaluators(
        [ExtractionMetrics],
        [ExtractionMetricsReport],
    )
    text = json.dumps(schema)
    assert "Invoice" in text
    assert '"doc_type"' in text
    packing_schema = typed_extraction_dataset(PackingSlip).model_json_schema_with_evaluators(
        [ExtractionMetrics],
        [ExtractionMetricsReport],
    )
    packing_text = json.dumps(packing_schema)
    assert "PackingSlip" in packing_text


def test_extraction_dataset_roundtrip_keeps_invoice_type(tmp_path: Path) -> None:
    invoice = Invoice(doc_type="invoice", number="INV-9")
    dataset_cls = typed_extraction_dataset(Invoice)
    original = dataset_cls(
        name="extraction-invoice",
        cases=[
            Case(
                name="invoice/sample",
                inputs=ExtractionInput(pdf_path=Path("data/sample.pdf"), doc_type="invoice"),
                expected_output=invoice,
            )
        ],
        evaluators=[ExtractionMetrics()],
        report_evaluators=[ExtractionMetricsReport()],
    )
    path = tmp_path / "extraction-invoice.yaml"
    original.to_file(
        path,
        custom_evaluator_types=[ExtractionMetrics],
        custom_report_evaluator_types=[ExtractionMetricsReport],
    )

    loaded = load_extraction_dataset(path, Invoice)
    gold = loaded.cases[0].expected_output
    assert isinstance(gold, Invoice)
    assert gold.number == "INV-9"
    assert loaded.cases[0].inputs.doc_type == "invoice"


def test_classification_metrics_drive_confusion_matrix() -> None:
    dataset = ClassificationDataset(
        name="classification",
        cases=[
            Case(
                name="invoice/a",
                inputs=ClassificationInput(pdf_path=Path("a.pdf")),
                expected_output=DocumentClassification(document_type="invoice"),
            ),
            Case(
                name="quote/b",
                inputs=ClassificationInput(pdf_path=Path("b.pdf")),
                expected_output=DocumentClassification(document_type="quote"),
            ),
        ],
        evaluators=[ClassificationMetrics()],
        report_evaluators=[classification_confusion_matrix()],
    )

    async def stub(inputs: ClassificationInput) -> DocumentClassification:
        if inputs.pdf_path.name == "a.pdf":
            return DocumentClassification(document_type="invoice")
        return DocumentClassification(document_type="invoice")

    report = asyncio.run(dataset.evaluate(stub, progress=False))
    by_name = {case.name: case for case in report.cases}

    assert by_name["invoice/a"].assertions["correct"].value is True
    assert by_name["quote/b"].assertions["correct"].value is False
    assert by_name["quote/b"].labels["predicted_type"].value == "invoice"
    assert by_name["quote/b"].labels["expected_type"].value == "quote"
    averages = report.averages()
    assert averages is not None
    assert averages.assertions == 0.5
    assert report.analyses
    matrix = report.analyses[0]
    assert getattr(matrix, "title", None) == "Document type confusion matrix"


def test_untyped_dataset_does_not_bind_invoice_schema() -> None:
    schema = Dataset.model_json_schema_with_evaluators()
    text = json.dumps(schema)
    assert "Invoice" not in text
