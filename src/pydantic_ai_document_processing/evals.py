from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from math import isclose
from typing import Any

from pydantic import BaseModel
from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
    ReportEvaluator,
    ReportEvaluatorContext,
)
from pydantic_evals.reporting.analyses import ScalarResult, TableResult

_LIST_ALIGN_KEYS = ("sku", "description", "rate", "label")
_MAX_REASON_MISMATCHES = 8
_REASON_VALUE_CHARS = 80


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: Counts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0

    def f1(self) -> float:
        precision = self.precision()
        recall = self.recall()
        denom = precision + recall
        return (2 * precision * recall / denom) if denom else 1.0

    def support(self) -> int:
        return self.tp + self.fn


@dataclass
class Mismatch:
    path: str
    gold: Any
    pred: Any


@dataclass
class ExtractionComparison:
    fields: Counts = field(default_factory=Counts)
    items: Counts = field(default_factory=Counts)
    extra_keys: Counts = field(default_factory=Counts)
    sections: dict[str, Counts] = field(default_factory=dict)
    mismatches: list[Mismatch] = field(default_factory=list)


def compare_extractions(
    expected: Any,
    output: Any,
    *,
    numeric_rel_tol: float = 0.0,
) -> ExtractionComparison:
    """Score nested extraction models by aligned leaves, not whole-object equality."""
    comparison = ExtractionComparison()
    gold = _to_plain(expected)
    pred = _to_plain(output)
    _compare_value(gold, pred, path="", comparison=comparison, numeric_rel_tol=numeric_rel_tol)
    return comparison


def _to_plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    return value


def _section(path: str) -> str:
    if not path:
        return "_root"
    return path.split(".", 1)[0]


def _join(path: str, key: str | int) -> str:
    piece = str(key)
    return piece if not path else f"{path}.{piece}"


def _section_counts(comparison: ExtractionComparison, path: str) -> Counts:
    name = _section(path)
    counts = comparison.sections.get(name)
    if counts is None:
        counts = Counts()
        comparison.sections[name] = counts
    return counts


def _record_tp(comparison: ExtractionComparison, path: str, *, extra: bool = False) -> None:
    target = comparison.extra_keys if extra else comparison.fields
    target.tp += 1
    _section_counts(comparison, path).tp += 1


def _record_fn(
    comparison: ExtractionComparison,
    path: str,
    gold: Any,
    pred: Any,
    *,
    extra: bool = False,
) -> None:
    target = comparison.extra_keys if extra else comparison.fields
    target.fn += 1
    _section_counts(comparison, path).fn += 1
    if not extra:
        comparison.mismatches.append(Mismatch(path=path, gold=gold, pred=pred))


def _record_fp(
    comparison: ExtractionComparison,
    path: str,
    gold: Any,
    pred: Any,
    *,
    extra: bool = False,
) -> None:
    target = comparison.extra_keys if extra else comparison.fields
    target.fp += 1
    _section_counts(comparison, path).fp += 1
    if not extra:
        comparison.mismatches.append(Mismatch(path=path, gold=gold, pred=pred))


def _as_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int | float):
        return Decimal(str(value))
    if isinstance(value, str):
        try:
            return Decimal(value.strip().replace(",", ""))
        except (InvalidOperation, ValueError):
            return None
    return None


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def _canonicalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def _values_equal(gold: Any, pred: Any, numeric_rel_tol: float) -> bool:
    gold_dec = _as_decimal(gold)
    pred_dec = _as_decimal(pred)
    if gold_dec is not None and pred_dec is not None:
        if numeric_rel_tol > 0:
            return isclose(float(gold_dec), float(pred_dec), rel_tol=numeric_rel_tol, abs_tol=0.0)
        return gold_dec == pred_dec

    gold_date = _as_date(gold)
    pred_date = _as_date(pred)
    if gold_date is not None and pred_date is not None and (
        isinstance(gold, date | datetime | str) and isinstance(pred, date | datetime | str)
    ):
        return gold_date == pred_date

    return _canonicalize_scalar(gold) == _canonicalize_scalar(pred)


def _is_empty(value: Any) -> bool:
    return value is None or value == {} or value == []


def _is_record_list(value: Any) -> bool:
    return isinstance(value, list) and any(isinstance(item, dict) for item in value)


def _find_aligned_pred(
    gold_row: dict[str, Any],
    pred_records: list[dict[str, Any]],
    used: set[int],
) -> int | None:
    for key in _LIST_ALIGN_KEYS:
        gold_value = gold_row.get(key)
        if gold_value is None:
            continue
        gold_canonical = _canonicalize_scalar(gold_value)
        for pred_index, pred_row in enumerate(pred_records):
            if pred_index in used:
                continue
            pred_value = pred_row.get(key)
            if pred_value is None:
                continue
            if _canonicalize_scalar(pred_value) == gold_canonical:
                return pred_index
    return None


def _flatten_extra_keys(value: Any, path: str) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        if path:
            keys.add(path)
        for key, child in value.items():
            keys |= _flatten_extra_keys(child, _join(path, key))
        return keys
    if isinstance(value, list):
        if path:
            keys.add(path)
        for index, child in enumerate(value):
            if isinstance(child, dict | list):
                keys |= _flatten_extra_keys(child, _join(path, index))
        return keys
    if path and value is not None:
        keys.add(path)
    return keys


def _compare_extra(
    gold: Any,
    pred: Any,
    path: str,
    comparison: ExtractionComparison,
) -> None:
    gold_keys = _flatten_extra_keys(gold, path)
    pred_keys = _flatten_extra_keys(pred, path)
    for key in sorted(gold_keys | pred_keys):
        in_gold = key in gold_keys
        in_pred = key in pred_keys
        if in_gold and in_pred:
            _record_tp(comparison, key, extra=True)
        elif in_gold:
            _record_fn(comparison, key, True, None, extra=True)
        else:
            _record_fp(comparison, key, None, True, extra=True)


def _compare_value(
    gold: Any,
    pred: Any,
    path: str,
    comparison: ExtractionComparison,
    numeric_rel_tol: float,
) -> None:
    if path.split(".", 1)[0] == "extra" or path == "extra":
        _compare_extra(gold, pred, path, comparison)
        return

    if _is_empty(gold) and _is_empty(pred):
        return

    if isinstance(gold, dict) or isinstance(pred, dict):
        gold_dict = gold if isinstance(gold, dict) else {}
        pred_dict = pred if isinstance(pred, dict) else {}
        if not isinstance(gold, dict) and not _is_empty(gold):
            _record_fn(comparison, path or "_root", gold, pred)
            return
        if not isinstance(pred, dict) and not _is_empty(pred):
            _record_fp(comparison, path or "_root", gold, pred)
            return
        keys = set(gold_dict) | set(pred_dict)
        for key in keys:
            child_path = _join(path, key)
            if key == "extra":
                _compare_extra(gold_dict.get(key), pred_dict.get(key), child_path, comparison)
                continue
            _compare_value(
                gold_dict.get(key),
                pred_dict.get(key),
                child_path,
                comparison,
                numeric_rel_tol,
            )
        return

    if _is_record_list(gold) or _is_record_list(pred):
        _compare_record_list(
            gold if isinstance(gold, list) else [],
            pred if isinstance(pred, list) else [],
            path,
            comparison,
            numeric_rel_tol,
        )
        return

    if isinstance(gold, list) or isinstance(pred, list):
        gold_list = gold if isinstance(gold, list) else []
        pred_list = pred if isinstance(pred, list) else []
        length = max(len(gold_list), len(pred_list))
        for index in range(length):
            _compare_value(
                gold_list[index] if index < len(gold_list) else None,
                pred_list[index] if index < len(pred_list) else None,
                _join(path, index),
                comparison,
                numeric_rel_tol,
            )
        return

    if _is_empty(gold):
        _record_fp(comparison, path, gold, pred)
        return
    if _is_empty(pred):
        _record_fn(comparison, path, gold, pred)
        return
    if _values_equal(gold, pred, numeric_rel_tol):
        _record_tp(comparison, path)
        return
    _record_fn(comparison, path, gold, pred)
    comparison.fields.fp += 1
    _section_counts(comparison, path).fp += 1


def _compare_record_list(
    gold_rows: list[Any],
    pred_rows: list[Any],
    path: str,
    comparison: ExtractionComparison,
    numeric_rel_tol: float,
) -> None:
    gold_records = [row if isinstance(row, dict) else {"_value": row} for row in gold_rows]
    pred_records = [row if isinstance(row, dict) else {"_value": row} for row in pred_rows]
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    unmatched_gold: list[int] = []

    for gold_index, gold_row in enumerate(gold_records):
        match_index = _find_aligned_pred(gold_row, pred_records, used)
        if match_index is None:
            unmatched_gold.append(gold_index)
            continue
        used.add(match_index)
        pairs.append((gold_index, match_index))

    leftover_gold: list[int] = []
    for gold_index in unmatched_gold:
        leftover = next(
            (pred_index for pred_index in range(len(pred_records)) if pred_index not in used),
            None,
        )
        if leftover is None:
            leftover_gold.append(gold_index)
            continue
        used.add(leftover)
        pairs.append((gold_index, leftover))
    unmatched_gold = leftover_gold

    unmatched_pred = [index for index in range(len(pred_records)) if index not in used]

    comparison.items.tp += len(pairs)
    comparison.items.fn += len(unmatched_gold)
    comparison.items.fp += len(unmatched_pred)

    for gold_index, pred_index in pairs:
        _compare_value(
            gold_records[gold_index],
            pred_records[pred_index],
            _join(path, gold_index),
            comparison,
            numeric_rel_tol,
        )
    for gold_index in unmatched_gold:
        _compare_value(
            gold_records[gold_index],
            None,
            _join(path, gold_index),
            comparison,
            numeric_rel_tol,
        )
    for pred_index in unmatched_pred:
        _compare_value(
            None,
            pred_records[pred_index],
            _join(path, f"unmatched_{pred_index}"),
            comparison,
            numeric_rel_tol,
        )


def _truncate(value: Any) -> str:
    text = repr(value)
    if len(text) > _REASON_VALUE_CHARS:
        return text[: _REASON_VALUE_CHARS - 3] + "..."
    return text


def _mismatch_reason(comparison: ExtractionComparison) -> str | None:
    if not comparison.mismatches:
        return None
    lines = [
        f"{item.path}: gold={_truncate(item.gold)} pred={_truncate(item.pred)}"
        for item in comparison.mismatches[:_MAX_REASON_MISMATCHES]
    ]
    remaining = len(comparison.mismatches) - _MAX_REASON_MISMATCHES
    if remaining > 0:
        lines.append(f"... {remaining} more mismatches")
    return "; ".join(lines)


def _round(value: float) -> float:
    return round(value, 4)


def _doc_type_of(case: Any) -> str:
    inputs = getattr(case, "inputs", None)
    doc_type = getattr(inputs, "doc_type", None)
    if doc_type:
        return str(doc_type)
    expected = getattr(case, "expected_output", None)
    if isinstance(expected, BaseModel):
        value = getattr(expected, "doc_type", None)
        if value:
            return str(value)
    if isinstance(expected, dict) and expected.get("doc_type"):
        return str(expected["doc_type"])
    return "unknown"


@dataclass(repr=False)
class ExtractionMetrics(Evaluator):
    """Field-level extraction scores for nested document schemas."""

    pass_threshold: float = 0.9
    numeric_rel_tol: float = 0.0

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, EvaluationReason | float | bool]:
        if ctx.expected_output is None or ctx.output is None:
            return {}
        comparison = compare_extractions(
            ctx.expected_output,
            ctx.output,
            numeric_rel_tol=self.numeric_rel_tol,
        )
        field_f1 = comparison.fields.f1()
        return {
            "field_precision": _round(comparison.fields.precision()),
            "field_recall": _round(comparison.fields.recall()),
            "field_f1": EvaluationReason(value=_round(field_f1), reason=_mismatch_reason(comparison)),
            "items_f1": _round(comparison.items.f1()),
            "extra_key_f1": _round(comparison.extra_keys.f1()),
            "extraction_pass": field_f1 >= self.pass_threshold,
        }


@dataclass(repr=False)
class ExtractionMetricsReport(ReportEvaluator):
    """Experiment-wide extraction F1 by schema section and document type."""

    numeric_rel_tol: float = 0.0

    def evaluate(self, ctx: ReportEvaluatorContext) -> list[ScalarResult | TableResult]:
        overall = Counts()
        sections: dict[str, Counts] = defaultdict(Counts)
        by_type: dict[str, Counts] = defaultdict(Counts)

        for case in ctx.report.cases:
            if case.expected_output is None or case.output is None:
                continue
            comparison = compare_extractions(
                case.expected_output,
                case.output,
                numeric_rel_tol=self.numeric_rel_tol,
            )
            overall.add(comparison.fields)
            by_type[_doc_type_of(case)].add(comparison.fields)
            for name, counts in comparison.sections.items():
                sections[name].add(counts)

        section_rows = [
            [
                name,
                _round(counts.precision()),
                _round(counts.recall()),
                _round(counts.f1()),
                counts.support(),
            ]
            for name, counts in sorted(sections.items())
        ]
        type_rows = [
            [
                name,
                _round(counts.precision()),
                _round(counts.recall()),
                _round(counts.f1()),
                counts.support(),
            ]
            for name, counts in sorted(by_type.items())
        ]
        columns = ["name", "precision", "recall", "f1", "support"]
        return [
            ScalarResult(
                title="Extraction field F1",
                description="Micro-averaged leaf F1 across all extraction cases.",
                value=_round(overall.f1()),
            ),
            TableResult(
                title="Extraction metrics by section",
                description="Leaf precision, recall, and F1 grouped by top-level schema field.",
                columns=columns,
                rows=section_rows,
            ),
            TableResult(
                title="Extraction metrics by document type",
                description="Leaf precision, recall, and F1 grouped by doc_type.",
                columns=columns,
                rows=type_rows,
            ),
        ]
