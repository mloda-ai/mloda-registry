"""Shared utilities for conditional aggregation via FilterMask.

Provides three mask-building strategies matched to framework capabilities:

1. ``build_mask_from_spec`` -- for DataFrame-based frameworks (Pandas, PyArrow)
   that have a ``BaseFilterMaskEngine`` returning native boolean arrays.
2. ``build_polars_mask_expr`` -- for Polars lazy evaluation, building a
   ``pl.Expr`` boolean chain directly (the engine needs an eager DataFrame).
3. ``build_sql_case_when`` -- for SQL-based frameworks (DuckDB, SQLite),
   generating a ``CASE WHEN ... THEN source END`` expression.
"""

from __future__ import annotations

from typing import Any

from mloda.core.filter.filter_mask_engine import BaseFilterMaskEngine
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident, quote_value

MASK_KEY = "mask"

MASK_OPERATORS: dict[str, str] = {
    "equal": "=",
    "greater_equal": ">=",
    "less_equal": "<=",
    "less_than": "<",
    "is_in": "IN",
}


def parse_mask_spec(mask_option: Any) -> list[tuple[str, str, Any]] | None:
    """Normalize a mask option value into a list of (column, operator, value) tuples.

    Accepts a single tuple or a list of tuples. Returns None if *mask_option*
    is None (mask not requested).

    Raises ValueError for malformed specs.
    """
    if mask_option is None:
        return None

    if isinstance(mask_option, tuple):
        specs = [mask_option]
    elif isinstance(mask_option, list):
        specs = mask_option
    else:
        raise ValueError(f"mask must be a tuple or list of tuples, got {type(mask_option).__name__}")

    parsed: list[tuple[str, str, Any]] = []
    for spec in specs:
        if not isinstance(spec, tuple):
            raise ValueError(f"Each mask condition must be a tuple, got {type(spec).__name__}")
        if len(spec) < 2 or len(spec) > 3:
            raise ValueError(f"Mask tuple must have 2 or 3 elements (column, operator[, value]), got {len(spec)}")

        col, op = spec[0], spec[1]
        val = spec[2] if len(spec) == 3 else None

        if not isinstance(col, str):
            raise ValueError(f"Mask column must be a string, got {type(col).__name__}")
        if not isinstance(op, str):
            raise ValueError(f"Mask operator must be a string, got {type(op).__name__}")
        if op not in MASK_OPERATORS:
            raise ValueError(f"Unsupported mask operator '{op}'. Supported: {sorted(MASK_OPERATORS)}")

        parsed.append((col, op, val))

    return parsed


def build_mask_from_spec(
    engine_cls: type[BaseFilterMaskEngine],
    data: Any,
    mask_spec: list[tuple[str, str, Any]],
) -> Any:
    """Build a boolean mask using a FilterMaskEngine (Pandas, PyArrow).

    Returns a framework-native boolean array/series.
    """
    mask = engine_cls.all_true(data)
    for col, op, val in mask_spec:
        single = _engine_op(engine_cls, data, col, op, val)
        mask = engine_cls.combine(mask, single)
    return mask


def _engine_op(
    engine_cls: type[BaseFilterMaskEngine],
    data: Any,
    col: str,
    op: str,
    val: Any,
) -> Any:
    """Dispatch a single mask condition to the engine method."""
    if op == "equal":
        return engine_cls.equal(data, col, val)
    if op == "greater_equal":
        return engine_cls.greater_equal(data, col, val)
    if op == "less_equal":
        return engine_cls.less_equal(data, col, val)
    if op == "less_than":
        return engine_cls.less_than(data, col, val)
    if op == "is_in":
        return engine_cls.is_in(data, col, val)
    raise ValueError(f"Unsupported mask operator: {op}")


def build_polars_mask_expr(mask_spec: list[tuple[str, str, Any]]) -> Any:
    """Build a lazy-compatible Polars boolean expression from a mask spec.

    Returns a ``pl.Expr`` that evaluates to a boolean column.
    """
    import polars as pl

    expr: pl.Expr | None = None
    for col, op, val in mask_spec:
        single = _polars_condition(pl, col, op, val)
        expr = single if expr is None else (expr & single)
    return expr


def _polars_condition(pl: Any, col: str, op: str, val: Any) -> Any:
    """Build a single Polars boolean expression for one condition."""
    if op == "equal":
        return pl.col(col) == val
    if op == "greater_equal":
        return pl.col(col) >= val
    if op == "less_equal":
        return pl.col(col) <= val
    if op == "less_than":
        return pl.col(col) < val
    if op == "is_in":
        return pl.col(col).is_in(val)
    raise ValueError(f"Unsupported mask operator: {op}")


def build_sql_case_when(
    mask_spec: list[tuple[str, str, Any]],
    source_expr: str,
) -> str:
    """Build a SQL ``CASE WHEN ... THEN source END`` expression.

    *source_expr* should already be a quoted identifier (via ``quote_ident``).
    Column names in conditions are quoted; literal values use ``quote_value``.
    """
    conditions = []
    for col, op, val in mask_spec:
        quoted_col = quote_ident(col)
        if op == "is_in":
            values_sql = ", ".join(quote_value(v) for v in val)
            conditions.append(f"{quoted_col} IN ({values_sql})")
        else:
            sql_op = MASK_OPERATORS[op]
            conditions.append(f"{quoted_col} {sql_op} {quote_value(val)}")

    where_clause = " AND ".join(conditions)
    return f"CASE WHEN {where_clause} THEN {source_expr} END"
