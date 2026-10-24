"""Shared utilities for conditional aggregation via FilterMask.

Provides mask-building strategies matched to framework capabilities:

1. ``build_mask_from_spec`` -- universal dispatcher that works with any
   ``BaseMaskEngine`` subclass (Pandas, PyArrow, and others).
2. ``build_polars_mask_expr`` -- for Polars lazy evaluation, delegates to
   upstream ``PolarsExprMaskEngine`` (via ``build_mask_from_spec``) to build a
   ``pl.Expr`` boolean chain.
3. ``build_sql_case_when`` -- for SQL-based frameworks (DuckDB, SQLite),
   delegates to the framework's concrete SQL mask engine (e.g.
   ``DuckDBMaskEngine``, ``SqliteMaskEngine``) for individual conditions,
   then wraps them in a ``CASE WHEN ... THEN source END`` expression.

Apply helpers ``apply_polars_mask`` and ``apply_pandas_mask`` add a masked temp column
(``MASKED_SOURCE_HELPER``, uniquified) and leave the source and keys unchanged;
``apply_pyarrow_mask`` returns the masked column array.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mloda.provider import BaseMaskEngine

from mloda.community.feature_groups.data_operations.helper_columns import unique_helper_name

MASK_KEY = "mask"

_SUPPORTED_OPS: frozenset[str] = frozenset(
    {"equal", "greater_than", "greater_equal", "less_equal", "less_than", "is_in"}
)


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
        if op not in _SUPPORTED_OPS:
            raise ValueError(f"Unsupported mask operator '{op}'. Supported: {sorted(_SUPPORTED_OPS)}")

        if len(spec) == 2 and op != "equal":
            raise ValueError(
                f"2-element mask tuple ('column', 'operator') is only valid for 'equal' (null or NaN). "
                f"Operator '{op}' requires a value: ('column', '{op}', value)."
            )

        if op == "is_in":
            if isinstance(val, (str, bytes)) or not isinstance(val, (list, tuple, set, frozenset)):
                raise ValueError(
                    f"is_in values must be a list, tuple, or set, got {type(val).__name__}. "
                    f"Use e.g. ('col', 'is_in', ['a', 'b']) instead of ('col', 'is_in', 'ab')."
                )
            if len(val) == 0:
                raise ValueError("is_in values must not be empty. Use a different condition to match nothing.")
        elif val is not None and not isinstance(val, (bool, int, float, str)):
            raise ValueError(
                f"Mask value must be None, bool, int, float, or str, got {type(val).__name__}. "
                f"Types like datetime or Decimal are not supported in mask conditions."
            )

        parsed.append((col, op, val))

    return parsed


def build_mask_from_spec(
    engine_cls: type[BaseMaskEngine],
    data: Any,
    mask_spec: list[tuple[str, str, Any]],
) -> Any:
    """Build a boolean mask using a MaskEngine (Pandas, PyArrow).

    Returns a framework-native boolean array/series.
    """
    mask = engine_cls.all_true(data)
    for col, op, val in mask_spec:
        single = _engine_op(engine_cls, data, col, op, val)
        mask = engine_cls.combine(mask, single)
    return mask


def _engine_op(
    engine_cls: type[BaseMaskEngine],
    data: Any,
    col: str,
    op: str,
    val: Any,
) -> Any:
    """Dispatch a single mask condition to the engine method."""
    if op == "equal":
        return engine_cls.equal(data, col, val)
    if op == "greater_than":
        return engine_cls.greater_than(data, col, val)
    if op == "greater_equal":
        return engine_cls.greater_equal(data, col, val)
    if op == "less_equal":
        return engine_cls.less_equal(data, col, val)
    if op == "less_than":
        return engine_cls.less_than(data, col, val)
    if op == "is_in":
        return engine_cls.is_in(data, col, val)
    raise ValueError(f"Unsupported mask operator: {op}")


def build_polars_mask_expr(data: Any, mask_spec: list[tuple[str, str, Any]]) -> Any:
    """Build a lazy-compatible Polars boolean expression from a mask spec.

    Returns a ``pl.Expr`` that evaluates to a boolean column.  Delegates to
    upstream ``PolarsExprMaskEngine``, which reads ``data.collect_schema()``
    for null/NaN-aware conditions, so *data* must be the LazyFrame/DataFrame.
    """
    from mloda_plugins.compute_framework.base_implementations.polars.polars_expr_mask_engine import (
        PolarsExprMaskEngine,
    )

    return build_mask_from_spec(PolarsExprMaskEngine, data, mask_spec)


MASKED_SOURCE_HELPER = "__mloda_masked_src__"


def apply_polars_mask(
    data: Any,
    source_col: str,
    mask_spec: list[tuple[str, str, Any]],
    taken: Iterable[str] = (),
) -> tuple[Any, str]:
    """Add a masked copy of *source_col* to a Polars LazyFrame under a collision-free temp name.

    Returns ``(data_with_temp, temp_name)``. The source column is left unchanged. Callers drop
    *temp_name* when done, except after a group_by (only keys and aggregates are returned there).
    """
    import polars as pl

    temp_name = unique_helper_name(MASKED_SOURCE_HELPER, set(data.collect_schema().names()) | set(taken))
    mask_expr = build_polars_mask_expr(data, mask_spec)
    data = data.with_columns(pl.when(mask_expr).then(pl.col(source_col)).otherwise(None).alias(temp_name))
    return data, temp_name


def apply_pandas_mask(
    data: Any,
    source_col: str,
    mask_spec: list[tuple[str, str, Any]],
    taken: Iterable[str] = (),
) -> tuple[Any, str]:
    """Return a copy of a pandas DataFrame with a masked copy of *source_col* under a temp name.

    Returns ``(data_copy, temp_name)``. The source column and all keys are left unchanged.
    """
    from mloda_plugins.compute_framework.base_implementations.pandas.pandas_mask_engine import PandasMaskEngine

    temp_name = unique_helper_name(MASKED_SOURCE_HELPER, set(data.columns) | set(taken))
    mask = build_mask_from_spec(PandasMaskEngine, data, mask_spec)
    data = data.copy()
    data[temp_name] = data[source_col].where(mask)
    return data, temp_name


def apply_pyarrow_mask(
    table: Any,
    source_col: str,
    mask_spec: list[tuple[str, str, Any]],
) -> Any:
    """Return the *source_col* array of a PyArrow table with non-matching rows set to null."""
    import pyarrow as pa
    import pyarrow.compute as pc
    from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_mask_engine import (
        PyArrowMaskEngine,
    )

    mask = build_mask_from_spec(PyArrowMaskEngine, table, mask_spec)
    null_scalar = pa.scalar(None, type=table.schema.field(source_col).type)
    return pc.if_else(mask, table.column(source_col), null_scalar)


def build_sql_case_when(
    engine_cls: type[BaseMaskEngine],
    data: Any,
    mask_spec: list[tuple[str, str, Any]],
    source_expr: str,
) -> str:
    """Build a SQL ``CASE WHEN ... THEN source END`` expression.

    Delegates individual conditions to the framework's concrete mask engine
    so core null/NaN semantics are preserved.

    *data* must be the relation the SQL runs against: it must hold every mask
    column, since DuckDB's engine reads column types from it to add the NaN guard.

    *source_expr* is a SQL expression, usually a quoted identifier (the
    frame-aggregate caller passes an alias-qualified ``s."col"``).
    """
    conditions = []
    for col, op, val in mask_spec:
        conditions.append(_engine_op(engine_cls, data, col, op, val))

    where_clause = " AND ".join(conditions)
    return f"CASE WHEN {where_clause} THEN {source_expr} END"
