"""Shared constants and utilities for all data operation feature groups."""

from __future__ import annotations

from enum import Enum
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Shared config key constants
# ---------------------------------------------------------------------------
# These are the keys used in Options to configure data operation feature groups.
# Core's DefaultOptionKeys defines "group" and "order_by" as enum members.
# The constants below are the string values used in Options dictionaries
# across all 9 data operation packages.

PARTITION_BY = "partition_by"
"""Config key for partitioning columns (list of column names).

Used by: window_aggregation, group_aggregation, rank, offset, frame_aggregate.
"""

ORDER_BY = "order_by"
"""Config key for ordering columns (list of column names).

Used by: rank, offset, frame_aggregate.
"""

FRAME_TYPE = "frame_type"
"""Config key for frame type in frame_aggregate operations.

Valid values: "rows", "time", "expanding", "cumulative".
"""

FRAME_SIZE = "frame_size"
"""Config key for frame size (positive integer).

Used by frame_aggregate with frame_type "rows" or "time".
"""

FRAME_UNIT = "frame_unit"
"""Config key for time unit in time-interval frames.

Used by frame_aggregate with frame_type "time".
Valid values: second, minute, hour, day, week, month, year.
"""


# ---------------------------------------------------------------------------
# Null handling policy constants (ticket 239)
# ---------------------------------------------------------------------------
# These document the data operations null handling contract.
# Implementations must match these defaults. PyArrow behavior is the reference.


class NullPolicy(str, Enum):
    """Null handling behavior constants for data operations.

    Each value describes a null handling rule. Implementations must match
    PyArrow's behavior as the reference. Where a framework diverges from
    these defaults, add explicit convergence code (e.g. pandas groupby
    needs ``dropna=False``; SQLite rank needs an explicit null-last clause).

    These constants are documentation and configuration anchors, not
    runtime enforcement. Each package's ``calculate_feature`` is responsible
    for honoring the policy.
    """

    PROPAGATE = "propagate"
    """Element-wise operations return null for null input (null in, null out).

    Applies to: datetime, string, binning.
    """

    SKIP = "skip"
    """Aggregations skip null values (e.g. SUM ignores nulls).

    Applies to: window_aggregation, group_aggregation, frame_aggregate.
    """

    NULL_IS_GROUP = "null_is_group"
    """Null is a valid group key in partitioned operations.

    Applies to: window_aggregation, group_aggregation, rank, offset, frame_aggregate.
    Pandas divergence: pass ``dropna=False`` to ``groupby()``.
    """

    NULLS_LAST = "nulls_last"
    """Nulls rank last in ordered operations.

    Applies to: rank.
    SQLite divergence: add ``CASE WHEN col IS NULL THEN 1 ELSE 0 END`` to ORDER BY.
    """

    EDGE_NULL = "edge_null"
    """Out-of-range positions produce null (e.g. lag/lead at table edges).

    Applies to: offset.
    """


# ---------------------------------------------------------------------------
# Column type categories for type checking (ticket 238)
# ---------------------------------------------------------------------------


class ColumnTypeCategory(str, Enum):
    """Expected column type categories for data operation type checks.

    Each data operation package declares which category it requires.
    The ``check_column_type`` helper validates at execution time.
    """

    NUMERIC = "numeric"
    """Integer or floating-point columns.

    Required by: aggregation, window_aggregation, group_aggregation,
    rank, offset, frame_aggregate, binning (value column).
    """

    STRING = "string"
    """String/text/utf8 columns.

    Required by: string operations.
    """

    DATETIME = "datetime"
    """Timestamp or date columns.

    Required by: datetime extraction.
    """

    ANY = "any"
    """Any column type is accepted.

    Used for: partition_by columns in grouped operations.
    """


# ---------------------------------------------------------------------------
# Type-check helper (ticket 238)
# ---------------------------------------------------------------------------
# Registry of framework-specific type checkers. Each framework implementation
# registers its own checker via ``register_type_checker()``. This keeps
# framework-specific imports out of this shared base module.

_TYPE_CHECKER_REGISTRY: dict[str, dict[ColumnTypeCategory, Any]] = {}


def register_type_checker(framework_name: str, checkers: dict[ColumnTypeCategory, Any]) -> None:
    """Register type-checking predicates for a framework.

    Called by framework-specific modules at import time. Each checker
    maps a ``ColumnTypeCategory`` to a callable that returns True when
    the column type matches.
    """
    _TYPE_CHECKER_REGISTRY[framework_name] = checkers


def _get_column_type_info(data: Any, column_name: str) -> tuple[str, Any, str]:
    """Extract the column type and framework name from a data object.

    Uses PyArrow as the default (always available). For other frameworks,
    checks if they are installed and if the data matches.

    Returns:
        Tuple of (framework_name, column_type, type_string).

    Raises:
        TypeError: If the data type is not recognized.
    """
    import pyarrow as pa

    if isinstance(data, pa.Table):
        col_type = data.schema.field(column_name).type
        return "pyarrow", col_type, str(col_type)

    # Check registered frameworks
    for name in _TYPE_CHECKER_REGISTRY:
        if name == "pyarrow":
            continue
        if name == "pandas":
            pd = _try_import("pandas")
            if pd is not None and isinstance(data, pd.DataFrame):
                col_type = data[column_name].dtype
                return "pandas", col_type, str(col_type)
        elif name == "polars":
            pl = _try_import("polars")
            if pl is not None and isinstance(data, (pl.DataFrame, pl.LazyFrame)):
                col_type = data.schema[column_name]
                return "polars", col_type, str(col_type)

    raise TypeError(
        f"Unsupported data type for column type checking: {type(data).__name__}. "
        f"No type checker registered for this framework."
    )


def _try_import(module_name: str) -> Any:
    """Import a module, returning None if not installed."""
    import importlib

    try:  # noqa: SIM105 (intentional: import fallback is a valid use of try/except)
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _pyarrow_type_checkers() -> dict[ColumnTypeCategory, Any]:
    """Return PyArrow type-checking predicates."""
    import pyarrow as pa

    return {
        ColumnTypeCategory.NUMERIC: lambda t: (
            pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t)
        ),
        ColumnTypeCategory.STRING: lambda t: pa.types.is_string(t) or pa.types.is_large_string(t),
        ColumnTypeCategory.DATETIME: lambda t: pa.types.is_timestamp(t) or pa.types.is_date(t),
        ColumnTypeCategory.ANY: lambda t: True,
    }


# Register PyArrow by default (always available as the reference framework)
register_type_checker("pyarrow", _pyarrow_type_checkers())


def check_column_type(
    data: Any,
    column_name: str,
    expected: ColumnTypeCategory,
    operation_name: str,
) -> None:
    """Validate that a column's type matches the expected category.

    Call this at the start of ``calculate_feature`` to give users a clear
    error instead of a cryptic framework exception.

    Args:
        data: The data object (PyArrow Table, pandas DataFrame, or Polars DataFrame/LazyFrame).
        column_name: Name of the column to check.
        expected: The required type category.
        operation_name: Human-readable operation name for the error message
            (e.g. "upper", "sum_groupby", "year").

    Raises:
        TypeError: If the column type does not match the expected category,
            with a message like "string operation 'upper' requires a string
            column, got int64".
    """
    if expected == ColumnTypeCategory.ANY:
        return

    framework, col_type, type_str = _get_column_type_info(data, column_name)

    checkers = _TYPE_CHECKER_REGISTRY.get(framework)
    if checkers is None:
        return  # No checker registered for this framework, skip validation

    checker = checkers[expected]
    if not checker(col_type):
        raise TypeError(
            f"{expected.value} operation '{operation_name}' requires a {expected.value} column, got {type_str}"
        )


def check_column_types(
    data: Any,
    column_names: Sequence[str],
    expected: ColumnTypeCategory,
    operation_name: str,
) -> None:
    """Validate that multiple columns match the expected type category.

    Convenience wrapper around ``check_column_type`` for operations that
    act on more than one column (e.g. multi-column aggregation).

    Args:
        data: The data object.
        column_names: Names of the columns to check.
        expected: The required type category.
        operation_name: Human-readable operation name for the error message.

    Raises:
        TypeError: On the first column that does not match.
    """
    for col in column_names:
        check_column_type(data, col, expected, operation_name)
