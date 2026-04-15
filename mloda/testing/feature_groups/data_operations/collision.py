"""Shared assertions for helper-column collision tests.

Every collision test across the data_operations feature groups follows
the same shape:

1. Build an input frame containing a user column whose name matches an
   internal helper-column name.
2. Run the implementation's private ``_compute_*`` method.
3. Assert the user column survived unchanged and the output column has
   the expected values.

Input construction stays in each test (types, kwargs, and expected
output values differ per feature group), but the assertion side is
framework-agnostic and is the same everywhere, so it lives here.
"""

from __future__ import annotations

import math
from typing import Any

import pyarrow as pa

from mloda.testing.feature_groups.data_operations.helpers import extract_column


def _normalize(values: list[Any]) -> list[Any]:
    """Normalize NaN (from pandas) to None so expected/actual compare cleanly."""
    out: list[Any] = []
    for v in values:
        if isinstance(v, float) and math.isnan(v):
            out.append(None)
        else:
            out.append(v)
    return out


def column_names(result: Any) -> list[str]:
    """Return the column names of *result* across all supported frameworks.

    Handles pa.Table (``column_names``), DuckDB/SQLite relations
    (via ``to_arrow_table()``), Polars LazyFrames (via ``collect()``),
    and pandas DataFrames (``columns``).
    """
    if isinstance(result, pa.Table):
        return list(result.column_names)
    if hasattr(result, "to_arrow_table"):
        return list(result.to_arrow_table().column_names)
    if hasattr(result, "collect"):
        return list(result.collect().columns)
    return list(result.columns)


def assert_user_column_preserved(
    result: Any,
    collision_col: str,
    expected_values: list[Any],
) -> None:
    """Assert the user-supplied column *collision_col* survived unchanged."""
    names = column_names(result)
    assert collision_col in names, f"user column '{collision_col}' was dropped (columns: {names})"
    actual = _normalize(extract_column(result, collision_col))
    expected_norm = _normalize(list(expected_values))
    assert actual == expected_norm, (
        f"user column '{collision_col}' values changed: got {actual}, expected {expected_norm}"
    )


def assert_output_column(
    result: Any,
    feature_name: str,
    expected_values: list[Any],
) -> None:
    """Assert *feature_name* is present with the expected values."""
    names = column_names(result)
    assert feature_name in names, f"output column '{feature_name}' missing (columns: {names})"
    actual = _normalize(extract_column(result, feature_name))
    expected_norm = _normalize(list(expected_values))
    assert actual == expected_norm, f"output column '{feature_name}' values: got {actual}, expected {expected_norm}"


def assert_collision_preserved(
    result: Any,
    collision_col: str,
    user_values: list[Any],
    feature_name: str,
    expected_values: list[Any],
) -> None:
    """Assert both: user column survived and output column has expected values.

    Use for row-preserving feature groups (frame_aggregate, offset, rank,
    window_aggregation, percentile, binning, scalar_aggregate).
    """
    assert_user_column_preserved(result, collision_col, user_values)
    assert_output_column(result, feature_name, expected_values)


def assert_column_absent(result: Any, column_name: str) -> None:
    """Assert *column_name* did not leak into the result.

    Use for the group-by aggregation case where the temp mask column must
    not be returned alongside the grouped output.
    """
    names = column_names(result)
    assert column_name not in names, f"column '{column_name}' leaked into result (columns: {names})"
