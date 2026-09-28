"""Shared helpers for data-operations tests.

Provides:
- ``extract_column``: Extract a column from any framework result as a Python list.
- ``make_feature_set``: Build a FeatureSet with optional partition_by/order_by.
- ``feature_set_for``: Build a FeatureSet around an Options that already exists.
- ``is_null``: True for None or a float NaN.
- ``assert_values_with_nulls``: Assert two lists match, null-aware, with optional cast/approx.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

import pyarrow as pa
import pytest
from mloda.provider import FeatureSet
from mloda.user import Feature, Options


def is_null(value: Any) -> bool:
    """True for None or a float NaN."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return False


def assert_values_with_nulls(
    actual: list[Any],
    expected: list[Any],
    *,
    approx: bool = False,
    rel: float = 1e-6,
    nan_is_null: bool = False,
    parse_iso: bool = False,
    cast: Callable[[Any], Any] | None = None,
) -> None:
    """Assert two lists are equal row by row, treating null consistently.

    ``nan_is_null`` treats NaN and None as interchangeable nulls, ``parse_iso`` parses a str
    actual value via ``datetime.fromisoformat``, ``cast`` converts the actual value before
    comparing, and ``approx`` compares with ``pytest.approx`` instead of exact equality.
    """
    assert len(actual) == len(expected), f"row count {len(actual)} != expected {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected)):
        expected_null = e is None or (nan_is_null and is_null(e))
        if expected_null:
            actual_null = a is None or (nan_is_null and is_null(a))
            assert actual_null, f"row {i}: expected None, got {a!r}"
        else:
            assert a is not None, f"row {i}: expected {e!r}, got None"
            if parse_iso and isinstance(a, str):
                a = datetime.fromisoformat(a)
            if cast is not None:
                a = cast(a)
            if approx:
                assert a == pytest.approx(e, rel=rel), f"row {i}: {a!r} != {e!r}"
            else:
                assert a == e, f"row {i}: {a!r} != {e!r}"


def extract_column(result: Any, column_name: str) -> list[Any]:
    """Extract a column from a result object as a Python list.

    Handles pa.Table (direct .column() access), relation types
    (DuckdbRelation, SqliteRelation) that expose .to_arrow_table(),
    Polars LazyFrames that expose .collect(), and pandas DataFrames.
    """
    if isinstance(result, pa.Table):
        return list(result.column(column_name).to_pylist())
    if hasattr(result, "to_arrow_table"):
        arrow_table = result.to_arrow_table()
        return list(arrow_table.column(column_name).to_pylist())
    if hasattr(result, "collect"):
        df = result.collect()
        return list(df[column_name].to_list())
    return list(result[column_name])


def make_feature_set(
    feature_name: str,
    partition_by: list[str] | None = None,
    order_by: str | None = None,
    mask: tuple[Any, ...] | list[tuple[Any, ...]] | None = None,
    **extra_context: Any,
) -> FeatureSet:
    """Build a FeatureSet with optional partition_by, order_by, mask, and extra context.

    Any additional keyword arguments are merged into the same Options context dict
    used by the explicit ``partition_by`` / ``order_by`` / ``mask`` arguments,
    enabling callers to pass operation-specific keys (e.g. ``constant=5``) without
    constructing ``Feature``/``Options`` manually. The explicit keyword arguments
    take precedence over ``extra_context`` on key collision.
    """
    context: dict[str, Any] = dict(extra_context)
    if partition_by is not None:
        context["partition_by"] = partition_by
    if order_by is not None:
        context["order_by"] = order_by
    if mask is not None:
        context["mask"] = mask
    feature = Feature(feature_name, options=Options(context=context))
    fs = FeatureSet()
    fs.add(feature)
    return fs


def feature_set_for(feature_name: str, options: Options) -> FeatureSet:
    """Build a FeatureSet holding one feature that carries ``options`` verbatim.

    ``make_feature_set`` assembles the Options from keyword arguments; this one takes an
    Options that has already been assembled, which is the shape ``compute_values`` in the
    scalar-arity harness hands to a family.
    """
    fs = FeatureSet()
    fs.add(Feature(feature_name, options=options))
    return fs
