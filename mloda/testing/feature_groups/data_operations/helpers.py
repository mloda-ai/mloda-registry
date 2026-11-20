"""Shared helpers for data-operations tests.

Provides:
- ``extract_column``: Extract a column from any framework result as a Python list.
- ``result_column_names``: The column names of any framework result.
- ``make_feature_set``: Build a FeatureSet with optional partition_by/order_by.
- ``make_multi_feature_set``: Build a FeatureSet holding one Feature per (name, context) pair.
- ``feature_set_for``: Build a FeatureSet around an Options that already exists.
- ``is_null``: True for None or a float NaN.
- ``sort_none_last``: Sort values with None ordered last.
- ``result_column_values``: One value list per result frame (pyarrow or pandas) holding a column.
- ``assert_values_with_nulls``: Assert two lists match, null-aware, with optional cast/approx.
- ``calculate_with``: Run a FeatureGroup class or an Implementation class on data with a FeatureSet
  (``group=`` runs a subclass of the Implementation's group, e.g. a hook).
- ``feature_group_of``: The FeatureGroup class of a FeatureGroup class or an Implementation class.
- ``load_implementation``: A named Implementation of a family package; skips when its optional backend is missing.
- ``empty_like``: Typed test data with the same columns and types and zero rows.
- ``canonical_data_empty``: The canonical typed data with zero rows, for empty-input cases.
- ``arrow_table_of``: Typed test data as an Arrow table; skips under pytest without pyarrow.
- ``append_typed_columns``: Typed test data plus extra typed columns.
- ``masked_column``: A table column with rows failing any mask condition set to null.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Callable

import pyarrow as pa
import pytest
from mloda.provider import FeatureSet, Implementation
from mloda.user import DataType, Feature, Options

from mloda.testing.data_creator.base import DataOperationsTestDataCreator
from mloda.testing.framework_adapter import to_arrow_table
from mloda.testing.import_isolation import require_pyarrow
from mloda.testing.typed_data import TypedTestData


def feature_group_of(plugin: Any) -> Any:
    """The FeatureGroup class of a plugin: ``plugin.feature_group`` for an Implementation, else the plugin."""
    if isinstance(plugin, type) and issubclass(plugin, Implementation):
        return plugin.feature_group
    return plugin


def load_implementation(package: str, module: str, name: str) -> Any:
    """The Implementation ``name`` in ``<package>.<module>``; skips when its optional backend is not installed."""
    return getattr(pytest.importorskip(f"{package}.{module}"), name)


def calculate_with(plugin: Any, data: Any, features: FeatureSet, *, group: Any = None) -> Any:
    """Run a FeatureGroup class or an Implementation class on data with a FeatureSet.

    For an Implementation, ``group`` (default ``plugin.feature_group``) is the group whose loop runs.
    """
    if isinstance(plugin, type) and issubclass(plugin, Implementation):
        features.implementation = plugin
        return (group or plugin.feature_group).calculate_feature(data, features)
    return plugin.calculate_feature(data, features)


_MASK_OPERATORS = frozenset({"equal", "greater_than", "greater_equal", "less_equal", "less_than", "is_in"})


def masked_column(table: pa.Table, source_col: str, mask_spec: list[tuple[str, str, Any]]) -> Any:
    """The ``source_col`` array with rows failing any ``(column, operator, value)`` condition set to null.

    The mask compares Python values, so mismatched operand types give False rather than an Arrow error.
    """
    import pyarrow.compute as pc
    from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_mask_engine import (
        PythonDictMaskEngine,
    )

    data = table.to_pydict()
    mask = PythonDictMaskEngine.all_true(data)
    for column, operator, value in mask_spec:
        if operator not in _MASK_OPERATORS:
            raise ValueError(f"Unsupported mask operator '{operator}'. Supported: {sorted(_MASK_OPERATORS)}")
        if column not in data:
            raise KeyError(column)
        mask = PythonDictMaskEngine.combine(mask, getattr(PythonDictMaskEngine, operator)(data, column, value))
    null_scalar = pa.scalar(None, type=table.schema.field(source_col).type)
    return pc.if_else(pa.array(mask, type=pa.bool_()), table.column(source_col), null_scalar)


def arrow_table_of(data: TypedTestData, arrow_types: Mapping[str, pa.DataType] | None = None) -> pa.Table:
    """The typed data as an Arrow table; skips under pytest when pyarrow is missing."""
    require_pyarrow()
    return to_arrow_table(data, arrow_types)


def append_typed_columns(data: TypedTestData, columns: Mapping[str, tuple[DataType, Sequence[Any]]]) -> TypedTestData:
    """The data plus extra ``name -> (type, values)`` columns; raises ValueError for an existing name."""
    duplicate = [name for name in columns if name in data.types]
    if duplicate:
        raise ValueError(f"Column '{duplicate[0]}' already exists")
    values = data.to_python_dict()
    base = {name: (data_type, values[name]) for name, data_type in data.types.items()}
    return TypedTestData({**base, **columns})


def typed_data_without(data: TypedTestData, *columns: str) -> TypedTestData:
    """The typed data minus the named columns; raises KeyError for an unknown column."""
    unknown = [column for column in columns if column not in data.types]
    if unknown:
        raise KeyError(unknown[0])
    values = data.to_python_dict()
    return TypedTestData({name: (kind, values[name]) for name, kind in data.types.items() if name not in columns})


def empty_like(data: TypedTestData) -> TypedTestData:
    """The data's columns and types with zero rows."""
    return TypedTestData({name: (data_type, []) for name, data_type in data.types.items()})


def canonical_data_empty() -> TypedTestData:
    """The canonical typed data's columns with zero rows."""
    return empty_like(DataOperationsTestDataCreator.typed_data())


def result_column_names(result: Any) -> list[str]:
    """The column names of a framework result, mirroring ``extract_column``'s dispatch."""
    if isinstance(result, dict):
        return list(result.keys())
    if isinstance(result, pa.Table):
        return list(result.column_names)
    if hasattr(result, "to_arrow_table"):
        return list(result.to_arrow_table().column_names)
    if hasattr(result, "collect"):
        return list(result.collect().columns)
    return list(result.columns)


def is_null(value: Any) -> bool:
    """True for None or a float NaN."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return False


def sort_none_last(values: list[Any]) -> list[Any]:
    """Sort values with ``None`` ordered last."""
    return sorted(values, key=lambda x: (x is None, x))


def result_column_values(results: list[Any], name: str) -> list[list[Any]]:
    """One value list per result frame (pyarrow or pandas) holding ``name``, NaN as None."""
    found = []
    for frame in results:
        if isinstance(frame, pa.Table):
            if name in frame.column_names:
                found.append(frame.column(name).to_pylist())
        elif hasattr(frame, "columns") and name in frame.columns:
            found.append([None if is_null(v) else v for v in frame[name].tolist()])
    return found


def _assert_strict_equal(actual: Any, expected: Any, row: int, rel: float) -> None:
    """One value of ``assert_values_with_nulls(strict=True)``, recursing into lists, tuples and dicts."""
    assert type(actual) is type(expected), f"row {row}: {actual!r} is not the type of {expected!r}"
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), f"row {row}: length {len(actual)} != {len(expected)}"
        for a, e in zip(actual, expected):
            _assert_strict_equal(a, e, row, rel)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected), (
            f"row {row}: keys {sorted(actual, key=repr)} != {sorted(expected, key=repr)}"
        )
        for key, e in expected.items():
            _assert_strict_equal(actual[key], e, row, rel)
    elif isinstance(expected, float):
        if math.isnan(expected):
            assert math.isnan(actual), f"row {row}: {actual!r} != NaN"
            return
        assert actual == pytest.approx(expected, rel=rel, abs=0), f"row {row}: {actual!r} != {expected!r}"
        assert math.copysign(1.0, actual) == math.copysign(1.0, expected), f"row {row}: zero sign of {actual!r}"
    else:
        assert actual == expected, f"row {row}: {actual!r} != {expected!r}"


def assert_values_with_nulls(
    actual: list[Any],
    expected: list[Any],
    *,
    approx: bool = False,
    rel: float = 1e-6,
    nan_is_null: bool = False,
    parse_iso: bool = False,
    cast: Callable[[Any], Any] | None = None,
    strict: bool = False,
) -> None:
    """Assert two lists are equal row by row, treating null consistently.

    ``nan_is_null`` treats NaN and None as interchangeable nulls, ``parse_iso`` parses a str
    actual value via ``datetime.fromisoformat``, ``cast`` converts the actual value before
    comparing, and ``approx`` compares with ``pytest.approx`` instead of exact equality. ``strict``
    compares floats with ``pytest.approx(rel)`` and the same sign, everything else exact and type-equal
    (``5`` is not ``5.0``); NaN matches only NaN. It recurses into lists, tuples and dicts, and
    cannot be combined with ``nan_is_null``.
    """
    if strict and nan_is_null:
        raise ValueError("strict=True and nan_is_null=True contradict each other")
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
            if strict:
                _assert_strict_equal(a, e, i, rel)
            elif approx:
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
    return make_multi_feature_set([(feature_name, context)])


def make_multi_feature_set(specs: list[tuple[str, dict[str, Any]]]) -> FeatureSet:
    """Build a FeatureSet holding one Feature per (name, context) pair, each with its own Options context."""
    fs = FeatureSet()
    for name, context in specs:
        fs.add(Feature(name, options=Options(context=dict(context))))
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
