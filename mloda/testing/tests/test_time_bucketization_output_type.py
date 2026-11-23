"""The bucketization suite must detect unit drift even when every timestamp value matches."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pyarrow as pa
import pytest

from mloda.testing.feature_groups.data_operations.helpers import arrow_table_of
from mloda.testing.feature_groups.data_operations.mixins.frameworks import AdapterTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.time_bucketization.time_bucketization import (
    TimeBucketizationTestBase,
)
from mloda.testing.framework_adapter import (
    FrameworkAdapter,
    PandasAdapter,
    PolarsLazyAdapter,
    PyArrowAdapter,
    PythonDictAdapter,
    SqliteAdapter,
)
from mloda.testing.typed_data import TypedTestData

_CASES = {
    name: row
    for name, row in TimeBucketizationTestBase.pinned_row_cases().items()
    if row.case in {"unit_kept", "empty"}
}
_NATIVE_ADAPTERS = [PyArrowAdapter, PandasAdapter, PolarsLazyAdapter]


def _result_table(case: str) -> pa.Table:
    row = _CASES[case]
    return arrow_table_of(TypedTestData({**row.inputs, **row.expected}))


@contextmanager
def _fixture_suite(table: pa.Table, adapter_type: type[FrameworkAdapter]) -> Iterator[TimeBucketizationTestBase]:
    """A fixture backend returns native data with known values and a deliberately chosen physical type."""
    output: Any = None

    class FixtureBackend:
        @classmethod
        def calculate_feature(cls, data: Any, features: Any) -> Any:
            return output

    class Suite(AdapterTestMixin, TimeBucketizationTestBase):
        adapter_class = adapter_type

        @classmethod
        def implementation_class(cls) -> Any:
            return FixtureBackend

        def get_expected_type(self) -> Any:
            return type(output)

    if not adapter_type().is_available():
        pytest.skip(f"{adapter_type.__name__} is not installed")
    suite = Suite()
    suite.setup_method()
    try:
        output = suite.adapter.from_arrow(table)
        yield suite
    finally:
        suite.teardown_method()


@pytest.mark.parametrize("adapter_type", _NATIVE_ADAPTERS)
@pytest.mark.parametrize("case", _CASES)
def test_pinned_row_accepts_the_declared_native_unit(case: str, adapter_type: type[FrameworkAdapter]) -> None:
    with _fixture_suite(_result_table(case), adapter_type) as suite:
        suite.test_pinned_row(case)


@pytest.mark.parametrize("adapter_type", _NATIVE_ADAPTERS)
@pytest.mark.parametrize("case", _CASES)
def test_pinned_row_detects_changed_native_unit_with_identical_values(
    case: str, adapter_type: type[FrameworkAdapter]
) -> None:
    table = _result_table(case)
    feature = _CASES[case].feature
    original = table.column(feature)
    changed_unit = "us" if original.type.unit == "ms" else "ns"
    changed = original.cast(pa.timestamp(changed_unit, tz=original.type.tz))
    assert changed.to_pylist() == original.to_pylist()
    table = table.set_column(table.schema.get_field_index(feature), feature, changed)

    with _fixture_suite(table, adapter_type) as suite:
        with pytest.raises(AssertionError):
            suite.test_pinned_row(case)


@pytest.mark.parametrize("adapter_type", [PythonDictAdapter, SqliteAdapter])
def test_pinned_row_keeps_value_checks_without_an_exact_timestamp_type(adapter_type: type[FrameworkAdapter]) -> None:
    case = next(name for name, row in _CASES.items() if row.case == "unit_kept")
    row = _CASES[case]
    table = _result_table(case)
    values = row.expected[row.feature][1]
    if adapter_type is SqliteAdapter:
        # SQLite bucketization returns ISO strings, as its public output contract specifies.
        strings = pa.array([None if value is None else value.isoformat() for value in values], type=pa.string())
        table = table.set_column(table.schema.get_field_index(row.feature), row.feature, strings)
    with _fixture_suite(table, adapter_type) as suite:
        suite.test_pinned_row(case)

    column = table.column(row.feature)
    wrong_values = pa.array([None] * len(values), type=column.type)
    wrong = table.set_column(table.schema.get_field_index(row.feature), row.feature, wrong_values)
    with _fixture_suite(wrong, adapter_type) as suite:
        with pytest.raises(AssertionError):
            suite.test_pinned_row(case)
