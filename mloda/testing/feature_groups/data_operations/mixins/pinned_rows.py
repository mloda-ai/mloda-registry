"""Run a group's hand-computed pinning rows through its backend under test."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import pytest
from mloda.provider import Implementation
from mloda.user import DataType, Options

from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import (
    assert_values_with_nulls,
    calculate_with,
    make_feature_set,
)
from mloda.testing.feature_groups.data_operations.mixins.case_parametrization import CaseParametrizationTestMixin
from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable
from mloda.testing.typed_data import TypedTestData


def _instant_and_offset(value: Any) -> Any:
    """Normalize a datetime-like value to (datetime, utcoffset) so instant and zone are both compared."""
    if value is None:
        return None
    as_datetime = value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
    return (as_datetime, value.utcoffset())


def _lossless_int(value: Any) -> int:
    """Pandas nullable results and DuckDB HUGEINT/Decimal may represent an integer differently."""
    assert not isinstance(value, bool), "a boolean is not an integer result"
    converted = int(value)
    assert converted == value, f"integer conversion would change {value!r}"
    return converted


def _integer_boolean(value: Any) -> bool:
    """Backends storing booleans as 0/1; reject every other integer instead of truthiness-casting it."""
    assert isinstance(value, (bool, int)) and value in (0, 1), f"invalid integer boolean {value!r}"
    return bool(value)


_INTEGER_TYPES = (DataType.INT8, DataType.INT32, DataType.INT64, DataType.UINT8, DataType.UINT64)


class PinnedRowTestMixin(DataOpsTestBase, CaseParametrizationTestMixin):
    """Share collection and execution; hosts supply the table and any declared divergent expectations."""

    _case_fixtures: ClassVar[dict[str, str]] = {"pinned_row_case": "pinned_row_cases"}

    @classmethod
    def pinning_table(cls) -> PinningTable:
        """Hand-computed rows of the group under test. Required."""
        raise NotImplementedError("Subclass must override pinning_table")

    @classmethod
    def pinned_divergences(cls) -> Mapping[str, Mapping[str, list[Any]]]:
        """Row id -> column -> the backend's own reproduced values, replacing the hand-computed column."""
        return {}

    @classmethod
    def booleans_as_integers(cls) -> bool:
        """True when the backend returns booleans as 0/1 integers."""
        return False

    @classmethod
    def pinned_row_cases(cls) -> dict[str, PinningRow]:
        """Rows keyed ``{index}-{case}-{feature}``; an Implementation's unsupported or refused rows are dropped."""
        impl = cls.implementation_class()
        checked = isinstance(impl, type) and issubclass(impl, Implementation)
        cases: dict[str, PinningRow] = {}
        for i, row in enumerate(cls.pinning_table().rows):
            if checked:
                options = Options(context=dict(row.context))
                subtype = impl.canonical_subtype(impl.feature_group, row.feature, options)
                if impl.supported is not None and subtype not in impl.supported:
                    continue
                if not impl.accepts(row.feature, options):
                    continue
            cases[f"{i}-{row.case}-{row.feature}"] = row
        return cases

    def _row_id(self, row: PinningRow) -> str:
        return next(row_id for row_id, candidate in self.pinned_row_cases().items() if candidate is row)

    def expected_values(self, row: PinningRow) -> dict[str, list[Any]]:
        """Expected columns with pinned divergences overlaid; override to skip unsupported operations."""
        expected = {column: list(values) for column, (_, values) in row.expected.items()}
        for column, values in self.pinned_divergences().get(self._row_id(row), {}).items():
            expected[column] = list(values)
        return expected

    def rejection_match(self, row: PinningRow) -> str | None:
        """Regex of the up-front error a declared rejection raises for this row; None computes it."""
        return None

    def rejection_type(self, row: PinningRow) -> type[Exception]:
        """Exception class a declared rejection raises."""
        return ValueError

    def assert_pinned_values(self, row: PinningRow, column: str, actual: list[Any], expected: list[Any]) -> None:
        """Compare one column strictly; timestamps by instant and offset, integers losslessly."""
        data_type = row.expected[column][0]
        if data_type.name.startswith("TIMESTAMP"):
            assert_values_with_nulls(
                actual,
                [_instant_and_offset(v) for v in expected],
                strict=True,
                parse_iso=True,
                cast=_instant_and_offset,
            )
        elif data_type in _INTEGER_TYPES:
            assert_values_with_nulls(actual, expected, strict=True, cast=_lossless_int)
        elif data_type is DataType.BOOLEAN and self.booleans_as_integers():
            assert_values_with_nulls(actual, expected, strict=True, cast=_integer_boolean)
        else:
            assert_values_with_nulls(actual, expected, strict=True)

    def test_mixin_pinned_divergences_name_rows(self) -> None:
        """Every pinned divergence names an existing row and one of its expected columns."""
        rows = self.pinned_row_cases()
        for row_id, columns in self.pinned_divergences().items():
            assert row_id in rows, f"stale pinned divergence row {row_id!r}"
            for column in columns:
                assert column in rows[row_id].expected, f"stale pinned divergence column {row_id!r}/{column!r}"

    def _create_pinned_data(self, row: PinningRow) -> Any:
        float_inputs = [name for name, (data_type, _) in row.inputs.items() if data_type == DataType.FLOAT]
        if not float_inputs:
            return self.create_test_data(TypedTestData(row.inputs))
        pytest.importorskip("pyarrow")
        # A Python dict cannot hold float32, so Arrow gives it the float32-rounded values a float32 column holds.
        arrow_types = {name: DataType.to_arrow_type(DataType.FLOAT) for name in float_inputs}
        return self.create_test_data(TypedTestData(row.inputs), arrow_types)

    def test_mixin_pinned_row(self, pinned_row_case: str) -> None:
        """A row matches the backend result; rejection_match takes precedence over expected_values."""
        row = self.pinned_row_cases()[pinned_row_case]
        fs = make_feature_set(row.feature, **row.context)
        pattern = self.rejection_match(row)
        if pattern is not None:
            data = self._create_pinned_data(row)
            with pytest.raises(self.rejection_type(row), match=pattern):
                calculate_with(self.implementation_class(), data, fs)
            return
        expected = self.expected_values(row)
        result = calculate_with(self.implementation_class(), self._create_pinned_data(row), fs)
        for column, values in expected.items():
            self.assert_pinned_values(row, column, self.extract_column(result, column), values)
        adapter = getattr(self, "adapter", None)
        if adapter is not None:
            expected_columns = set(row.expected)
            if self.pinning_table().row_preserving:
                expected_columns |= set(row.inputs)
            assert set(adapter.column_names(result)) == expected_columns
