"""Tests for ``mixins/pinned_rows.py``: a declared rejection hook versus the default compare path."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from mloda.user import DataType

from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.mixins.pinned_rows import PinnedRowTestMixin
from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable
from mloda.testing.typed_data import TypedTestData

_ROW = PinningRow(
    case="c",
    feature="f",
    inputs={"a": (DataType.INT64, (1, 2))},
    expected={"f": (DataType.INT64, (10, 20))},
)
_PATTERN = "not supported"


class _StubBase(DataOpsTestBase):
    """Minimal concrete base: the run result is a column dict."""

    @classmethod
    def implementation_class(cls) -> Any:
        return object

    def create_test_data(self, data: TypedTestData, arrow_types: Mapping[str, Any] | None = None) -> Any:
        return data

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return list(result[column_name])

    def get_row_count(self, result: Any) -> int:
        return 0

    def get_expected_type(self) -> Any:
        return dict


class _Raises:
    @staticmethod
    def calculate_feature(data: Any, fs: Any) -> Any:
        raise ValueError("operation is not supported here")


class _RaisesOther:
    @staticmethod
    def calculate_feature(data: Any, fs: Any) -> Any:
        raise ValueError("boom")


class _Computes:
    @staticmethod
    def calculate_feature(data: Any, fs: Any) -> Any:
        return {"f": [10, 20]}


class _ComputesWrong:
    @staticmethod
    def calculate_feature(data: Any, fs: Any) -> Any:
        return {"f": [10, 21]}


def _make_host(impl: type, pattern: str | None, expected_values_raises: bool = False) -> Any:
    class Host(PinnedRowTestMixin, _StubBase):
        @classmethod
        def implementation_class(cls) -> Any:
            return impl

        @classmethod
        def pinning_table(cls) -> PinningTable:
            return PinningTable(operation="op", rules={}, rows=(_ROW,))

        def expected_values(self, row: PinningRow) -> Any:
            if expected_values_raises:
                raise AssertionError("expected_values must not run")
            return super().expected_values(row)

        def rejection_match(self, row: PinningRow) -> str | None:
            return pattern

    return Host()


_CASE_ID = f"0-{_ROW.case}-{_ROW.feature}"


class TestRejectionHook:
    def test_declared_rejection_that_raises_matching_value_error_passes(self) -> None:
        _make_host(_Raises, _PATTERN).test_mixin_pinned_row(_CASE_ID)

    def test_declared_rejection_that_computes_a_result_fails(self) -> None:
        with pytest.raises(pytest.fail.Exception):
            _make_host(_Computes, _PATTERN).test_mixin_pinned_row(_CASE_ID)

    def test_declared_rejection_with_non_matching_message_fails(self) -> None:
        with pytest.raises(AssertionError, match="did not match"):
            _make_host(_RaisesOther, _PATTERN).test_mixin_pinned_row(_CASE_ID)

    def test_no_declared_rejection_still_compares_values(self) -> None:
        _make_host(_Computes, None).test_mixin_pinned_row(_CASE_ID)
        with pytest.raises(AssertionError):
            _make_host(_ComputesWrong, None).test_mixin_pinned_row(_CASE_ID)

    def test_rejection_match_takes_precedence_over_expected_values(self) -> None:
        _make_host(_Raises, _PATTERN, expected_values_raises=True).test_mixin_pinned_row(_CASE_ID)
