"""Tests for the shared ``DataOpsTestBase`` helpers."""

from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
import pytest

from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations import helpers
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.mixins import mask as mask_module
from mloda.testing.feature_groups.data_operations.mixins import mask_integration as mask_integration_module
from mloda.testing.feature_groups.data_operations.row_changing.resample.resample import ResampleTestBase
from mloda.testing.feature_groups.data_operations.row_preserving.ema.ema import EmaTestBase
from mloda.testing.feature_groups.data_operations.row_preserving.ffill.ffill import FfillTestBase
from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate import (
    frame_aggregate as frame_aggregate_module,
)
from mloda.testing.feature_groups.data_operations.row_preserving.rank.rank import RankTestBase
from mloda.testing.feature_groups.data_operations.row_preserving.sessionization.sessionization import (
    SessionizationTestBase,
)
from mloda.testing.feature_groups.data_operations.row_preserving.time_bucketization.time_bucketization import (
    TimeBucketizationTestBase,
)


class _StubDataOpsTestBase(DataOpsTestBase):
    """Minimal concrete base: the 5 adapter methods, no capability declaration."""

    @classmethod
    def implementation_class(cls) -> Any:
        return object

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return arrow_table

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return []

    def get_row_count(self, result: Any) -> int:
        return 0

    def get_expected_type(self) -> Any:
        return float


class _DeclaresNothing(_StubDataOpsTestBase):
    """A base that declares none of the probed ``supported_*`` methods."""


class _DeclaresAggTypes(_StubDataOpsTestBase):
    """A base that declares the aggregation vocabulary."""

    @classmethod
    def supported_agg_types(cls) -> set[str]:
        return {"sum", "mean"}


def test_skip_if_unsupported_raises_when_no_capability_is_declared() -> None:
    """A base declaring no ``supported_*`` method must fail, not skip.

    Skipping here would let a shared test pass on every framework without ever
    running, which is the silent coverage loss this guards against.
    """
    with pytest.raises(TypeError) as excinfo:
        _DeclaresNothing()._skip_if_unsupported("sum")

    message = str(excinfo.value)
    assert "_DeclaresNothing" in message
    for attr in ("supported_agg_types", "supported_ops", "supported_offset_types", "supported_rank_types"):
        assert attr in message


def test_skip_if_unsupported_still_skips_an_undeclared_op() -> None:
    """A base that declares a supported set keeps skipping ops outside it."""
    with pytest.raises(pytest.skip.Exception, match="median not supported by this framework"):
        _DeclaresAggTypes()._skip_if_unsupported("median")


def test_skip_if_unsupported_returns_for_a_declared_op() -> None:
    """An op inside the declared set is neither skipped nor an error."""
    _DeclaresAggTypes()._skip_if_unsupported("sum")


# -- helpers.is_null --------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, True),
        (float("nan"), True),
        (0.0, False),
        ("", False),
        (1, False),
    ],
)
def test_is_null(value: Any, expected: bool) -> None:
    assert helpers.is_null(value) is expected


# -- helpers.assert_values_with_nulls ---------------------------------------


def test_assert_values_with_nulls_length_mismatch() -> None:
    with pytest.raises(AssertionError, match=r"row count 1 != expected 2"):
        helpers.assert_values_with_nulls([1], [1, 2])


def test_assert_values_with_nulls_none_vs_none_passes() -> None:
    helpers.assert_values_with_nulls([None], [None])


def test_assert_values_with_nulls_none_expected_value_actual_fails() -> None:
    with pytest.raises(AssertionError, match=r"row 0: expected None, got 1\.0"):
        helpers.assert_values_with_nulls([1.0], [None])


def test_assert_values_with_nulls_value_expected_none_actual_fails() -> None:
    with pytest.raises(AssertionError, match=r"row 0: expected 1\.0, got None"):
        helpers.assert_values_with_nulls([None], [1.0])


def test_assert_values_with_nulls_exact_by_default_fails_on_float_noise() -> None:
    with pytest.raises(AssertionError):
        helpers.assert_values_with_nulls([1.0000001], [1.0])


def test_assert_values_with_nulls_approx_true_passes_with_rel() -> None:
    helpers.assert_values_with_nulls([1.0000001], [1.0], approx=True)


def test_assert_values_with_nulls_nan_is_null_true_both_directions() -> None:
    helpers.assert_values_with_nulls([float("nan")], [None], nan_is_null=True)
    helpers.assert_values_with_nulls([None], [float("nan")], nan_is_null=True)


def test_assert_values_with_nulls_without_nan_is_null_expected_none_actual_nan_fails() -> None:
    with pytest.raises(AssertionError):
        helpers.assert_values_with_nulls([float("nan")], [None])


def test_assert_values_with_nulls_parse_iso_accepts_iso_string_for_expected_datetime() -> None:
    expected = datetime(2024, 1, 1, tzinfo=timezone.utc)
    helpers.assert_values_with_nulls(["2024-01-01T00:00:00+00:00"], [expected], parse_iso=True)


def test_assert_values_with_nulls_cast_int_turns_float_into_int_exact() -> None:
    helpers.assert_values_with_nulls([1.0], [1], cast=int)


def test_assert_values_with_nulls_cast_float_with_approx() -> None:
    helpers.assert_values_with_nulls([1], [1.0000001], cast=float, approx=True)


# -- DataOpsTestBase.source_arrow_table --------------------------------------


def test_source_arrow_table_default_returns_canonical_table() -> None:
    table = _StubDataOpsTestBase.source_arrow_table()
    assert table.equals(PyArrowDataOpsTestDataCreator.create())


def test_setup_method_uses_source_arrow_table_default() -> None:
    stub = _StubDataOpsTestBase()
    stub.setup_method()
    assert stub._arrow_table.equals(PyArrowDataOpsTestDataCreator.create())


class _StubWithCustomSourceTable(_StubDataOpsTestBase):
    """A stub overriding ``source_arrow_table`` with a non-canonical table."""

    @classmethod
    def source_arrow_table(cls) -> pa.Table:
        return pa.table({"custom": [1, 2, 3]})


def test_setup_method_uses_overridden_source_arrow_table() -> None:
    stub = _StubWithCustomSourceTable()
    stub.setup_method()
    assert stub._arrow_table.column_names == ["custom"]
    assert stub.test_data.column_names == ["custom"]


# -- Structural guards: setup_method deleted, source_arrow_table declared ---


@pytest.mark.parametrize(
    "cls",
    [EmaTestBase, FfillTestBase, ResampleTestBase, SessionizationTestBase, TimeBucketizationTestBase],
)
def test_operation_base_has_no_setup_method_override_and_declares_source_arrow_table(cls: type) -> None:
    assert "setup_method" not in cls.__dict__
    assert "source_arrow_table" in cls.__dict__


def test_rank_test_base_no_longer_overrides_skip_if_unsupported() -> None:
    assert "_skip_if_unsupported" not in RankTestBase.__dict__


# -- Structural guards: old private helper duplicates removed ---------------


def test_ema_private_helper_removed() -> None:
    assert "_assert_float_list_with_nulls" not in EmaTestBase.__dict__


def test_ffill_private_helper_removed() -> None:
    assert "_assert_float_list_with_nulls" not in FfillTestBase.__dict__


def test_sessionization_private_helper_removed() -> None:
    assert "_assert_int_list" not in SessionizationTestBase.__dict__


def test_time_bucketization_private_helper_removed() -> None:
    assert "_assert_equal_with_nulls" not in TimeBucketizationTestBase.__dict__


def test_frame_aggregate_module_helpers_removed() -> None:
    assert not hasattr(frame_aggregate_module, "_assert_values_with_nulls")
    assert not hasattr(frame_aggregate_module, "_is_null")


def test_mask_module_helper_removed() -> None:
    assert not hasattr(mask_module, "_is_null")


def test_mask_integration_module_helper_removed() -> None:
    assert not hasattr(mask_integration_module, "_is_null")
