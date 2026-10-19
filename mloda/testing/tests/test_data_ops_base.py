"""Tests for the shared ``DataOpsTestBase`` helpers."""

from datetime import datetime, timezone
from typing import Any, ClassVar

import pyarrow as pa
import pytest
from mloda.provider import FeatureGroup, property_spec
from mloda.user import Options

from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations import helpers
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.row_changing.resample.resample import ResampleTestBase
from mloda.testing.feature_groups.data_operations.row_preserving.ema.ema import EmaTestBase
from mloda.testing.feature_groups.data_operations.row_preserving.ffill.ffill import FfillTestBase
from mloda.testing.feature_groups.data_operations.row_preserving.lookup.lookup import LookupTestBase
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
    [EmaTestBase, FfillTestBase, LookupTestBase, ResampleTestBase, SessionizationTestBase, TimeBucketizationTestBase],
)
def test_operation_base_has_no_setup_method_override_and_declares_source_arrow_table(cls: type) -> None:
    assert "setup_method" not in cls.__dict__
    assert "source_arrow_table" in cls.__dict__


# -- DataOpsTestBase._compare_with_reference extra context ------------------


class _RecordingImpl(FeatureGroup):
    """Implementation/reference stand-in that records the Options it receives."""

    PROPERTY_MAPPING: ClassVar[dict[str, Any]] = {"factor": property_spec("Scale factor", default=2)}

    seen: list[dict[str, Any]] = []
    seen_factor: list[Any] = []

    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        options = next(iter(features.features)).options
        cls.seen.append({k: v for k, v in options.context.items() if k != "factor"})
        cls.seen_factor.append(options.get("factor"))
        return pa.table({"f": [1]})


class _CompareStub(_StubDataOpsTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return _RecordingImpl

    @classmethod
    def reference_implementation_class(cls) -> Any:
        return _RecordingImpl

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return list(result.column(column_name).to_pylist())


def test_compare_with_reference_forwards_extra_context_to_both_calls() -> None:
    _RecordingImpl.seen = []
    _RecordingImpl.seen_factor = []
    stub = _CompareStub()
    stub.setup_method()
    stub._compare_with_reference("f", partition_by=["g"], lookup_mapping={"a": "b"}, on_unmatched="null")
    expected = {"partition_by": ["g"], "lookup_mapping": {"a": "b"}, "on_unmatched": "null"}
    assert _RecordingImpl.seen == [expected, expected]
    assert _RecordingImpl.seen_factor == [2, 2]


# -- helpers feature_group keyword -------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda **kwargs: helpers.make_feature_set("f", **kwargs),
        lambda **kwargs: helpers.feature_set_for("f", Options(), **kwargs),
    ],
    ids=["make_feature_set", "feature_set_for"],
)
def test_feature_set_builder_with_feature_group_carries_its_default(build: Any) -> None:
    with_group = build(feature_group=_RecordingImpl)
    assert next(iter(with_group.features)).options.get("factor") == 2
    without = build()
    assert next(iter(without.features)).options.get("factor") is None


# -- DataOpsTestBase contract hooks -------------------------------------------


class _ContractStub(_StubDataOpsTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return _RecordingImpl

    @classmethod
    def compute_framework_class(cls) -> Any:
        return _RecordingFramework

    def output_contract_feature_set(self) -> Any:
        return helpers.feature_set_for("f", Options(context={"k": "v"}))


class _RecordingFramework:
    """Stand-in for a ComputeFramework type."""


def test_contract_hooks_return_implementation_and_framework() -> None:
    stub = _ContractStub()
    assert stub.contract_feature_group_class() is _RecordingImpl
    assert stub.contract_expected_compute_frameworks() == {_RecordingFramework}


def test_contract_matching_feature_names_returns_the_output_contract_feature() -> None:
    (feature,) = _ContractStub().contract_matching_feature_names()
    assert str(feature.name) == "f"
    assert feature.options.get("k") == "v"


@pytest.mark.parametrize("call", ["compute_framework_class", "contract_expected_compute_frameworks"])
def test_contract_frameworks_fail_without_a_framework_mixin(call: str) -> None:
    with pytest.raises(pytest.fail.Exception, match="contract_expected_compute_frameworks"):
        getattr(_StubDataOpsTestBase(), call)()
