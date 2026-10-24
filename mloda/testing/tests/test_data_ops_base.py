"""Tests for the shared ``DataOpsTestBase`` helpers."""

from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
import pytest

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


class _RecordingImpl:
    """Implementation/reference stand-in that records the Options context it receives."""

    seen: list[dict[str, Any]] = []

    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        cls.seen.append(dict(next(iter(features.features)).options.context))
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
    stub = _CompareStub()
    stub.setup_method()
    stub._compare_with_reference("f", partition_by=["g"], lookup_mapping={"a": "b"}, on_unmatched="null")
    expected = {"partition_by": ["g"], "lookup_mapping": {"a": "b"}, "on_unmatched": "null"}
    assert _RecordingImpl.seen == [expected, expected]


# -- calculate_with / feature_group_of / reference_calculate ------------------


class _FeatureGroupPlugin:
    """FeatureGroup stand-in whose calculate_feature tags the result."""

    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        return ("feature_group", data, features)


def test_calculate_with_calls_calculate_feature_on_a_feature_group_class() -> None:
    fs = helpers.make_feature_set("f")
    assert helpers.calculate_with(_FeatureGroupPlugin, "data", fs) == ("feature_group", "data", fs)


def test_feature_group_of_returns_a_feature_group_class_itself() -> None:
    assert helpers.feature_group_of(_FeatureGroupPlugin) is _FeatureGroupPlugin


def test_calculate_with_and_feature_group_of_for_an_implementation() -> None:
    import gc

    from mloda.provider import Implementation, OperationFeatureGroup, get_all_subclasses
    from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

    class _ProbeGroup(OperationFeatureGroup):
        pass

    class _ProbeImpl(Implementation[_ProbeGroup]):
        feature_group = _ProbeGroup
        compute_framework = PyArrowTable

        @classmethod
        def calculate(cls, group: Any, data: Any, features: Any, framework: Any) -> Any:
            return ("implementation", group, data)

    try:
        fs = helpers.make_feature_set("f")
        assert helpers.feature_group_of(_ProbeImpl) is _ProbeGroup
        assert helpers.calculate_with(_ProbeImpl, "data", fs) == ("implementation", _ProbeGroup, "data")
        assert fs.implementation is _ProbeImpl
    finally:
        del fs, _ProbeImpl, _ProbeGroup
        gc.collect()
    assert not [c for c in get_all_subclasses(Implementation) if c.__name__ == "_ProbeImpl"]


def test_calculate_with_runs_an_implementation_on_the_passed_group() -> None:
    import gc

    from mloda.provider import Implementation, OperationFeatureGroup, get_all_subclasses
    from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

    class _ProbeGroup(OperationFeatureGroup):
        pass

    class _ProbeSubGroup(_ProbeGroup):
        pass

    class _ProbeImpl(Implementation[_ProbeGroup]):
        feature_group = _ProbeGroup
        compute_framework = PyArrowTable

        @classmethod
        def calculate(cls, group: Any, data: Any, features: Any, framework: Any) -> Any:
            return ("implementation", group, data)

    try:
        fs = helpers.make_feature_set("f")
        assert helpers.calculate_with(_ProbeImpl, "data", fs, group=_ProbeSubGroup) == (
            "implementation",
            _ProbeSubGroup,
            "data",
        )
        assert fs.implementation is _ProbeImpl
    finally:
        del fs, _ProbeImpl, _ProbeSubGroup, _ProbeGroup
        gc.collect()
    assert not [c for c in get_all_subclasses(Implementation) if c.__name__ == "_ProbeImpl"]


def test_reference_calculate_defaults_to_the_reference_class_calculate_feature() -> None:
    _RecordingImpl.seen = []
    table = pa.table({"f": [1]})
    fs = helpers.make_feature_set("f", partition_by=["g"])
    result = _CompareStub.reference_calculate(table, fs)
    assert result.column("f").to_pylist() == [1]
    assert _RecordingImpl.seen == [{"partition_by": ["g"]}]


def _reference_calculate_with_an_implementation_reference() -> Any:
    from mloda.provider import Implementation, OperationFeatureGroup
    from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

    class _RefGroup(OperationFeatureGroup):
        @classmethod
        def calculate_feature(cls, data: Any, features: Any) -> Any:
            return ("group", data, features.implementation)

    class _RefImpl(Implementation[_RefGroup]):
        feature_group = _RefGroup
        compute_framework = PyArrowTable

    class _ImplRefStub(_StubDataOpsTestBase):
        @classmethod
        def implementation_class(cls) -> Any:
            return _RefImpl

        @classmethod
        def reference_implementation_class(cls) -> Any:
            return _RefImpl

    result = _ImplRefStub.reference_calculate("data", helpers.make_feature_set("f"))
    return result, _RefImpl.__name__


def test_reference_calculate_runs_an_implementation_reference_through_its_feature_group() -> None:
    (tag, data, implementation), name = _reference_calculate_with_an_implementation_reference()
    assert (tag, data) == ("group", "data")
    assert implementation.__name__ == name
