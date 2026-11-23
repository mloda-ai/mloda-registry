"""Tests for the shared ``DataOpsTestBase`` helpers."""

import gc
import importlib
import importlib.util
import sqlite3
import sys
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from mloda.user import DataType

from mloda.testing.data_creator.base import DataOperationsTestDataCreator
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.data_creator.sqlite import SqliteDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations import helpers
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.operation_group import ImplSpec, OperationGroupTestBase
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
from mloda.testing.framework_adapter import to_arrow_table
from mloda.testing.tests.test_data_ops_integration_compat import _ToyIntegration
from mloda.testing.typed_data import TypedTestData


class _StubDataOpsTestBase(DataOpsTestBase):
    """Minimal concrete base: the 5 adapter methods, no capability declaration."""

    @classmethod
    def implementation_class(cls) -> Any:
        return object

    def create_test_data(self, data: TypedTestData, arrow_types: Mapping[str, pa.DataType] | None = None) -> Any:
        return data

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return []

    def get_row_count(self, result: Any) -> int:
        return 0

    def get_expected_type(self) -> Any:
        return float


def test_compute_framework_class_without_an_adapter_raises_naming_the_hook() -> None:
    with pytest.raises(NotImplementedError, match="compute_framework_class"):
        _StubDataOpsTestBase().compute_framework_class()


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


def test_assert_values_with_nulls_strict_passes_exact_values() -> None:
    helpers.assert_values_with_nulls(
        [1, 2.5, float("nan"), None, True, "x"], [1, 2.5, float("nan"), None, True, "x"], strict=True
    )


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(5.0, 5, id="float_is_not_int"),
        pytest.param(5, 5.0, id="int_is_not_float"),
        pytest.param(1, True, id="int_is_not_bool"),
        pytest.param(True, 1, id="bool_is_not_int"),
        pytest.param(None, float("nan"), id="none_is_not_nan"),
        pytest.param(float("nan"), None, id="nan_is_not_none"),
        pytest.param(1.0, float("nan"), id="value_is_not_nan"),
        pytest.param(float("nan"), 1.0, id="nan_is_not_value"),
        pytest.param(-0.0, 0.0, id="negative_zero_is_not_zero"),
        pytest.param(0.0, -0.0, id="zero_is_not_negative_zero"),
        pytest.param(1.5, 2.5, id="different_floats"),
        pytest.param(1e-13, 0.0, id="tiny_is_not_zero"),
        pytest.param(5e-324, 0.0, id="subnormal_is_not_zero"),
    ],
)
def test_assert_values_with_nulls_strict_rejects(actual: Any, expected: Any) -> None:
    with pytest.raises(AssertionError):
        helpers.assert_values_with_nulls([actual], [expected], strict=True)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param([1.0, float("nan"), None], [1.0, float("nan"), None], id="nested_list_with_nan"),
        pytest.param((1, 2), (1, 2), id="tuple"),
        pytest.param({"a": 1.0, "b": float("nan")}, {"a": 1.0, "b": float("nan")}, id="dict_with_nan"),
        pytest.param({"a": [1, {"b": None}]}, {"a": [1, {"b": None}]}, id="nested_dict_in_list"),
    ],
)
def test_assert_values_with_nulls_strict_recurses_and_passes(actual: Any, expected: Any) -> None:
    helpers.assert_values_with_nulls([actual], [expected], strict=True)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param([5.0], [5], id="nested_float_is_not_int"),
        pytest.param([True], [1], id="nested_bool_is_not_int"),
        pytest.param([1.0, float("nan")], [1.0, 2.0], id="nested_nan_is_not_value"),
        pytest.param([None], [float("nan")], id="nested_none_is_not_nan"),
        pytest.param([-0.0], [0.0], id="nested_negative_zero_is_not_zero"),
        pytest.param([1, 2], [1, 2, 3], id="list_length_differs"),
        pytest.param({"a": 5.0}, {"a": 5}, id="dict_float_is_not_int"),
        pytest.param({"a": 1}, {"b": 1}, id="dict_keys_differ"),
        pytest.param({"a": 1}, {"a": 1, "b": 2}, id="dict_key_set_differs"),
        pytest.param({"a": [True]}, {"a": [1]}, id="dict_value_recurses_into_list"),
    ],
)
def test_assert_values_with_nulls_strict_recursion_rejects(actual: Any, expected: Any) -> None:
    with pytest.raises(AssertionError):
        helpers.assert_values_with_nulls([actual], [expected], strict=True)


def test_assert_values_with_nulls_strict_with_nan_is_null_is_contradictory() -> None:
    with pytest.raises(ValueError):
        helpers.assert_values_with_nulls([1.0], [1.0], strict=True, nan_is_null=True)


def test_assert_values_with_nulls_strict_floats_compare_approximately() -> None:
    helpers.assert_values_with_nulls([0.1 + 0.2], [0.3], strict=True)


def test_assert_values_with_nulls_not_strict_keeps_loose_equality() -> None:
    helpers.assert_values_with_nulls([5.0, 1], [5, True])


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


# -- helpers.typed_data_without ---------------------------------------------


def test_typed_data_without_drops_the_column_and_keeps_values_and_types() -> None:
    canonical = DataOperationsTestDataCreator.typed_data()
    result = helpers.typed_data_without(canonical, "value_int", "name")
    assert list(result.types) == [c for c in canonical.types if c not in ("value_int", "name")]
    assert result.types == {c: t for c, t in canonical.types.items() if c in result.types}
    assert result.to_python_dict() == {c: v for c, v in canonical.to_python_dict().items() if c in result.types}


def test_typed_data_without_unknown_column_raises_key_error() -> None:
    with pytest.raises(KeyError):
        helpers.typed_data_without(DataOperationsTestDataCreator.typed_data(), "no_such_column")


# -- helpers.append_typed_columns and helpers.empty_like ---------------------


def test_append_typed_columns_rejects_an_existing_column_name() -> None:
    with pytest.raises(ValueError, match="value_int"):
        helpers.append_typed_columns(
            DataOperationsTestDataCreator.typed_data(), {"value_int": (DataType.INT64, [0] * 12)}
        )


def test_append_typed_columns_keeps_values_and_types_of_the_others() -> None:
    canonical = DataOperationsTestDataCreator.typed_data()
    result = helpers.append_typed_columns(canonical, {"extra": (DataType.INT64, list(range(12)))})
    assert list(result.types) == [*canonical.types, "extra"]
    assert result.types == {**canonical.types, "extra": DataType.INT64}
    assert result.to_python_dict() == {**canonical.to_python_dict(), "extra": list(range(12))}


def test_empty_like_keeps_columns_and_types_with_zero_rows() -> None:
    canonical = DataOperationsTestDataCreator.typed_data()
    result = helpers.empty_like(canonical)
    assert list(result.types) == list(canonical.types)
    assert result.types == canonical.types
    assert result.to_python_dict() == {name: [] for name in canonical.types}


def test_canonical_data_empty_equals_empty_like_of_the_canonical_data() -> None:
    expected = helpers.empty_like(DataOperationsTestDataCreator.typed_data())
    actual = helpers.canonical_data_empty()
    assert actual.types == expected.types
    assert actual.to_python_dict() == expected.to_python_dict()


# -- DataOpsTestBase.source_data ----------------------------------------------


def test_source_data_default_returns_canonical_typed_data() -> None:
    data = _StubDataOpsTestBase.source_data()
    assert isinstance(data, TypedTestData)
    assert data.to_python_dict() == DataOperationsTestDataCreator.typed_data().to_python_dict()
    assert to_arrow_table(data).equals(PyArrowDataOpsTestDataCreator.create())


def test_source_arrow_types_defaults_to_empty() -> None:
    assert not _StubDataOpsTestBase.source_arrow_types()


def test_source_arrow_table_is_gone_from_the_base() -> None:
    assert not hasattr(DataOpsTestBase, "source_arrow_table")


def test_reference_implementation_class_is_gone_and_reference_calculate_must_be_overridden() -> None:
    assert not hasattr(DataOpsTestBase, "reference_implementation_class")
    with pytest.raises(NotImplementedError, match="reference_calculate"):
        _StubDataOpsTestBase.reference_calculate(pa.table({"f": [1]}), helpers.make_feature_set("f"))


def test_setup_method_uses_source_data_default() -> None:
    stub = _StubDataOpsTestBase()
    stub.setup_method()
    assert stub._source.to_python_dict() == DataOperationsTestDataCreator.typed_data().to_python_dict()
    assert stub.test_data is stub._source
    assert stub._arrow_table.equals(PyArrowDataOpsTestDataCreator.create())


def test_sqlite_creator_round_trips_to_the_canonical_table() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        relation = SqliteDataOpsTestDataCreator.create(connection)
        assert relation.to_arrow_table().equals(PyArrowDataOpsTestDataCreator.create())
    finally:
        connection.close()


class _StubWithCustomSourceData(_StubDataOpsTestBase):
    """A stub overriding ``source_data`` with a non-canonical table."""

    @classmethod
    def source_data(cls) -> TypedTestData:
        return TypedTestData({"custom": (DataType.INT64, [1, 2, 3])})


def test_setup_method_uses_overridden_source_data() -> None:
    stub = _StubWithCustomSourceData()
    stub.setup_method()
    assert stub._arrow_table.column_names == ["custom"]
    assert stub.test_data.to_python_dict() == {"custom": [1, 2, 3]}


# -- Typed canonical data -------------------------------------------------------


def test_canonical_typed_data_declares_the_expected_types() -> None:
    expected = {
        "region": DataType.STRING,
        "category": DataType.STRING,
        "name": DataType.STRING,
        "timestamp": DataType.TIMESTAMP_MICROS_TZ,
        "event_date": DataType.DATE,
        "value_int": DataType.INT64,
        "value_float": DataType.DOUBLE,
        "amount": DataType.DOUBLE,
        "score": DataType.DOUBLE,
        "is_active": DataType.BOOLEAN,
    }
    assert DataOperationsTestDataCreator.typed_data().types == expected


def test_canonical_typed_data_matches_the_arrow_table() -> None:
    typed = DataOperationsTestDataCreator.typed_data()
    schema = pa.schema(
        [
            ("region", pa.string()),
            ("category", pa.string()),
            ("timestamp", pa.timestamp("us", tz="UTC")),
            ("event_date", pa.date32()),
            ("value_int", pa.int64()),
            ("value_float", pa.float64()),
            ("amount", pa.float64()),
            ("name", pa.string()),
            ("is_active", pa.bool_()),
            ("score", pa.float64()),
        ]
    )
    assert typed.to_python_dict() == to_arrow_table(typed).to_pydict()
    assert to_arrow_table(typed).schema == schema
    assert to_arrow_table(typed).equals(PyArrowDataOpsTestDataCreator.create())


def test_pyarrow_creator_has_no_explicit_types() -> None:
    assert not hasattr(PyArrowDataOpsTestDataCreator, "EXPLICIT_TYPES")


# -- Lazy per-instance _arrow_table -------------------------------------------------


def test_arrow_table_is_built_lazily_and_cached_per_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[Any] = []
    real = helpers.arrow_table_of

    def spy(data: Any, arrow_types: Any = None) -> Any:
        built.append(data)
        return real(data, arrow_types)

    monkeypatch.setattr(helpers, "arrow_table_of", spy)
    first = _StubDataOpsTestBase()
    first.setup_method()
    assert built == []
    table = first._arrow_table
    assert first._arrow_table is table
    assert len(built) == 1
    second = _StubDataOpsTestBase()
    second.setup_method()
    assert second._arrow_table is not table
    assert second._arrow_table.equals(table)


# -- helpers.arrow_table_of ----------------------------------------------------------


def test_arrow_table_of_returns_the_arrow_table() -> None:
    typed = DataOperationsTestDataCreator.typed_data()
    assert helpers.arrow_table_of(typed).equals(to_arrow_table(typed))
    data = TypedTestData({"v": (DataType.DOUBLE, [0.1])})
    assert helpers.arrow_table_of(data, {"v": pa.float32()}).schema.field("v").type == pa.float32()


def test_arrow_table_of_requires_pyarrow_first(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(helpers, "require_pyarrow", lambda: calls.append("require"))
    helpers.arrow_table_of(DataOperationsTestDataCreator.typed_data())
    assert calls == ["require"]


# -- helpers.assert_pyarrow_groups_rows_together -------------------------------------


@pytest.mark.parametrize("keys", [[float("nan"), float("nan"), 1.0], [None, None, 1.0]])
def test_assert_pyarrow_groups_rows_together_passes_when_rows_share_a_group(keys: list[float | None]) -> None:
    helpers.assert_pyarrow_groups_rows_together(keys)


def test_assert_pyarrow_groups_rows_together_fails_when_rows_are_split() -> None:
    with pytest.raises(AssertionError, match="place rows"):
        helpers.assert_pyarrow_groups_rows_together([1.0, 2.0, 1.0])


def test_assert_pyarrow_groups_rows_together_requires_pyarrow_first(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(helpers, "require_pyarrow", lambda: calls.append("require"))
    helpers.assert_pyarrow_groups_rows_together([None, None, 1.0])
    assert calls == ["require"]


# -- Structural guards: setup_method deleted, source_data declared ---


@pytest.mark.parametrize(
    "cls",
    [EmaTestBase, FfillTestBase, LookupTestBase, ResampleTestBase, SessionizationTestBase, TimeBucketizationTestBase],
)
def test_operation_base_has_no_setup_method_override_and_declares_source_data(cls: type) -> None:
    assert "setup_method" not in cls.__dict__
    assert "source_data" in cls.__dict__


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
    def reference_calculate(cls, table: Any, features: Any) -> Any:
        return helpers.calculate_with(_RecordingImpl, table, features)

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


def test_data_ops_feature_group_class_returns_a_feature_group_class_itself() -> None:
    class _Host(_StubDataOpsTestBase):
        @classmethod
        def implementation_class(cls) -> Any:
            return _FeatureGroupPlugin

    assert _Host.feature_group_class() is _FeatureGroupPlugin


def test_calculate_with_and_feature_group_of_for_an_implementation() -> None:
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

        class _Host(_StubDataOpsTestBase):
            _impl: Any = _ProbeImpl

            @classmethod
            def implementation_class(cls) -> Any:
                return cls._impl

        assert _Host.feature_group_class() is _ProbeGroup
        assert helpers.calculate_with(_ProbeImpl, "data", fs) == ("implementation", _ProbeGroup, "data")
        assert fs.implementation is _ProbeImpl
    finally:
        del fs, _Host, _ProbeImpl, _ProbeGroup
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


# -- DataOpsTestBase.contract_version ------------------------------------------


class _VersionedPlugin:
    """FeatureGroup stand-in declaring a contract version."""

    CONTRACT_VERSION = 1


class _NewerVersionedPlugin:
    CONTRACT_VERSION = 2


def _contract_stub(impl: Any, version: int | None = None) -> _StubDataOpsTestBase:
    class _ContractStub(_StubDataOpsTestBase):
        @classmethod
        def implementation_class(cls) -> Any:
            return impl

        if version is not None:

            @classmethod
            def contract_version(cls) -> int:
                return version

    return _ContractStub()


def test_contract_version_defaults_to_one() -> None:
    assert _StubDataOpsTestBase.contract_version() == 1


def test_group_contract_version_passes_when_it_matches_the_kit() -> None:
    _contract_stub(_VersionedPlugin).test_group_contract_version_is_the_kit_version()


def test_group_contract_version_passes_for_an_implementation_of_a_matching_group() -> None:
    def check(group: Any, impl: Any) -> None:
        group.CONTRACT_VERSION = 1
        _contract_stub(impl).test_group_contract_version_is_the_kit_version()

    _with_probe_impl(check)


def test_group_contract_version_fails_naming_both_versions_on_a_mismatch() -> None:
    with pytest.raises(AssertionError) as excinfo:
        _contract_stub(_NewerVersionedPlugin).test_group_contract_version_is_the_kit_version()
    message = str(excinfo.value)
    assert "CONTRACT_VERSION is 2" in message
    assert "expects 1" in message
    assert "contract_version()" in message


def test_group_contract_version_passes_when_the_stub_overrides_the_kit_version() -> None:
    _contract_stub(_NewerVersionedPlugin, version=2).test_group_contract_version_is_the_kit_version()


def test_group_contract_version_fails_for_a_group_declaring_none() -> None:
    with pytest.raises(AssertionError) as excinfo:
        _contract_stub(_FeatureGroupPlugin).test_group_contract_version_is_the_kit_version()
    message = str(excinfo.value)
    assert _FeatureGroupPlugin.__name__ in message
    assert "declares no CONTRACT_VERSION" in message
    assert "override" not in message


# -- helpers.masked_column ----------------------------------------------------

_MASK_TABLE = pa.table({"v": [1, 2, 3, 4], "c": ["X", "Y", "X", "Z"], "n": [5, 6, None, 8]})


@pytest.mark.parametrize(
    ("mask_spec", "expected"),
    [
        pytest.param([("c", "equal", "X")], [1, None, 3, None], id="equal"),
        pytest.param([("c", "equal", "X"), ("v", "greater_than", 1)], [None, None, 3, None], id="and-combined"),
        pytest.param([("c", "is_in", ["X", "Z"]), ("v", "less_than", 4)], [1, None, 3, None], id="is-in-and-less-than"),
        pytest.param([("c", "is_in", ["Y", "Z"])], [None, 2, None, 4], id="is-in"),
        pytest.param([("n", "greater_equal", 6)], [None, 2, None, 4], id="null-mask-value-is-masked-out"),
        pytest.param([("v", "greater_than", 2), ("v", "less_equal", 3)], [None, None, 3, None], id="range"),
    ],
)
def test_masked_column_nulls_rows_failing_any_condition(mask_spec: list[tuple[str, str, Any]], expected: Any) -> None:
    assert helpers.masked_column(_MASK_TABLE, "v", mask_spec).to_pylist() == expected


def test_masked_column_keeps_the_source_column_type() -> None:
    table = pa.table({"v": pa.array([1.5, 2.5], type=pa.float32()), "c": ["X", "Y"]})
    assert helpers.masked_column(table, "v", [("c", "equal", "X")]).type == pa.float32()


def test_masked_column_rejects_an_unknown_operator() -> None:
    with pytest.raises(ValueError, match="bogus"):
        helpers.masked_column(_MASK_TABLE, "v", [("c", "bogus", "X")])


def test_masked_column_does_not_use_the_pyarrow_mask_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_mask_engine import PyArrowMaskEngine

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("PyArrowMaskEngine used")

    monkeypatch.setattr(PyArrowMaskEngine, "all_true", boom)
    assert helpers.masked_column(_MASK_TABLE, "v", [("c", "equal", "X")]).to_pylist() == [1, None, 3, None]


def test_masked_column_raises_key_error_for_a_missing_mask_column() -> None:
    with pytest.raises(KeyError):
        helpers.masked_column(_MASK_TABLE, "v", [("missing", "equal", "X")])


def _make_impl(
    divergences: tuple[Any, ...] | None = None,
    accepts: Callable[..., bool] | None = None,
    supported: frozenset[str] | None = None,
) -> tuple[Any, Any]:
    """A throwaway (group, implementation) pair; use ``_with_probe_impl`` so it is unregistered again."""
    from mloda.provider import Implementation, OperationFeatureGroup
    from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

    class _ProbeGroup(OperationFeatureGroup):
        pass

    class _ProbeImpl(Implementation[_ProbeGroup]):
        feature_group = _ProbeGroup
        compute_framework = PyArrowTable

        @classmethod
        def calculate(cls, group: Any, data: Any, features: Any, framework: Any) -> Any:
            return ("implementation", group, data)

    if divergences is not None:
        # Post-hoc assignment bypasses core's subtype-universe validation, since _ProbeGroup declares no SUBTYPES.
        _ProbeImpl.divergences = divergences
    if accepts is not None:
        _ProbeImpl.accepts = classmethod(accepts)  # type: ignore[method-assign, assignment]
    if supported is not None:
        _ProbeImpl.supported = supported
    return _ProbeGroup, _ProbeImpl


def _probe_impl_names() -> list[str]:
    from mloda.provider import Implementation, get_all_subclasses

    return [c.__name__ for c in get_all_subclasses(Implementation) if c.__name__ == "_ProbeImpl"]


def _with_probe_impl(
    check: Callable[[Any, Any], None],
    divergences: tuple[Any, ...] | None = None,
    accepts: Callable[..., bool] | None = None,
    supported: frozenset[str] | None = None,
) -> None:
    """Run ``check(group, impl)`` on a throwaway pair, then drop it and assert the probe is unregistered."""
    group, impl = _make_impl(divergences, accepts, supported)
    try:
        check(group, impl)
    finally:
        del group, impl
        gc.collect()
    assert not _probe_impl_names()


def _records(*subtypes: str) -> tuple[Any, ...]:
    from mloda.provider import Cause, Divergence, DivergenceKind

    return (
        Divergence(kind=DivergenceKind.VALUE, cause=Cause.BACKEND, subtypes=frozenset(subtypes), reason="skips NaN"),
    )


def _stub_for(impl: Any) -> type[_StubDataOpsTestBase]:
    class _ImplStub(_StubDataOpsTestBase):
        @classmethod
        def implementation_class(cls) -> Any:
            return impl

    return _ImplStub


# -- nan_divergent_agg_types default ------------------------------------------


def test_nan_divergent_agg_types_default_reads_the_implementation_divergent() -> None:
    def check(_: Any, impl: Any) -> None:
        assert _stub_for(impl).nan_divergent_agg_types() == frozenset({"sum", "mean"})

    _with_probe_impl(check, _records("sum", "mean"))


def test_nan_divergent_agg_types_default_is_empty_for_an_implementation_without_divergence() -> None:
    def check(_: Any, impl: Any) -> None:
        assert _stub_for(impl).nan_divergent_agg_types() == frozenset()

    _with_probe_impl(check, ())


def test_nan_divergent_agg_types_default_is_empty_for_a_non_implementation() -> None:
    assert _StubDataOpsTestBase.nan_divergent_agg_types() == frozenset()
    assert _CompareStub.nan_divergent_agg_types() == frozenset()


def test_aggregation_test_base_does_not_override_nan_divergent_agg_types() -> None:
    from mloda.testing.feature_groups.data_operations.aggregation.aggregation import AggregationTestBase

    assert "nan_divergent_agg_types" not in vars(AggregationTestBase)


# -- DataOpsIntegrationTestBase.pipeline_compute_frameworks --------------------


def test_pipeline_compute_frameworks_defaults_to_the_compute_framework_class() -> None:
    from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase

    class _Marker:
        pass

    class _IntegrationStub(DataOpsIntegrationTestBase):
        @classmethod
        def compute_framework_class(cls) -> type:
            return _Marker

    assert _IntegrationStub.pipeline_compute_frameworks() == [_Marker]


def test_nan_is_null_defaults_to_false() -> None:
    from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase

    assert DataOpsIntegrationTestBase.nan_is_null() is False


def test_group_options_defaults_to_empty() -> None:
    from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase

    assert DataOpsIntegrationTestBase.group_options() == {}


def test_routing_frameworks_defaults_to_the_pipeline_frameworks() -> None:
    from mloda.testing.feature_groups.data_operations.mixins.operation_group_integration import (
        OperationGroupIntegrationTestMixin,
    )

    class _Marker:
        pass

    class _RoutingStub(OperationGroupIntegrationTestMixin):
        @classmethod
        def pipeline_compute_frameworks(cls) -> list[type]:
            return [_Marker]

    assert _RoutingStub.routing_frameworks() == [_Marker]


def test_variant_required_case_defaults_to_none() -> None:
    from mloda.testing.feature_groups.data_operations.mixins.operation_group_integration import (
        OperationGroupIntegrationTestMixin,
    )

    assert OperationGroupIntegrationTestMixin.variant_required_case() is None


def test_variant_required_cases_defaults_to_empty() -> None:
    from mloda.testing.feature_groups.data_operations.mixins.operation_group_integration import (
        OperationGroupIntegrationTestMixin,
    )

    class _Host(OperationGroupIntegrationTestMixin):
        pass

    assert _Host.variant_required_cases() == []


def test_variant_required_cases_wraps_an_overridden_variant_required_case() -> None:
    from mloda.testing.feature_groups.data_operations.mixins.operation_group_integration import (
        OperationGroupIntegrationTestMixin,
        RoutingCase,
    )

    case = RoutingCase("f", {}, "m:Impl", "m:Framework")

    class _Host(OperationGroupIntegrationTestMixin):
        @classmethod
        def variant_required_case(cls) -> RoutingCase | None:
            return case

    assert _Host.variant_required_cases() == [case]


def test_routing_case_data_creator_defaults_to_empty() -> None:
    from mloda.testing.feature_groups.data_operations.mixins.operation_group_integration import RoutingCase

    assert RoutingCase("f").data_creator == ""


def test_case_creator_prefers_the_case_data_creator_over_routing_data_creator() -> None:
    """A set ``data_creator`` is resolved without calling ``routing_data_creator()``; an empty one falls back."""
    from mloda.testing.feature_groups.data_operations.mixins.operation_group_integration import (
        SQLITE_DATA_CREATOR,
        OperationGroupIntegrationTestMixin,
        RoutingCase,
    )

    class _Raises(OperationGroupIntegrationTestMixin):
        @classmethod
        def routing_data_creator(cls) -> type[Any]:
            raise AssertionError("routing_data_creator must not be called")

    class _Fallback(OperationGroupIntegrationTestMixin):
        @classmethod
        def routing_data_creator(cls) -> type[Any]:
            return PyArrowDataOpsTestDataCreator

    assert _Raises._case_creator(RoutingCase("f", data_creator=SQLITE_DATA_CREATOR)) is SqliteDataOpsTestDataCreator
    assert _Fallback._case_creator(RoutingCase("f")) is PyArrowDataOpsTestDataCreator


def test_framework_rejection_loads_group_implementations_before_reading_frameworks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rejection test loads the group's Implementations before it reads the group's frameworks."""
    pytest.importorskip("polars")
    from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

    from mloda.testing.feature_groups.data_operations import integration
    from mloda.testing.feature_groups.data_operations.mixins.operation_group_integration import (
        OperationGroupIntegrationTestMixin,
    )

    loaded: list[type] = []

    class _StubGroup:
        @classmethod
        def compute_framework_definition(cls) -> Any:
            return {PolarsDataFrame} if loaded else set()

    def fake_load(group: type) -> None:
        if group is _StubGroup:
            loaded.append(group)

    monkeypatch.setattr(integration, "load_group_implementations", fake_load)

    class _Host(OperationGroupIntegrationTestMixin, _ToyIntegration):
        @classmethod
        def feature_group_class(cls) -> type:
            return _StubGroup

    with pytest.raises(pytest.skip.Exception, match="eager Polars"):
        _Host().test_group_is_rejected_when_none_of_its_frameworks_is_enabled()


class TestEnabledWithImplementations:
    """The integration helper loads a group's Implementations through its manifest, so tests need no ad hoc imports."""

    def test_operation_group_loads_its_manifest(self) -> None:
        from mloda.community.feature_groups.data_operations.aggregation.base import AggregationFeatureGroup
        from mloda.testing.feature_groups.data_operations.integration import enabled_with_implementations

        collector = enabled_with_implementations({PyArrowDataOpsTestDataCreator, AggregationFeatureGroup})
        assert "mloda.community.feature_groups.data_operations.aggregation.manifest" in sys.modules
        assert collector.applicable_feature_group_class(AggregationFeatureGroup)

    def test_non_operation_group_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mloda.testing.feature_groups.data_operations.integration import load_group_implementations

        imported: list[str] = []
        real = importlib.import_module

        def recording(name: str, package: str | None = None) -> Any:
            imported.append(name)
            return real(name, package)

        monkeypatch.setattr(importlib, "import_module", recording)
        load_group_implementations(PyArrowDataOpsTestDataCreator)
        assert imported == []

    def test_group_without_a_manifest_module_is_left_alone(self) -> None:
        """A plugin package that ships no ``manifest`` module is not an error."""
        from mloda.provider import OperationFeatureGroup

        from mloda.testing.feature_groups.data_operations.integration import load_group_implementations

        class _ProbeNoManifestGroup(OperationFeatureGroup):
            pass

        try:
            assert f"{_ProbeNoManifestGroup.__module__.rsplit('.', 1)[0]}.manifest" not in sys.modules
            load_group_implementations(_ProbeNoManifestGroup)
        finally:
            del _ProbeNoManifestGroup
            gc.collect()
        assert not [c.__name__ for c in OperationFeatureGroup.__subclasses__() if c.__name__ == "_ProbeNoManifestGroup"]

    def test_import_error_inside_a_manifest_is_not_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the manifest module itself may be missing; a failing import inside it must surface."""
        from mloda.provider import OperationFeatureGroup

        from mloda.testing.feature_groups.data_operations.integration import load_group_implementations

        package = tmp_path / "probe_manifest_pkg"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "manifest.py").write_text("import probe_manifest_missing_dependency\n")
        monkeypatch.syspath_prepend(str(tmp_path))

        class _ProbeBrokenManifestGroup(OperationFeatureGroup):
            pass

        _ProbeBrokenManifestGroup.__module__ = "probe_manifest_pkg.base"
        try:
            with pytest.raises(ModuleNotFoundError, match="probe_manifest_missing_dependency"):
                load_group_implementations(_ProbeBrokenManifestGroup)
        finally:
            del _ProbeBrokenManifestGroup
            for name in ("probe_manifest_pkg", "probe_manifest_pkg.manifest"):
                sys.modules.pop(name, None)
            gc.collect()


# -- CapabilityHookTestMixin guard ---------------------------------------------


def _capability_host(impl: Any) -> Any:
    from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

    from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin

    class _CapabilityHost(CapabilityHookTestMixin, _StubDataOpsTestBase):
        @classmethod
        def implementation_class(cls) -> Any:
            return impl

        def compute_framework_class(self) -> Any:
            return PyArrowTable

    return _CapabilityHost()


def test_capability_hook_rejects_fails_for_an_accepts_override_without_probes() -> None:
    def check(_: Any, impl: Any) -> None:
        try:
            with pytest.raises(AssertionError, match="accepts"):
                _capability_host(impl).test_mixin_capability_hook_rejects()
        except pytest.skip.Exception:
            pytest.fail("guard skipped instead of asserting for an accepts() override without probes")

    _with_probe_impl(check, accepts=lambda cls, feature_name, options, *, part=None: False)


def test_capability_hook_rejects_fails_for_a_supported_restriction_without_probes() -> None:
    def check(_: Any, impl: Any) -> None:
        try:
            with pytest.raises(AssertionError, match="supported"):
                _capability_host(impl).test_mixin_capability_hook_rejects()
        except pytest.skip.Exception:
            pytest.fail("guard skipped instead of asserting for a supported restriction without probes")

    _with_probe_impl(check, supported=frozenset({"sum"}))


def test_capability_hook_rejects_skips_an_unrestricted_implementation_without_probes() -> None:
    def check(_: Any, impl: Any) -> None:
        with pytest.raises(pytest.skip.Exception):
            _capability_host(impl).test_mixin_capability_hook_rejects()

    _with_probe_impl(check)


def test_implementation_import_fails_unless_the_dependency_is_declared() -> None:
    _Stub = _OperationGroupStub
    missing = ImplSpec("x", "no_such_module", "X", "m", "M")
    with pytest.raises(ModuleNotFoundError):
        _Stub._implementation(missing)
    with pytest.raises(pytest.skip.Exception):
        _Stub._implementation(missing._replace(dependency="no_such_dependency"))
    with pytest.raises(ModuleNotFoundError):
        _Stub._implementation(missing._replace(dependency="pytest"))


# -- divergent_subtypes ---------------------------------------------------------


def test_importing_the_data_op_helpers_defines_no_feature_group() -> None:
    from mloda.testing.import_isolation import fresh_import_report

    report = fresh_import_report("mloda.testing.feature_groups.data_operations.base", "mloda.testing")
    assert report.groups == ()


def test_divergent_subtypes_unions_the_subtypes_of_non_gap_records() -> None:
    from mloda.provider import Cause, Divergence, DivergenceKind

    from mloda.testing.feature_groups.data_operations.base import divergent_subtypes

    records = (
        *_records("sum"),
        Divergence(kind=DivergenceKind.TYPE, cause=Cause.BACKEND, subtypes=frozenset({"mean"}), reason="type"),
    )

    def check(_: Any, impl: Any) -> None:
        assert divergent_subtypes(impl) == frozenset({"sum", "mean"})

    _with_probe_impl(check, records)


def test_divergent_subtypes_ignores_gap_and_outputs_records() -> None:
    from mloda.provider import Cause, Divergence, DivergenceKind

    from mloda.testing.feature_groups.data_operations.base import divergent_subtypes

    records = (
        Divergence(kind=DivergenceKind.GAP, cause=Cause.BACKEND, subtypes=frozenset({"sum"}), reason="gap"),
        Divergence(kind=DivergenceKind.VALUE, cause=Cause.BACKEND, outputs=frozenset({"o"}), reason="out"),
    )

    def check(_: Any, impl: Any) -> None:
        assert divergent_subtypes(impl) == frozenset()

    _with_probe_impl(check, records)


def test_divergent_subtypes_is_empty_without_records() -> None:
    from mloda.testing.feature_groups.data_operations.base import divergent_subtypes

    def check(_: Any, impl: Any) -> None:
        assert divergent_subtypes(impl) == frozenset()

    _with_probe_impl(check, ())


def test_divergent_subtypes_record_without_scope_covers_the_whole_universe() -> None:
    from mloda.provider import Cause, Divergence, DivergenceKind

    from mloda.testing.feature_groups.data_operations.base import divergent_subtypes

    records = (Divergence(kind=DivergenceKind.VALUE, cause=Cause.BACKEND, reason="all"),)

    def check(group: Any, impl: Any) -> None:
        group.subtype_universe = classmethod(lambda cls: frozenset({"sum", "mean", "min"}))
        assert divergent_subtypes(impl) == frozenset({"sum", "mean", "min"})

    _with_probe_impl(check, records)


# -- OperationGroupTestBase.test_operation_group_shape ---------------------------


def test_operation_group_test_base_names_the_group_hook_feature_group_class() -> None:
    assert "feature_group_class" in OperationGroupTestBase.__abstractmethods__
    assert not hasattr(OperationGroupTestBase, "group_class")


def test_operation_group_test_base_rejects_renamed_group_class_hook() -> None:
    """Defining the old ``group_class`` hook raises instead of silently making the host abstract."""
    with pytest.raises(TypeError, match="feature_group_class"):

        class _OldHookHost(OperationGroupTestBase):
            group_class = None


class _OperationGroupStub(OperationGroupTestBase):
    """Concrete hooks for a throwaway group; a host adds ``feature_group_class`` and ``implementations``."""

    @classmethod
    def backend_package(cls) -> str:
        return "mloda.testing"

    @classmethod
    def subtype_key(cls) -> str | None:
        return None

    @classmethod
    def loop_method(cls) -> str:
        return "calculate_feature"

    @classmethod
    def compute_method(cls) -> str:
        return "_compute_group"

    @classmethod
    def probe_feature_set(cls) -> Any:
        return helpers.make_feature_set("f")

    @classmethod
    def name_probes(cls) -> list[tuple[str, str]]:
        return []

    @classmethod
    def option_probes(cls) -> list[tuple[str, Any, str]]:
        return []


def _operation_group_host(group: Any, impl: Any, specs: list[ImplSpec] | None = None) -> Any:
    """A host whose specs without a ``dependency`` resolve to ``impl``; a ``dependency`` takes the real loader."""
    group.CONTRACT_VERSION = "1"
    listed = specs or [ImplSpec("probe", "unused", "_ProbeImpl", "unused", "Unused")]

    class _Host(_OperationGroupStub):
        @classmethod
        def feature_group_class(cls) -> Any:
            return group

        @classmethod
        def implementations(cls) -> list[ImplSpec]:
            return listed

        @classmethod
        def _implementation(cls, spec: ImplSpec) -> Any:
            return super()._implementation(spec) if spec.dependency else impl

    return _Host()


def test_operation_group_shape_passes_for_a_clean_implementation() -> None:
    def check(group: Any, impl: Any) -> None:
        _operation_group_host(group, impl).test_operation_group_shape()

    _with_probe_impl(check, ())


def test_operation_group_shape_fails_for_a_supported_subtype_outside_the_universe() -> None:
    def check(group: Any, impl: Any) -> None:
        with pytest.raises(pytest.fail.Exception, match=r"outside the subtype universe"):
            _operation_group_host(group, impl).test_operation_group_shape()

    _with_probe_impl(check, supported=frozenset({"not_a_subtype"}))


def test_operation_group_shape_fails_for_a_divergence_record_without_a_reason() -> None:
    from mloda.provider import Cause, Divergence, DivergenceKind

    # NOT_IMPLEMENTED is the one cause Divergence accepts without a reason; core's shape check rejects it.
    records = (Divergence(kind=DivergenceKind.VALUE, cause=Cause.NOT_IMPLEMENTED, reason="skips NaN"),)

    def check(group: Any, impl: Any) -> None:
        for record in impl.divergences:
            object.__setattr__(record, "reason", "")
        with pytest.raises(pytest.fail.Exception, match="gives no reason"):
            _operation_group_host(group, impl).test_operation_group_shape()

    _with_probe_impl(check, records)


def test_operation_group_shape_checks_installed_specs_when_another_backend_is_missing() -> None:
    specs = [
        ImplSpec("missing", "unused", "_MissingImpl", "unused", "Unused", dependency="no_such_dependency"),
        ImplSpec("probe", "unused", "_ProbeImpl", "unused", "Unused"),
    ]

    def check(group: Any, impl: Any) -> None:
        with pytest.raises(pytest.fail.Exception, match=r"outside the subtype universe"):
            _operation_group_host(group, impl, specs).test_operation_group_shape()

    _with_probe_impl(check, supported=frozenset({"not_a_subtype"}))


def test_divergent_subtypes_named_only_skips_a_scopeless_record() -> None:
    from mloda.provider import Cause, Divergence, DivergenceKind

    from mloda.testing.feature_groups.data_operations.base import divergent_subtypes

    records = (Divergence(kind=DivergenceKind.VALUE, cause=Cause.BACKEND, reason="all"),)

    def check(group: Any, impl: Any) -> None:
        group.subtype_universe = classmethod(lambda cls: frozenset({"sum", "mean", "min"}))
        assert divergent_subtypes(impl, named_only=True) == frozenset()

    _with_probe_impl(check, records)


def test_divergent_subtypes_named_only_still_counts_scoped_records_and_ignores_gap() -> None:
    from mloda.provider import Cause, Divergence, DivergenceKind

    from mloda.testing.feature_groups.data_operations.base import divergent_subtypes

    records = (
        *_records("sum"),
        Divergence(kind=DivergenceKind.VALUE, cause=Cause.BACKEND, reason="all"),
        Divergence(kind=DivergenceKind.GAP, cause=Cause.BACKEND, subtypes=frozenset({"mean"}), reason="gap"),
    )

    def check(group: Any, impl: Any) -> None:
        group.subtype_universe = classmethod(lambda cls: frozenset({"sum", "mean", "min"}))
        assert divergent_subtypes(impl, named_only=True) == frozenset({"sum"})
        assert divergent_subtypes(impl) == frozenset({"sum", "mean", "min"})

    _with_probe_impl(check, records)


def test_nan_divergent_agg_types_default_ignores_a_scopeless_record() -> None:
    from mloda.provider import Cause, Divergence, DivergenceKind

    records = (
        *_records("sum"),
        Divergence(kind=DivergenceKind.VARIANT, variant="nan_as_null", cause=Cause.BACKEND, reason="all"),
    )

    def check(group: Any, impl: Any) -> None:
        group.subtype_universe = classmethod(lambda cls: frozenset({"sum", "mean", "min"}))
        assert _stub_for(impl).nan_divergent_agg_types() == frozenset({"sum"})

    _with_probe_impl(check, records)


# -- declared_supported ---------------------------------------------------------


def test_declared_supported_returns_the_implementation_supported_set() -> None:
    def check(_: Any, impl: Any) -> None:
        assert _stub_for(impl).declared_supported({"x"}) == {"sum", "mean"}

    _with_probe_impl(check, supported=frozenset({"sum", "mean"}))


def test_declared_supported_returns_the_group_universe_when_supported_is_none() -> None:
    def check(group: Any, impl: Any) -> None:
        group.subtype_universe = classmethod(lambda cls: frozenset({"sum", "mean", "min"}))
        assert impl.supported is None
        assert _stub_for(impl).declared_supported({"x"}) == {"sum", "mean", "min"}

    _with_probe_impl(check)


def test_declared_supported_returns_the_default_for_a_non_implementation() -> None:
    assert _StubDataOpsTestBase.declared_supported({"x"}) == {"x"}
    assert _CompareStub.declared_supported({"x", "y"}) == {"x", "y"}


def test_declared_supported_returns_the_default_when_the_universe_is_empty() -> None:
    def check(group: Any, impl: Any) -> None:
        assert impl.supported is None
        assert group.subtype_universe() == frozenset()
        assert _stub_for(impl).declared_supported({"x"}) == {"x"}

    _with_probe_impl(check)


# -- orders_nan_before_none -----------------------------------------------------


def _order_tie_bases() -> list[tuple[type, str]]:
    from mloda.testing.feature_groups.data_operations.row_preserving.ema.ema import EmaTestBase
    from mloda.testing.feature_groups.data_operations.row_preserving.ffill.ffill import FfillTestBase
    from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate.frame_aggregate import (
        FrameAggregateTestBase,
    )
    from mloda.testing.feature_groups.data_operations.row_preserving.offset.offset import OffsetTestBase
    from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.window_aggregation import (
        WindowAggregationTestBase,
    )

    return [
        (EmaTestBase, "ema"),
        (FfillTestBase, "ffill"),
        (FrameAggregateTestBase, "frame"),
        (OffsetTestBase, "offset"),
        (WindowAggregationTestBase, "window"),
    ]


def _hook_for(base: type, impl: Any) -> bool:
    class _Probe(base):  # type: ignore[misc]
        @classmethod
        def implementation_class(cls) -> Any:
            return impl

    result: bool = _Probe.orders_nan_before_none()
    return result


def _nan_as_null(**scope: Any) -> Any:
    from mloda.provider import Cause, Divergence, DivergenceKind

    return Divergence(kind=DivergenceKind.VARIANT, variant="nan_as_null", cause=Cause.BACKEND, reason="tie", **scope)


def _value_record(**scope: Any) -> Any:
    from mloda.provider import Cause, Divergence, DivergenceKind

    return Divergence(kind=DivergenceKind.VALUE, cause=Cause.BACKEND, reason="differs", **scope)


def _hooks(records: tuple[Any, ...]) -> dict[str, bool]:
    seen: dict[str, bool] = {}

    def check(group: Any, impl: Any) -> None:
        group.VARIANTS = frozenset({"nan_as_null"})
        for base, label in _order_tie_bases():
            seen[label] = _hook_for(base, impl)

    _with_probe_impl(check, records)
    return seen


_ALL = ("ema", "ffill", "frame", "offset", "window")


def _flags(**overrides: bool) -> dict[str, bool]:
    return {label: overrides.get(label, True) for label in _ALL}


def test_orders_nan_before_none_ignores_a_scopeless_value_record() -> None:
    assert _hooks((_value_record(),)) == _flags()


def test_orders_nan_before_none_window_ignores_a_value_record_naming_first() -> None:
    assert _hooks((_value_record(subtypes=frozenset({"first"})),))["window"] is True


def test_orders_nan_before_none_flips_on_a_scopeless_nan_as_null_variant() -> None:
    assert _hooks((_nan_as_null(),)) == _flags(**{label: False for label in _ALL})


def test_orders_nan_before_none_window_flips_on_a_nan_as_null_variant_naming_first_or_last() -> None:
    assert _hooks((_nan_as_null(subtypes=frozenset({"first"})),)) == _flags(window=False)
    assert _hooks((_nan_as_null(subtypes=frozenset({"last", "sum"})),)) == _flags(window=False)


def test_orders_nan_before_none_ignores_a_nan_as_null_variant_scoped_elsewhere() -> None:
    assert _hooks((_nan_as_null(subtypes=frozenset({"sum"})),)) == _flags()
    assert _hooks((_nan_as_null(outputs=frozenset({"o"})),)) == _flags()


def test_orders_nan_before_none_is_true_for_a_non_implementation() -> None:
    for base, _label in _order_tie_bases():
        assert _hook_for(base, object) is True


# -- declares_nan_as_null -------------------------------------------------------


def _declares(records: tuple[Any, ...], subtypes: frozenset[str] = frozenset()) -> bool:
    from mloda.testing.feature_groups.data_operations.base import declares_nan_as_null

    seen: list[bool] = []

    def check(group: Any, impl: Any) -> None:
        group.VARIANTS = frozenset({"nan_as_null"})
        seen.append(declares_nan_as_null(impl, subtypes))

    _with_probe_impl(check, records)
    return seen[0]


def test_declares_nan_as_null_is_false_without_records_and_for_other_kinds() -> None:
    assert _declares(()) is False
    assert _declares((_value_record(),)) is False


def test_declares_nan_as_null_is_true_for_a_scopeless_outputless_variant() -> None:
    assert _declares((_nan_as_null(),)) is True
    assert _declares((_nan_as_null(),), frozenset({"first"})) is True


def test_declares_nan_as_null_ignores_an_outputs_scoped_variant() -> None:
    assert _declares((_nan_as_null(outputs=frozenset({"o"})),)) is False


def test_declares_nan_as_null_needs_a_named_subtype_when_scoped() -> None:
    record = _nan_as_null(subtypes=frozenset({"first"}))
    assert _declares((record,)) is False
    assert _declares((record,), frozenset({"first", "last"})) is True
    assert _declares((record,), frozenset({"last"})) is False


def test_declares_nan_as_null_is_false_for_a_non_implementation() -> None:
    from mloda.testing.feature_groups.data_operations.base import declares_nan_as_null

    assert declares_nan_as_null(object) is False
    assert declares_nan_as_null(object, frozenset({"first"})) is False
