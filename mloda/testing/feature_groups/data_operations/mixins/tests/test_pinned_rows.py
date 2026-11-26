"""Tests for ``mixins/pinned_rows.py``: rejection, divergence, ordering, strict comparison and column sets."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, ClassVar

import pytest
from mloda.provider import Implementation
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


def _make_host(
    impl: type,
    pattern: str | None,
    expected_values_raises: bool = False,
    *,
    table: PinningTable | None = None,
    divergences: Mapping[str, Mapping[str, list[Any]]] | None = None,
    rejection_exc: type[Exception] | None = None,
    as_integers: bool = False,
    adapter: Any = None,
    seen_types: list[Any] | None = None,
    lazy_exc: type[Exception] | None = None,
    override_f: list[Any] | None = None,
    implementation: Any = None,
    checks_types: bool = False,
) -> Any:
    pinned = table or PinningTable(operation="op", rules={}, rows=(_ROW,))

    class Host(PinnedRowTestMixin, _StubBase):
        @classmethod
        def implementation_class(cls) -> Any:
            return implementation if implementation is not None else impl

        @classmethod
        def pinning_table(cls) -> PinningTable:
            return pinned

        @classmethod
        def pinned_divergences(cls) -> Mapping[str, Mapping[str, list[Any]]]:
            return divergences if divergences is not None else super().pinned_divergences()

        @classmethod
        def booleans_as_integers(cls) -> bool:
            return as_integers

        @classmethod
        def checks_pinned_types(cls) -> bool:
            return checks_types

        def create_test_data(self, data: TypedTestData, arrow_types: Mapping[str, Any] | None = None) -> Any:
            if seen_types is not None:
                seen_types.append(arrow_types)
            return super().create_test_data(data, arrow_types)

        def extract_column(self, result: Any, column_name: str) -> list[Any]:
            if lazy_exc is not None:
                raise lazy_exc("operation is not supported lazily")
            return super().extract_column(result, column_name)

        def expected_values(self, row: PinningRow) -> Any:
            if expected_values_raises:
                raise AssertionError("expected_values must not run")
            expected = super().expected_values(row)
            if override_f is not None:
                expected["f"] = override_f
            return expected

        def rejection_match(self, row: PinningRow) -> str | None:
            return pattern

        def rejection_type(self, row: PinningRow) -> type[Exception]:
            return rejection_exc or super().rejection_type(row)

    host = Host()
    if adapter is not None:
        host.adapter = adapter
    return host


def _returns(result: dict[str, list[Any]]) -> type:
    class Returns:
        @staticmethod
        def calculate_feature(data: Any, fs: Any) -> Any:
            return result

    return Returns


def _one_row(
    in_type: DataType,
    in_values: tuple[Any, ...],
    out_type: DataType,
    out_values: tuple[Any, ...],
    row_preserving: bool = True,
) -> PinningTable:
    row = PinningRow(case="c", feature="f", inputs={"a": (in_type, in_values)}, expected={"f": (out_type, out_values)})
    return PinningTable(operation="op", rules={}, rows=(row,), row_preserving=row_preserving)


class _FakeAdapter:
    @staticmethod
    def column_names(result: Any) -> list[str]:
        return list(result)


class _TypedAdapter(_FakeAdapter):
    def __init__(self, has_type_result: bool | None) -> None:
        self._has_type_result = has_type_result

    def has_type(self, result: Any, column: str, data_type: DataType) -> bool | None:
        if not isinstance(result, dict):
            return True  # the fixture keeps its declared types
        return self._has_type_result

    @staticmethod
    def exact_type(result: Any, column: str) -> str:
        return "native_type_x"


class _OtherError(Exception):
    pass


class _RaisesOtherError:
    @staticmethod
    def calculate_feature(data: Any, fs: Any) -> Any:
        raise _OtherError("native failure")


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


class TestPinnedDivergences:
    def test_pin_replaces_the_hand_computed_column(self) -> None:
        pins = {_CASE_ID: {"f": [10, 21]}}
        _make_host(_ComputesWrong, None, divergences=pins).test_mixin_pinned_row(_CASE_ID)
        with pytest.raises(AssertionError):
            _make_host(_Computes, None, divergences=pins).test_mixin_pinned_row(_CASE_ID)

    def test_pin_wins_over_a_host_override_of_the_column(self) -> None:
        pins = {_CASE_ID: {"f": [10, 21]}}
        host = _make_host(_ComputesWrong, None, divergences=pins, override_f=[10, 20])
        host.test_mixin_pinned_row(_CASE_ID)

    def test_valid_pin_passes_the_guard(self) -> None:
        _make_host(_Computes, None, divergences={_CASE_ID: {"f": [1, 2]}}).test_mixin_pinned_divergences_name_rows()

    def test_unknown_row_id_fails_the_guard(self) -> None:
        host = _make_host(_Computes, None, divergences={"9-nope-f": {"f": [1, 2]}})
        with pytest.raises(AssertionError):
            host.test_mixin_pinned_divergences_name_rows()

    def test_unknown_column_fails_the_guard(self) -> None:
        host = _make_host(_Computes, None, divergences={_CASE_ID: {"g": [1, 2]}})
        with pytest.raises(AssertionError):
            host.test_mixin_pinned_divergences_name_rows()


class TestOrderingAndRejectionType:
    def test_skip_in_expected_values_happens_before_create_test_data(self) -> None:
        class Skipping(PinnedRowTestMixin, _StubBase):
            @classmethod
            def pinning_table(cls) -> PinningTable:
                return PinningTable(operation="op", rules={}, rows=(_ROW,))

            def create_test_data(self, data: TypedTestData, arrow_types: Mapping[str, Any] | None = None) -> Any:
                raise AssertionError("create_test_data must not run")

            def expected_values(self, row: PinningRow) -> Any:
                pytest.skip("unsupported")

        with pytest.raises(pytest.skip.Exception):
            Skipping().test_mixin_pinned_row(_CASE_ID)

    def test_rejection_type_accepts_a_native_error(self) -> None:
        _make_host(_RaisesOtherError, "native", rejection_exc=_OtherError).test_mixin_pinned_row(_CASE_ID)

    def test_lazy_rejection_raised_on_column_read_passes(self) -> None:
        _make_host(_Computes, _PATTERN, rejection_exc=_OtherError, lazy_exc=_OtherError).test_mixin_pinned_row(_CASE_ID)

    def test_default_rejection_type_is_value_error(self) -> None:
        with pytest.raises(_OtherError):
            _make_host(_RaisesOtherError, "native").test_mixin_pinned_row(_CASE_ID)


class TestStrictComparison:
    def test_int64_rejects_bool_and_fraction_accepts_lossless_values(self) -> None:
        table = _one_row(DataType.INT64, (1,), DataType.INT64, (1,))
        for ok in (1, 1.0, Decimal(1)):
            _make_host(_returns({"f": [ok]}), None, table=table).test_mixin_pinned_row(_CASE_ID)
        for bad in (True, 1.5):
            with pytest.raises(AssertionError):
                _make_host(_returns({"f": [bad]}), None, table=table).test_mixin_pinned_row(_CASE_ID)

    def test_timestamp_tz_fails_when_the_offset_differs(self) -> None:
        expected = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
        same_instant = datetime(2024, 1, 1, 13, tzinfo=timezone(timedelta(hours=1)))
        table = _one_row(DataType.INT64, (1,), DataType.TIMESTAMP_MICROS_TZ, (expected,))
        _make_host(_returns({"f": [expected]}), None, table=table).test_mixin_pinned_row(_CASE_ID)
        with pytest.raises(AssertionError):
            _make_host(_returns({"f": [same_instant]}), None, table=table).test_mixin_pinned_row(_CASE_ID)

    def test_boolean_accepts_integers_only_when_declared(self) -> None:
        table = _one_row(DataType.INT64, (1,), DataType.BOOLEAN, (True,))
        _make_host(_returns({"f": [1]}), None, table=table, as_integers=True).test_mixin_pinned_row(_CASE_ID)
        with pytest.raises(AssertionError):
            _make_host(_returns({"f": [2]}), None, table=table, as_integers=True).test_mixin_pinned_row(_CASE_ID)
        with pytest.raises(AssertionError):
            _make_host(_returns({"f": [1]}), None, table=table, as_integers=False).test_mixin_pinned_row(_CASE_ID)


@pytest.mark.parametrize(
    ("values", "lossy"), [((1.5,), False), ((0.5, 1.5), False), ((0.1,), True), ((0.5, 0.1), True)]
)
def test_float_inputs_get_float32_arrow_type_only_when_lossy(values: tuple[float, ...], lossy: bool) -> None:
    pa = pytest.importorskip("pyarrow")
    seen: list[Any] = []
    table = _one_row(DataType.FLOAT, values, DataType.INT64, (1,))
    _make_host(_returns({"f": [1]}), None, table=table, seen_types=seen).test_mixin_pinned_row(_CASE_ID)
    if lossy:
        assert seen[0] is not None
        assert seen[0]["a"] == pa.float32()
    else:
        assert seen[0] is None


class TestColumnSet:
    def test_extra_result_column_fails(self) -> None:
        host = _make_host(_returns({"f": [10, 20], "a": [1, 2], "x": [0, 0]}), None, adapter=_FakeAdapter())
        with pytest.raises(AssertionError):
            host.test_mixin_pinned_row(_CASE_ID)

    def test_row_preserving_requires_the_input_columns(self) -> None:
        _make_host(_returns({"f": [10, 20], "a": [1, 2]}), None, adapter=_FakeAdapter()).test_mixin_pinned_row(_CASE_ID)
        with pytest.raises(AssertionError):
            _make_host(_returns({"f": [10, 20]}), None, adapter=_FakeAdapter()).test_mixin_pinned_row(_CASE_ID)

    def test_not_row_preserving_excludes_the_input_columns(self) -> None:
        table = _one_row(DataType.INT64, (1, 2), DataType.INT64, (10, 20), row_preserving=False)
        _make_host(_returns({"f": [10, 20]}), None, table=table, adapter=_FakeAdapter()).test_mixin_pinned_row(_CASE_ID)
        with pytest.raises(AssertionError):
            _make_host(
                _returns({"f": [10, 20], "a": [1, 2]}), None, table=table, adapter=_FakeAdapter()
            ).test_mixin_pinned_row(_CASE_ID)

    def test_without_an_adapter_no_column_check_runs(self) -> None:
        _make_host(_returns({"f": [10, 20], "x": [0, 0]}), None).test_mixin_pinned_row(_CASE_ID)


class TestOutputType:
    @pytest.mark.parametrize(
        ("checks_types", "has_type", "fails"),
        [(True, False, True), (True, None, False), (True, True, False), (False, False, False)],
    )
    def test_declared_type_is_checked_only_when_the_hook_is_on(
        self, checks_types: bool, has_type: bool | None, fails: bool
    ) -> None:
        host = _make_host(
            _returns({"f": [10, 20], "a": [1, 2]}),
            None,
            adapter=_TypedAdapter(has_type),
            checks_types=checks_types,
        )
        if fails:
            with pytest.raises(AssertionError, match=r"f.*INT64.*native_type_x"):
                host.test_mixin_pinned_row(_CASE_ID)
        else:
            host.test_mixin_pinned_row(_CASE_ID)


class _FilterImpl(Implementation[Any]):
    """Abstract (no calculate), so it needs no real group or framework; the subtype is the feature name."""

    feature_group: ClassVar[Any] = object
    supported: ClassVar[frozenset[str] | None] = frozenset({"keep", "refuse"})

    @classmethod
    def canonical_subtype(cls, group: Any, feature_name: str, options: Any) -> str | None:
        return feature_name

    @classmethod
    def accepts(cls, feature_name: str, options: Any, *, part: str | None = None) -> bool:
        return feature_name != "refuse"


def _filter_row(case: str, feature: str) -> PinningRow:
    return PinningRow(
        case=case,
        feature=feature,
        inputs={"a": (DataType.INT64, (1,))},
        expected={feature: (DataType.INT64, (1,))},
    )


class TestCollectionFilter:
    def test_unsupported_and_refused_rows_are_dropped_keeping_the_original_index(self) -> None:
        rows = (
            _filter_row("c", "drop"),
            _filter_row("c", "keep"),
            _filter_row("c", "refuse"),
            _filter_row("d", "keep"),
        )
        table = PinningTable(operation="op", rules={}, rows=rows)
        host = _make_host(_Computes, None, table=table, implementation=_FilterImpl)
        assert list(host.pinned_row_cases()) == ["1-c-keep", "3-d-keep"]

    def test_served_pins_drops_a_pin_on_an_unsupported_row_and_keeps_a_served_one(self) -> None:
        rows = (_filter_row("c", "drop"), _filter_row("c", "keep"))
        table = PinningTable(operation="op", rules={}, rows=rows)
        host = _make_host(_Computes, None, table=table, implementation=_FilterImpl)
        pins = {"0-c-drop": {"drop": [2]}, "1-c-keep": {"keep": [2]}}
        assert host.served_pins(pins) == {"1-c-keep": {"keep": [2]}}


class _PerColumnAdapter(_FakeAdapter):
    """Answers per (frame kind, column); the fixture is a TypedTestData, the result a dict."""

    def __init__(self, fixture: Mapping[str, tuple[bool | None, str]], result: Mapping[str, tuple[bool | None, str]]):
        self._fixture, self._result = fixture, result

    def _lookup(self, frame: Any, column: str) -> tuple[bool | None, str]:
        return (self._result if isinstance(frame, dict) else self._fixture)[column]

    def has_type(self, frame: Any, column: str, data_type: DataType) -> bool | None:
        return self._lookup(frame, column)[0]

    def exact_type(self, frame: Any, column: str) -> str:
        return self._lookup(frame, column)[1]


def _lost_input_table() -> PinningTable:
    row = PinningRow(
        case="c",
        feature="f",
        inputs={"a": (DataType.INT64, (1, None)), "s": (DataType.STRING, ("x", "y"))},
        expected={"f": (DataType.INT64, (1, None)), "g": (DataType.STRING, ("x", "y"))},
    )
    return PinningTable(operation="op", rules={}, rows=(row,))


_LOST_RESULT = {"f": [1, None], "g": ["x", "y"], "a": [1, None], "s": ["x", "y"]}


class TestLostFixtureType:
    @staticmethod
    def _host(fixture_a: str, result_f: tuple[bool | None, str], result_g: bool | None = True) -> Any:
        adapter = _PerColumnAdapter(
            fixture={"a": (False, fixture_a), "s": (True, "str")},
            result={"f": result_f, "g": (result_g, "str"), "a": (True, "x"), "s": (True, "str")},
        )
        return _make_host(_returns(_LOST_RESULT), None, table=_lost_input_table(), adapter=adapter, checks_types=True)

    def test_output_with_the_lost_input_fixture_type_passes(self) -> None:
        self._host("float64", (False, "float64")).test_mixin_pinned_row(_CASE_ID)

    def test_output_with_another_exact_type_fails(self) -> None:
        with pytest.raises(AssertionError):
            self._host("float64", (True, "int64")).test_mixin_pinned_row(_CASE_ID)

    def test_column_matching_no_lost_input_keeps_the_declared_check(self) -> None:
        with pytest.raises(AssertionError, match=r"g.*STRING"):
            self._host("float64", (False, "float64"), result_g=False).test_mixin_pinned_row(_CASE_ID)

    # nothing lost (declared check unchanged) is already covered by test_declared_type_is_checked_only_when_the_hook_is_on
