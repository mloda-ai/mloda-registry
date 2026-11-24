"""Tests for ``mixins/result_type.py``: the declared-return-type check, driven on stub hosts and plugins."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pytest
from mloda.provider import (
    Cause,
    Divergence,
    DivergenceKind,
    Implementation,
    OperationFeatureGroup,
    SubtypeDeclaration,
    get_all_subclasses,
)
from mloda.user import DataType, Feature, Options
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.frameworks import (
    PandasObjectStringTestMixin,
    PandasTestMixin,
    PyArrowTestMixin,
    PythonDictTestMixin,
    SqliteComputedColumnTestMixin,
    SqliteTestMixin,
)
from mloda.testing.feature_groups.data_operations.mixins.result_type import ResultTypeTestMixin, _blocking
from mloda.testing.feature_groups.data_operations.mixins.tests.test_frameworks import _StubTeardownBase
from mloda.testing.typed_data import TypedTestData

pa = pytest.importorskip("pyarrow")

_FRAMEWORK_MIXINS: dict[str, type] = {
    "pyarrow": PyArrowTestMixin,
    "pandas": PandasTestMixin,
    "pandas_object_string": PandasObjectStringTestMixin,
    "python_dict": PythonDictTestMixin,
    "sqlite": SqliteTestMixin,
    "sqlite_computed": SqliteComputedColumnTestMixin,
}

_CASE = "out_cnt"
_GROUP_NAME = "_ResultTypeStubGroup"
_IMPL_NAME = "_ResultTypeStubImpl"

_NULL_ONLY_IN_PARTITION = TypedTestData(
    {"value_int": (DataType.INT64, [1, 2, 3]), "grp": (DataType.INT64, [1, None, 2])}
)
_TIMESTAMP_PASSENGER = TypedTestData(
    {
        "value_int": (DataType.INT64, [1, None, 3]),
        "ts": (DataType.TIMESTAMP_MICROS, [datetime(2024, 1, 1)] * 3),
    }
)
_NO_NULL_INPUT = TypedTestData({"value_int": (DataType.INT64, [1, 2, 3])})


def _record(kind: DivergenceKind, *subtypes: str, variant: str | None = None) -> Divergence:
    return Divergence(
        kind=kind, cause=Cause.BACKEND, subtypes=frozenset(subtypes), reason="stub reason", variant=variant
    )


def _with_column(framework: str, data: Any, kind: str) -> Any:
    """``data`` plus the output column: ``exact`` int64, ``narrow`` int32, ``text`` strings (object dtype on Pandas).

    PyArrow only: ``zero_type`` writes strings on zero rows, ``rewrite`` and ``zero_retype`` alter ``value_int``.
    SQLite adapters: ``rewrite`` also alters ``value_int``, ``rewrite_ts`` replaces ``ts`` with another timestamp.
    """
    if framework == "pyarrow":
        rows = data.num_rows
        if kind == "zero_type" and rows == 0:
            kind = "text"
        if kind == "rewrite":
            data = data.set_column(
                data.schema.get_field_index("value_int"), "value_int", pa.array([99] * rows, pa.int64())
            )
        if kind == "zero_retype" and rows == 0:
            data = data.set_column(data.schema.get_field_index("value_int"), "value_int", pa.array([], pa.string()))
        values = ["a"] * rows if kind == "text" else [1] * rows
        arrow_type = pa.string() if kind == "text" else pa.int32() if kind == "narrow" else pa.int64()
        return data.append_column(_CASE, pa.array(values, type=arrow_type))
    if framework in ("pandas", "pandas_object_string"):
        import pandas as pd

        if kind == "text":
            return data.assign(**{_CASE: pd.Series(["a"] * len(data), dtype=object)})
        return data.assign(**{_CASE: pd.Series([1] * len(data), dtype="int32" if kind == "narrow" else "int64")})
    if framework == "python_dict":
        rows = len(next(iter(data.values())))
        return {**data, _CASE: ["a"] * rows if kind == "text" else [1] * rows}
    if kind == "text":
        return data.append_column(_CASE, ["a"] * len(data))
    if kind == "rewrite":
        # SELECT * would repeat value_int, so list the other columns and replace it.
        others = ", ".join(quote_ident(c) for c in data.columns if c != "value_int")
        return data.select(_raw_sql=f"{others}, 99 AS value_int, CAST(1 AS INTEGER) AS {quote_ident(_CASE)}")
    if kind == "rewrite_ts":
        others = ", ".join(quote_ident(c) for c in data.columns if c != "ts")
        return data.select(
            _raw_sql=f"{others}, '2030-05-05 00:00:00' AS ts, CAST(1 AS INTEGER) AS {quote_ident(_CASE)}"
        )
    # A raw select keeps the integer type on zero rows, unlike append_column.
    return data.select(_raw_sql=f"*, CAST(1 AS INTEGER) AS {quote_ident(_CASE)}")


@dataclass
class _Outcome:
    """What the test method did on a stub host: passed, failed with a message, or skipped."""

    state: str
    message: str = ""
    seen_options: list[Options] = field(default_factory=list)


def _run(
    framework: str,
    *,
    kind: str = "exact",
    rule: DataType | None = DataType.INT64,
    impl_records: tuple[Divergence, ...] = (),
    group_records: tuple[Divergence, ...] = (),
    supported: frozenset[str] | None = None,
    input_name: str = "value_int",
    partition_input: str | None = None,
    data: TypedTestData | None = None,
    empty_cases: bool = False,
    raises: bool = False,
    writes: bool = True,
    matches_any: bool = False,
    typed: bool = True,
    untyped: bool = False,
    extra_untyped_ids: frozenset[str] = frozenset(),
    known_findings: dict[str, str] | None = None,
) -> _Outcome:
    """Run ``test_mixin_result_type`` on a stub host over stub plugins, then remove every trace of them."""
    seen: list[Options] = []
    state, message = "passed", ""
    mixin = _FRAMEWORK_MIXINS[framework]
    if framework.startswith("pandas"):
        pytest.importorskip("pandas")
    framework_class = mixin.adapter_class().compute_framework()  # type: ignore[attr-defined]

    class _Group(OperationFeatureGroup):
        SUBTYPES = SubtypeDeclaration(universe=("cnt", "other"), resolver=lambda name, options: name.rsplit("_", 1)[-1])
        VARIANTS = frozenset({"nan_as_null"})
        DIVERGENCES = group_records

        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Any, data_access_collection: Any = None
        ) -> bool:
            return matches_any or str(feature_name).startswith("out_")

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"out_cnt", "out_other"}

        if typed:

            @classmethod
            def return_data_type_rule(cls, feature: Feature) -> DataType | None:
                return rule

        def input_features(self, options: Options, feature_name: Any) -> set[Feature]:
            names = [input_name] if partition_input is None else [input_name, partition_input]
            return {Feature(n) for n in names}

    class _Impl(Implementation[_Group]):
        feature_group = _Group
        compute_framework = framework_class
        divergences = impl_records

        @classmethod
        def calculate(cls, group: Any, data: Any, features: Any, compute: Any) -> Any:
            seen.extend(f.options for f in features.features)
            if raises:
                raise RuntimeError("stub boom")
            return _with_column(framework, data, kind) if writes else data

    _Group.__name__ = _GROUP_NAME
    _Impl.__name__ = _IMPL_NAME
    if supported is not None:
        _Impl.supported = supported
    namespace: dict[str, Any] = {
        "implementation_class": classmethod(lambda cls, impl=_Impl: impl),
        "result_type_cases": classmethod(
            lambda cls: (
                {}
                if empty_cases
                else {_CASE: make_feature_set(_CASE, partition_by=[partition_input] if partition_input else None)}
            )
        ),
    }
    marked = frozenset({_CASE} if untyped else ()) | extra_untyped_ids
    namespace["untyped_result_type_ids"] = classmethod(lambda cls: marked)
    namespace["result_type_known_findings"] = classmethod(lambda cls: dict(known_findings or {}))
    if data is not None:
        namespace["result_type_data"] = classmethod(lambda cls: data)
    host: Any = type("_ResultTypeHost", (mixin, ResultTypeTestMixin, _StubTeardownBase), namespace)()
    try:
        host.setup_method()
        try:
            host.test_mixin_result_type(_CASE)
        except AssertionError as error:
            state, message = "failed", str(error)
        except pytest.skip.Exception as skipped:
            state, message = "skipped", str(skipped)
        finally:
            host.teardown_method()
    finally:
        del host, namespace, _Impl, _Group
        gc.collect()
    leaked = [c.__name__ for c in (*get_all_subclasses(Implementation), *get_all_subclasses(OperationFeatureGroup))]
    assert _IMPL_NAME not in leaked
    assert _GROUP_NAME not in leaked
    return _Outcome(state, message, seen)


class TestExactType:
    @pytest.mark.parametrize("framework", ["pyarrow", "pandas"])
    def test_wrong_exact_type_fails_naming_column_actual_and_declared(self, framework: str) -> None:
        outcome = _run(framework, kind="narrow")
        assert outcome.state == "failed"
        assert "out_cnt" in outcome.message
        assert "int32" in outcome.message
        assert "INT64" in outcome.message

    @pytest.mark.parametrize("framework", ["pyarrow", "pandas", "pandas_object_string"])
    def test_matching_exact_type_passes(self, framework: str) -> None:
        assert _run(framework).state == "passed"


class TestDivergenceRecords:
    def test_type_record_excuses_an_exact_mismatch(self) -> None:
        outcome = _run("pyarrow", kind="narrow", impl_records=(_record(DivergenceKind.TYPE),))
        assert outcome.state == "passed"

    @pytest.mark.parametrize(
        "records",
        [
            pytest.param({"impl_records": (_record(DivergenceKind.GAP),)}, id="gap_on_implementation"),
            pytest.param({"group_records": (_record(DivergenceKind.GAP),)}, id="gap_on_group"),
            pytest.param({"impl_records": (_record(DivergenceKind.TYPE, "other"),)}, id="type_for_another_subtype"),
        ],
    )
    def test_gap_or_unrelated_record_does_not_excuse_an_exact_mismatch(self, records: dict[str, Any]) -> None:
        outcome = _run("pyarrow", kind="narrow", **records)
        assert outcome.state == "failed"
        assert "int32" in outcome.message

    @pytest.mark.parametrize(
        "records",
        [
            pytest.param({"impl_records": (_record(DivergenceKind.GAP),)}, id="gap_on_implementation"),
            pytest.param({"group_records": (_record(DivergenceKind.GAP),)}, id="gap_on_group"),
            pytest.param({"impl_records": (_record(DivergenceKind.GAP, "cnt"),)}, id="gap_for_the_subtype"),
        ],
    )
    def test_gap_record_does_not_excuse_a_column_that_lost_its_type(self, records: dict[str, Any]) -> None:
        outcome = _run("pandas", kind="text", rule=DataType.INT64, **records)
        assert outcome.state == "failed"
        assert "out_cnt" in outcome.message


class TestUnresolvedType:
    @pytest.mark.parametrize("framework", ["pandas", "pandas_object_string"])
    def test_undeclared_unresolved_type_fails_on_a_readback_adapter(self, framework: str) -> None:
        outcome = _run(framework, kind="text", rule=DataType.STRING)
        assert outcome.state == "failed"
        assert "out_cnt" in outcome.message

    @pytest.mark.parametrize(
        "records",
        [
            pytest.param({"impl_records": (_record(DivergenceKind.GAP),)}, id="gap_on_implementation"),
            pytest.param({"group_records": (_record(DivergenceKind.GAP),)}, id="gap_on_group"),
            pytest.param({"impl_records": (_record(DivergenceKind.GAP, "cnt"),)}, id="gap_for_the_subtype"),
        ],
    )
    def test_gap_record_covers_an_unresolved_type(self, records: dict[str, Any]) -> None:
        assert _run("pandas", kind="text", rule=DataType.STRING, **records).state == "passed"

    def test_gap_for_another_subtype_does_not_cover(self) -> None:
        outcome = _run(
            "pandas", kind="text", rule=DataType.STRING, impl_records=(_record(DivergenceKind.GAP, "other"),)
        )
        assert outcome.state == "failed"

    @pytest.mark.parametrize("framework", ["python_dict", "sqlite"])
    def test_adapter_without_exact_readback_is_value_fit_only(self, framework: str) -> None:
        assert _run(framework, kind="text", rule=DataType.STRING).state == "passed"
        assert _run(framework).state == "passed"


class TestValueFit:
    def test_python_dict_fails_when_a_value_does_not_fit_the_declared_type(self) -> None:
        outcome = _run("python_dict", kind="text", rule=DataType.INT64)
        assert outcome.state == "failed"
        assert "out_cnt" in outcome.message

    @pytest.mark.parametrize(
        "records",
        [
            pytest.param({"impl_records": (_record(DivergenceKind.GAP),)}, id="gap_on_implementation"),
            pytest.param({"group_records": (_record(DivergenceKind.GAP),)}, id="gap_on_group"),
            pytest.param({"impl_records": (_record(DivergenceKind.GAP, "cnt"),)}, id="gap_for_the_subtype"),
        ],
    )
    def test_gap_record_does_not_excuse_values_that_do_not_fit(self, records: dict[str, Any]) -> None:
        outcome = _run("python_dict", kind="text", rule=DataType.INT64, **records)
        assert outcome.state == "failed"
        assert "out_cnt" in outcome.message


class TestRunsThatDoNotHappen:
    @pytest.mark.parametrize("framework", ["pyarrow", "python_dict"])
    def test_raising_implementation_fails(self, framework: str) -> None:
        outcome = _run(framework, raises=True)
        assert outcome.state == "failed"
        assert "stub boom" in outcome.message

    @pytest.mark.parametrize("framework", ["pyarrow", "python_dict"])
    def test_missing_input_column_fails(self, framework: str) -> None:
        outcome = _run(framework, input_name="no_such_column")
        assert outcome.state == "failed"
        assert "no_such_column" in outcome.message

    @pytest.mark.parametrize("framework", ["pyarrow", "python_dict"])
    def test_output_column_missing_from_the_result_fails(self, framework: str) -> None:
        outcome = _run(framework, writes=False)
        assert outcome.state == "failed"
        assert "out_cnt" in outcome.message


class TestShapeFailuresBeyondTheType:
    @pytest.mark.parametrize(
        ("kind", "matches_any", "expected", "typed", "marked"),
        [
            pytest.param("zero_type", False, "zero_rows", True, False, id="zero_rows_type_change_on_output"),
            pytest.param("rewrite", False, "was changed", True, False, id="input_column_changed"),
            pytest.param("exact", True, "foreign", True, False, id="match_time_foreign_name"),
            pytest.param(
                "zero_retype",
                False,
                "column 'value_int' changes type on zero rows",
                True,
                False,
                id="group_retypes_input_on_zero_rows",
            ),
            pytest.param("zero_type", False, "zero_rows", False, False, id="untyped_group_zero_rows_type_change"),
            pytest.param("zero_type", False, "zero_rows", True, True, id="marked_case_zero_rows_type_change"),
        ],
    )
    def test_failure_fails_the_test(
        self, kind: str, matches_any: bool, expected: str, typed: bool, marked: bool
    ) -> None:
        rule = None if marked else DataType.INT64
        outcome = _run("pyarrow", kind=kind, matches_any=matches_any, typed=typed, rule=rule, untyped=marked)
        assert outcome.state == "failed"
        assert expected in outcome.message

    @pytest.mark.parametrize(
        ("failure", "keyword", "columns", "blocks"),
        [
            pytest.param(
                "zero_rows: column 'c' changes type on zero rows",
                "adapter_retypes",
                frozenset({"c"}),
                False,
                id="retype_by_the_adapter_is_excused",
            ),
            pytest.param(
                "zero_rows: column 'c' changes type on zero rows",
                "adapter_retypes",
                frozenset({"other"}),
                True,
                id="retype_of_another_column_blocks",
            ),
            pytest.param(
                "zero_rows: column 'c' changes type on zero rows",
                "adapter_retypes",
                frozenset(),
                True,
                id="no_adapter_retype_blocks",
            ),
            pytest.param(
                "outputs: input column 'c' was changed",
                "adapter_rewrites",
                frozenset({"c"}),
                False,
                id="rewrite_by_the_adapter_is_excused",
            ),
            pytest.param(
                "outputs: input column 'c' was changed",
                "adapter_rewrites",
                frozenset({"other"}),
                True,
                id="rewrite_of_another_column_blocks",
            ),
            pytest.param(
                "outputs: input column 'c' was changed",
                "adapter_rewrites",
                frozenset(),
                True,
                id="no_adapter_rewrite_blocks",
            ),
        ],
    )
    def test_failure_is_excused_only_for_columns_the_adapter_alters(
        self, failure: str, keyword: str, columns: frozenset[str], blocks: bool
    ) -> None:
        """A zero-row retype or a changed input blocks unless the adapter itself alters that column."""
        excuse: Any = {keyword: columns}
        bad = _blocking([failure], [], readback=False, gap_covers=False, **excuse)
        assert bad == ([failure] if blocks else [])

    def test_sqlite_computed_adapter_timestamp_readback_is_not_a_changed_input(self) -> None:
        """The adapter reads a TIMESTAMP passenger column back as ISO text; a clean stub still passes."""
        assert _run("sqlite_computed", data=_TIMESTAMP_PASSENGER).state == "passed"

    @pytest.mark.parametrize(
        ("kind", "column"),
        [("rewrite", "value_int"), ("rewrite_ts", "ts")],
        ids=["plain_column", "adapter_altered_column"],
    )
    def test_sqlite_computed_adapter_still_fails_a_rewrite_of_a_source_column(self, kind: str, column: str) -> None:
        outcome = _run("sqlite_computed", kind=kind, data=_TIMESTAMP_PASSENGER)
        assert outcome.state == "failed"
        assert f"input column '{column}' was changed" in outcome.message


class TestHookContract:
    def test_untyped_group_passes_on_a_clean_run(self) -> None:
        assert _run("pyarrow", typed=False).state == "passed"

    def test_case_without_a_type_rule_fails(self) -> None:
        outcome = _run("pyarrow", rule=None)
        assert outcome.state == "failed"
        assert "out_cnt" in outcome.message

    def test_marked_case_without_a_type_rule_passes_a_clean_run(self) -> None:
        assert _run("pyarrow", rule=None, untyped=True).state == "passed"

    def test_marked_case_whose_rule_returns_a_type_fails_naming_the_case(self) -> None:
        outcome = _run("pyarrow", untyped=True)
        assert outcome.state == "failed"
        assert "out_cnt" in outcome.message
        assert "untyped_result_type_ids" in outcome.message

    def test_marked_id_outside_the_cases_fails(self) -> None:
        outcome = _run("pyarrow", extra_untyped_ids=frozenset({"out_ghost"}))
        assert outcome.state == "failed"
        assert "out_ghost" in outcome.message

    def test_marked_ids_on_an_untyped_group_fail(self) -> None:
        outcome = _run("pyarrow", typed=False, untyped=True)
        assert outcome.state == "failed"
        assert "untyped_result_type_ids" in outcome.message

    def test_known_finding_skips_the_case_with_its_reason(self) -> None:
        reason = "stub finding https://example.invalid/issues/1"
        outcome = _run("pyarrow", kind="narrow", known_findings={_CASE: reason})
        assert outcome.state == "skipped"
        assert reason in outcome.message

    def test_known_finding_outside_the_cases_fails(self) -> None:
        outcome = _run("pyarrow", known_findings={"out_ghost": "stub finding https://example.invalid/issues/1"})
        assert outcome.state == "failed"
        assert "out_ghost" in outcome.message

    def test_empty_case_dict_fails(self) -> None:
        outcome = _run("pyarrow", empty_cases=True)
        assert outcome.state == "failed"
        assert "result_type_cases" in outcome.message

    def test_case_whose_input_column_holds_no_null_fails(self) -> None:
        outcome = _run("pyarrow", data=_NO_NULL_INPUT)
        assert outcome.state == "failed"
        assert "value_int" in outcome.message

    def test_null_only_in_a_context_named_input_column_fails_naming_the_source(self) -> None:
        outcome = _run("pyarrow", partition_input="grp", data=_NULL_ONLY_IN_PARTITION)
        assert outcome.state == "failed"
        assert "value_int" in outcome.message


class TestSupportAndVariants:
    def test_subtype_the_implementation_does_not_support_is_skipped(self) -> None:
        assert _run("pyarrow", kind="narrow", supported=frozenset({"other"})).state == "skipped"

    def test_allow_variants_is_set_for_the_implementations_variant_records(self) -> None:
        records = (_record(DivergenceKind.VARIANT, variant="nan_as_null"),)
        outcome = _run("pyarrow", impl_records=records)
        assert outcome.state == "passed"
        assert outcome.seen_options
        assert all("nan_as_null" in str(options.get("allow_variants")) for options in outcome.seen_options)

    def test_allow_variants_is_absent_without_variant_records(self) -> None:
        outcome = _run("pyarrow")
        assert outcome.seen_options
        assert all(options.get("allow_variants") is None for options in outcome.seen_options)
