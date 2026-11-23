"""Tests for ``mixins/result_type.py``: the declared-return-type check, driven on stub hosts and plugins."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
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

from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.frameworks import (
    PandasObjectStringTestMixin,
    PandasTestMixin,
    PyArrowTestMixin,
    PythonDictTestMixin,
    SqliteTestMixin,
)
from mloda.testing.feature_groups.data_operations.mixins.result_type import ResultTypeTestMixin
from mloda.testing.feature_groups.data_operations.mixins.tests.test_frameworks import _StubTeardownBase
from mloda.testing.typed_data import TypedTestData

pa = pytest.importorskip("pyarrow")

_FRAMEWORK_MIXINS: dict[str, type] = {
    "pyarrow": PyArrowTestMixin,
    "pandas": PandasTestMixin,
    "pandas_object_string": PandasObjectStringTestMixin,
    "python_dict": PythonDictTestMixin,
    "sqlite": SqliteTestMixin,
}

_CASE = "out_cnt"
_GROUP_NAME = "_ResultTypeStubGroup"
_IMPL_NAME = "_ResultTypeStubImpl"

_NO_NULL_INPUT = TypedTestData({"value_int": (DataType.INT64, [1, 2, 3])})


def _record(kind: DivergenceKind, *subtypes: str, variant: str | None = None) -> Divergence:
    return Divergence(
        kind=kind, cause=Cause.BACKEND, subtypes=frozenset(subtypes), reason="stub reason", variant=variant
    )


def _with_column(framework: str, data: Any, kind: str) -> Any:
    """``data`` plus the output column: ``exact`` int64, ``narrow`` int32, ``text`` strings (object dtype on Pandas)."""
    if framework == "pyarrow":
        rows = data.num_rows
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
    return data.append_column(_CASE, ["a"] * len(data) if kind == "text" else [1] * len(data))


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
    data: TypedTestData | None = None,
    empty_cases: bool = False,
    raises: bool = False,
    writes: bool = True,
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
            return str(feature_name).startswith("out_")

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"out_cnt", "out_other"}

        @classmethod
        def return_data_type_rule(cls, feature: Feature) -> DataType | None:
            return rule

        def input_features(self, options: Options, feature_name: Any) -> set[Feature]:
            return {Feature(input_name)}

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
        "result_type_cases": classmethod(lambda cls: {} if empty_cases else {_CASE: make_feature_set(_CASE)}),
    }
    if data is not None:
        namespace["result_type_data"] = classmethod(lambda cls: data)
    host: Any = type("_ResultTypeHost", (mixin, ResultTypeTestMixin, _StubTeardownBase), namespace)()
    try:
        host.setup_method()
        try:
            host.test_mixin_result_type(_CASE)
        except AssertionError as error:
            state, message = "failed", str(error)
        except pytest.skip.Exception:
            state = "skipped"
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


class TestHookContract:
    def test_case_without_a_type_rule_fails(self) -> None:
        outcome = _run("pyarrow", rule=None)
        assert outcome.state == "failed"
        assert "out_cnt" in outcome.message

    def test_empty_case_dict_fails(self) -> None:
        outcome = _run("pyarrow", empty_cases=True)
        assert outcome.state == "failed"
        assert "result_type_cases" in outcome.message

    def test_case_whose_input_column_holds_no_null_fails(self) -> None:
        outcome = _run("pyarrow", data=_NO_NULL_INPUT)
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
