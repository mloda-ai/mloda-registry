"""Structural checks shared by the operation groups (one OperationFeatureGroup, one Implementation per framework).

Each check runs once per family in ``FAMILIES``. Extend the list when another family converts.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from dataclasses import dataclass, field
from typing import Any

import pytest
from mloda.provider import FeatureGroup, Implementation, OperationFeatureGroup
from mloda.user import Options

from mloda.community.feature_groups.data_operations.aggregation.base import AggregationFeatureGroup
from mloda.community.feature_groups.data_operations.aggregation_base import AggregationFeatureGroupBase
from mloda.community.feature_groups.data_operations.capability_hook import SubtypeCapabilityHook
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.base import (
    ScalarAggregateFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)
from mloda.testing.feature_groups.data_operations.helpers import load_implementation, make_feature_set
from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate.frame_aggregate import (
    config_frame_options,
)

_PYARROW = ("mloda_plugins.compute_framework.base_implementations.pyarrow.table", "PyArrowTable")
_PANDAS = ("mloda_plugins.compute_framework.base_implementations.pandas.dataframe", "PandasDataFrame")
_POLARS = ("mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe", "PolarsLazyDataFrame")
_DUCKDB = ("mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework", "DuckDBFramework")
_SQLITE = ("mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework", "SqliteFramework")
_PYTHON_DICT = (
    "mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework",
    "PythonDictFramework",
)


@dataclass(frozen=True)
class _Impl:
    module: str
    name: str
    framework: tuple[str, str]
    dependency: str | None = None


@dataclass(frozen=True)
class _Resolve:
    """A ``resolve_subtype(feature_name, options)`` call and the subtype it must return."""

    feature_name: str
    options: Options
    expected: str | None


@dataclass(frozen=True)
class _Family:
    id: str
    group: type[Any]
    package: str
    compute_name: str
    feature_name: str
    partition_by: list[str]
    implementations: tuple[_Impl, ...]
    order_by: str | None = None
    # The SUBTYPES key, None for a resolver-based declaration.
    subtype_key: str | None = None
    # Exact ``supported`` / ``divergent`` sets by implementation module; a module not listed is unrestricted / clean.
    supported: dict[str, frozenset[str]] = field(default_factory=dict)
    divergent: dict[str, frozenset[str]] = field(default_factory=dict)
    resolve_cases: tuple[_Resolve, ...] = ()


_AGGREGATION = _Family(
    id="aggregation",
    group=AggregationFeatureGroup,
    package="mloda.community.feature_groups.data_operations.aggregation",
    compute_name="_compute_group",
    feature_name="value_int__sum_agg",
    partition_by=["region"],
    implementations=(
        _Impl("pyarrow_aggregation", "PyArrowAggregation", _PYARROW),
        _Impl("pandas_aggregation", "PandasAggregation", _PANDAS, "pandas"),
        _Impl("polars_lazy_aggregation", "PolarsLazyAggregation", _POLARS, "polars"),
        _Impl("duckdb_aggregation", "DuckdbAggregation", _DUCKDB, "duckdb"),
        _Impl("sqlite_aggregation", "SqliteAggregation", _SQLITE),
        _Impl("python_dict_aggregation", "PythonDictAggregation", _PYTHON_DICT),
    ),
    subtype_key="aggregation_type",
    supported={
        "pyarrow_aggregation": frozenset(
            {
                "sum",
                "avg",
                "mean",
                "count",
                "min",
                "max",
                "std",
                "var",
                "std_pop",
                "std_samp",
                "var_pop",
                "var_samp",
                "nunique",
                "first",
                "last",
            }
        ),
        "sqlite_aggregation": frozenset({"sum", "avg", "mean", "count", "min", "max"}),
    },
    divergent={
        "pandas_aggregation": frozenset(
            {"mode", "sum", "avg", "mean", "count", "std", "var", "std_pop", "var_pop", "std_samp", "var_samp"}
        ),
        "sqlite_aggregation": frozenset({"sum", "avg", "mean", "count"}),
    },
    resolve_cases=(
        _Resolve("x__sum_agg", Options(), "sum"),
        _Resolve(
            "my_result",
            Options(context={"aggregation_type": "median", "in_features": "x", "partition_by": ["g"]}),
            "median",
        ),
    ),
)

_WINDOW = _Family(
    id="window",
    group=WindowAggregationFeatureGroup,
    package="mloda.community.feature_groups.data_operations.row_preserving.window_aggregation",
    compute_name="_compute_window",
    feature_name="value_int__sum_window",
    partition_by=["region"],
    implementations=(
        _Impl("pyarrow_window_aggregation", "PyArrowWindowAggregation", _PYARROW),
        _Impl("pandas_window_aggregation", "PandasWindowAggregation", _PANDAS, "pandas"),
        _Impl("polars_lazy_window_aggregation", "PolarsLazyWindowAggregation", _POLARS, "polars"),
        _Impl("duckdb_window_aggregation", "DuckdbWindowAggregation", _DUCKDB, "duckdb"),
        _Impl("sqlite_window_aggregation", "SqliteWindowAggregation", _SQLITE),
        _Impl("python_dict_window_aggregation", "PythonDictWindowAggregation", _PYTHON_DICT),
    ),
    subtype_key="window_aggregation_type",
    supported={
        "pyarrow_window_aggregation": frozenset(
            {
                "sum",
                "avg",
                "mean",
                "count",
                "min",
                "max",
                "nunique",
                "std",
                "var",
                "std_pop",
                "std_samp",
                "var_pop",
                "var_samp",
                "first",
                "last",
            }
        ),
        "sqlite_window_aggregation": frozenset({"sum", "avg", "mean", "count", "min", "max"}),
    },
    divergent={
        "pandas_window_aggregation": frozenset(
            {"mode", "sum", "avg", "mean", "count", "std", "var", "std_pop", "var_pop", "std_samp", "var_samp"}
        ),
        "sqlite_window_aggregation": frozenset({"sum", "avg", "mean", "count"}),
    },
    resolve_cases=(
        _Resolve("x__sum_window", Options(), "sum"),
        _Resolve(
            "my_result",
            Options(context={"window_aggregation_type": "median", "in_features": "x", "partition_by": ["g"]}),
            "median",
        ),
    ),
)

_SCALAR = _Family(
    id="scalar",
    group=ScalarAggregateFeatureGroup,
    package="mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate",
    compute_name="_compute_aggregation",
    feature_name="value_int__sum_scalar",
    partition_by=[],
    implementations=(
        _Impl("pyarrow_scalar_aggregate", "PyArrowScalarAggregate", _PYARROW),
        _Impl("pandas_scalar_aggregate", "PandasScalarAggregate", _PANDAS, "pandas"),
        _Impl("polars_lazy_scalar_aggregate", "PolarsLazyScalarAggregate", _POLARS, "polars"),
        _Impl("duckdb_scalar_aggregate", "DuckdbScalarAggregate", _DUCKDB, "duckdb"),
        _Impl("sqlite_scalar_aggregate", "SqliteScalarAggregate", _SQLITE),
        _Impl("python_dict_scalar_aggregate", "PythonDictScalarAggregate", _PYTHON_DICT),
    ),
    subtype_key="aggregation_type",
    supported={"sqlite_scalar_aggregate": frozenset({"sum", "min", "max", "avg", "mean", "count"})},
    divergent={
        "pandas_scalar_aggregate": frozenset(
            {"sum", "avg", "mean", "count", "std", "var", "std_pop", "var_pop", "std_samp", "var_samp"}
        ),
        "sqlite_scalar_aggregate": frozenset({"sum", "avg", "mean", "count"}),
    },
    resolve_cases=(
        _Resolve("x__sum_scalar", Options(), "sum"),
        _Resolve(
            "my_result",
            Options(context={"aggregation_type": "median", "in_features": "x"}),
            "median",
        ),
    ),
)

# Frame has no PyArrow Implementation: PyArrow tables are not a frame backend.
_FRAME = _Family(
    id="frame",
    group=FrameAggregateFeatureGroup,
    package="mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate",
    compute_name="_compute_frame",
    feature_name="value_int__sum_rolling_3",
    partition_by=["region"],
    order_by="value_int",
    implementations=(
        _Impl("pandas_frame_aggregate", "PandasFrameAggregate", _PANDAS, "pandas"),
        _Impl("polars_lazy_frame_aggregate", "PolarsLazyFrameAggregate", _POLARS, "polars"),
        _Impl("duckdb_frame_aggregate", "DuckdbFrameAggregate", _DUCKDB, "duckdb"),
        _Impl("sqlite_frame_aggregate", "SqliteFrameAggregate", _SQLITE),
        _Impl("python_dict_frame_aggregate", "PythonDictFrameAggregate", _PYTHON_DICT),
    ),
    supported={"sqlite_frame_aggregate": frozenset({"sum", "avg", "count", "min", "max"})},
    divergent={
        "pandas_frame_aggregate": frozenset({"sum", "avg", "count", "std", "var"}),
        "sqlite_frame_aggregate": frozenset({"sum", "avg", "count"}),
    },
    resolve_cases=(
        _Resolve("x__sum_rolling_3", Options(), "sum"),
        _Resolve("x__avg_7_day_window", Options(), "avg"),
        _Resolve("x__cumsum", Options(), "sum"),
        _Resolve("x__expanding_max", Options(), "max"),
        _Resolve("my_result", config_frame_options("median", "rolling"), "median"),
        # A name matching no shape resolves to None.
        _Resolve("plain_feature", Options(), None),
    ),
)

FAMILIES: list[_Family] = [_AGGREGATION, _WINDOW, _SCALAR, _FRAME]

_family = pytest.mark.parametrize("family", FAMILIES, ids=lambda family: family.id)
_family_resolve = pytest.mark.parametrize(
    ("family", "case"),
    [
        pytest.param(family, case, id=f"{family.id}-{case.feature_name}")
        for family in FAMILIES
        for case in family.resolve_cases
    ],
)
_family_impl = pytest.mark.parametrize(
    ("family", "impl"),
    [
        pytest.param(family, impl, id=f"{family.id}-{impl.module}")
        for family in FAMILIES
        for impl in family.implementations
    ],
)


def _implementation(family: _Family, impl: _Impl) -> Any:
    """The named Implementation; skips when its optional backend is not installed."""
    return load_implementation(family.package, impl.module, impl.name)


class TestOperationGroupShape:
    """Each group is one OperationFeatureGroup served by one Implementation per framework."""

    @_family
    def test_group_is_a_concrete_operation_group(self, family: _Family) -> None:
        assert issubclass(family.group, OperationFeatureGroup)
        assert not inspect.isabstract(family.group)

    @_family
    def test_group_owns_the_loop_and_no_per_framework_compute(self, family: _Family) -> None:
        assert callable(family.group.aggregate)
        assert not hasattr(family.group, family.compute_name)

    @_family
    def test_group_needs_an_implementation_on_the_feature_set(self, family: _Family) -> None:
        """calculate_feature dispatches to the step's implementation; none on the feature set is an error."""
        from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator

        fs = make_feature_set(family.feature_name, family.partition_by, family.order_by)
        with pytest.raises(ValueError, match="no implementation"):
            family.group.calculate_feature(PyArrowDataOpsTestDataCreator.create(), fs)

    @_family_impl
    def test_implementation_names_its_pair(self, family: _Family, impl: _Impl) -> None:
        implementation = _implementation(family, impl)
        framework = getattr(pytest.importorskip(impl.framework[0]), impl.framework[1])
        assert issubclass(implementation, Implementation)
        assert not issubclass(implementation, FeatureGroup)
        assert implementation.feature_group is family.group
        assert implementation.compute_framework is framework

    @_family_impl
    def test_implementation_keeps_its_compute(self, family: _Family, impl: _Impl) -> None:
        assert callable(getattr(_implementation(family, impl), family.compute_name))


class TestSupportedAndDivergent:
    """Capability and NaN-divergence are declared per Implementation, inside the group's subtype universe."""

    @_family_impl
    def test_supported_is_the_declared_set_or_unrestricted(self, family: _Family, impl: _Impl) -> None:
        assert _implementation(family, impl).supported == family.supported.get(impl.module)

    @_family_impl
    def test_divergent_is_the_declared_set_or_empty(self, family: _Family, impl: _Impl) -> None:
        divergent = _implementation(family, impl).divergent
        assert set(divergent) == family.divergent.get(impl.module, frozenset())

    @_family_impl
    def test_divergent_is_inside_the_universe_with_a_reason(self, family: _Family, impl: _Impl) -> None:
        divergent = _implementation(family, impl).divergent
        assert set(divergent) <= family.group.subtype_universe()
        assert all(isinstance(reason, str) and reason for reason in divergent.values())

    @_family_impl
    def test_divergent_avg_implies_mean(self, family: _Family, impl: _Impl) -> None:
        """mean is an alias of avg, so a backend that diverges on one diverges on the other."""
        universe = family.group.subtype_universe()
        if not {"avg", "mean"} <= universe:
            pytest.skip("the universe lacks avg or mean")
        divergent = _implementation(family, impl).divergent
        assert ("avg" in divergent) == ("mean" in divergent)

    @_family_impl
    def test_supported_stays_inside_the_universe(self, family: _Family, impl: _Impl) -> None:
        """supported stays inside the universe, so a typo cannot silently hide a real subtype."""
        supported = _implementation(family, impl).supported
        if supported is None:
            pytest.skip("unrestricted implementation")
        assert supported <= family.group.subtype_universe()


class TestSubtypeDeclaration:
    """supported and divergent apply only if the group resolves the subtype; a miss is silent None."""

    @_family
    def test_group_declares_its_subtype_key(self, family: _Family) -> None:
        declaration = family.group.SUBTYPES
        assert declaration is not None
        assert declaration.key == family.subtype_key

    @_family_resolve
    def test_resolve_subtype(self, family: _Family, case: _Resolve) -> None:
        assert family.group.resolve_subtype(case.feature_name, case.options) == case.expected


class TestManifest:
    """The entry-point manifest lists the group and its Implementations."""

    @_family
    def test_manifest_lists_the_group_and_every_installed_implementation(self, family: _Family) -> None:
        manifest = importlib.import_module(f"{family.package}.manifest")
        feature_groups = manifest.FEATURE_GROUPS

        assert family.group in feature_groups
        listed = {cls.__name__ for cls in feature_groups if issubclass(cls, Implementation)}
        expected = {
            impl.name
            for impl in family.implementations
            if impl.dependency is None or importlib.util.find_spec(impl.dependency) is not None
        }
        assert listed == expected
        implementations: list[Any] = [cls for cls in feature_groups if cls is not family.group]
        assert all(cls.feature_group is family.group for cls in implementations)


class TestAggregationBaseHasNoCapabilityHook:
    """Capability lives on Implementation.supported; the shared aggregation base carries no hook."""

    def test_base_is_not_a_subtype_capability_hook(self) -> None:
        assert not issubclass(AggregationFeatureGroupBase, SubtypeCapabilityHook)

    def test_base_has_no_capability_subtype(self) -> None:
        assert not hasattr(AggregationFeatureGroupBase, "_capability_subtype")

    @_family
    def test_groups_carry_no_hook(self, family: _Family) -> None:
        assert not issubclass(family.group, SubtypeCapabilityHook)
