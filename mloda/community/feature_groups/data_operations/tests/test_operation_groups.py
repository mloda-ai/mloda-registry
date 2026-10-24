"""Structural checks shared by the operation groups (one OperationFeatureGroup, one Implementation per framework).

Each check runs once per family in ``FAMILIES``. Extend the list when another family converts.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from dataclasses import dataclass
from typing import Any

import pytest
from mloda.provider import FeatureGroup, Implementation, OperationFeatureGroup

from mloda.community.feature_groups.data_operations.aggregation.base import AggregationFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set

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
class _Family:
    id: str
    group: type[Any]
    package: str
    compute_name: str
    feature_name: str
    partition_by: list[str]
    implementations: tuple[_Impl, ...]


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
)

# Later conversions (scalar, frame) append their family here.
FAMILIES: list[_Family] = [_AGGREGATION, _WINDOW]

_family = pytest.mark.parametrize("family", FAMILIES, ids=lambda family: family.id)
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
    return getattr(pytest.importorskip(f"{family.package}.{impl.module}"), impl.name)


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

        fs = make_feature_set(family.feature_name, family.partition_by)
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
    """Capability and NaN-divergence declarations stay inside the group's subtype universe."""

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
