"""Shape of the time-ops families: one OperationFeatureGroup served by one Implementation per framework."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from dataclasses import dataclass
from typing import Any

import pytest
from mloda.provider import FeatureGroup, Implementation, OperationFeatureGroup, get_all_subclasses
from mloda.user import Feature, FeatureResolutionError, Options, PluginCollector, mloda

from mloda.testing.data_creator.pandas import PandasDataOpsTestDataCreator
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set

_ROW_PRESERVING = "mloda.community.feature_groups.data_operations.row_preserving"
_ROW_CHANGING = "mloda.community.feature_groups.data_operations.row_changing"

_PYARROW = ("mloda_plugins.compute_framework.base_implementations.pyarrow.table", "PyArrowTable")
_PANDAS = ("mloda_plugins.compute_framework.base_implementations.pandas.dataframe", "PandasDataFrame")
_POLARS = ("mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe", "PolarsLazyDataFrame")


@dataclass(frozen=True)
class Impl:
    """One Implementation: module, class, framework (module, class) and the optional dependency it needs."""

    module: str
    name: str
    framework: tuple[str, str]
    dependency: str | None = None


def _impls(op: str, suffix: str, *, skip: tuple[str, ...] = ()) -> tuple[Impl, ...]:
    """The time-ops Implementations of one operation (module ``<fw>_<op>``, class ``<Fw><Suffix>``)."""
    base = "mloda_plugins.compute_framework.base_implementations"
    every = (
        Impl(f"duckdb_{op}", f"Duckdb{suffix}", (f"{base}.duckdb.duckdb_framework", "DuckDBFramework"), "duckdb"),
        Impl(f"pandas_{op}", f"Pandas{suffix}", _PANDAS, "pandas"),
        Impl(
            f"polars_lazy_{op}",
            f"PolarsLazy{suffix}",
            (f"{base}.polars.lazy_dataframe", "PolarsLazyDataFrame"),
            "polars",
        ),
        Impl(f"pyarrow_{op}", f"PyArrow{suffix}", _PYARROW),
        Impl(
            f"python_dict_{op}",
            f"PythonDict{suffix}",
            (f"{base}.python_dict.python_dict_framework", "PythonDictFramework"),
        ),
        Impl(f"sqlite_{op}", f"Sqlite{suffix}", (f"{base}.sqlite.sqlite_framework", "SqliteFramework")),
    )
    return tuple(i for i in every if i.module.split("_" + op)[0] not in skip)


@dataclass(frozen=True)
class Family:
    """What the shape tests need to know about one family."""

    key: str
    package: str
    group: str
    loop: str
    kept_hook: str
    removed_hooks: tuple[str, ...]
    feature_name: str
    context: dict[str, Any]
    implementations: tuple[Impl, ...]
    mixin_owners: tuple[tuple[str, str], ...]
    frameworks: tuple[tuple[str, str], ...] = (_PYARROW, _PANDAS)
    source: Any = PyArrowDataOpsTestDataCreator

    def group_class(self) -> Any:
        return getattr(importlib.import_module(f"{self.package}.base"), self.group)

    def implementation(self, impl: Impl) -> Any:
        """The Implementation class; skips only when its optional backend is not installed."""
        path = f"{self.package}.{impl.module}"
        module = pytest.importorskip(path) if impl.dependency else importlib.import_module(path)
        return getattr(module, impl.name)

    def installed(self) -> list[Impl]:
        return [i for i in self.implementations if i.dependency is None or importlib.util.find_spec(i.dependency)]

    def feature(self) -> Feature:
        return Feature(self.feature_name, options=Options(context=dict(self.context)))

    def collector(self) -> PluginCollector:
        importlib.import_module(f"{self.package}.manifest")  # loads the Implementations that serve the group
        return PluginCollector.enabled_feature_groups({self.source, self.group_class()})


FAMILIES = [
    Family(
        key="ffill",
        package=f"{_ROW_PRESERVING}.ffill",
        group="FfillFeatureGroup",
        loop="ffill",
        kept_hook="_compute_ffill",
        removed_hooks=("_compute_ffill",),
        feature_name="value_float__ffill",
        context={"order_by": "timestamp", "partition_by": ["region"]},
        implementations=_impls("ffill", "Ffill"),
        mixin_owners=(("input_features", "KeyColumnInputsMixin"),),
    ),
    Family(
        key="sessionization",
        package=f"{_ROW_PRESERVING}.sessionization",
        group="SessionizationFeatureGroup",
        loop="sessionize",
        kept_hook="_compute_session",
        removed_hooks=("_compute_session",),
        feature_name="timestamp__sessionize_30_minute",
        context={"partition_by": ["region"]},
        implementations=_impls("sessionization", "Sessionization"),
        mixin_owners=(("match_feature_group_criteria", "NameOnlyMatchMixin"),),
    ),
    Family(
        key="time_bucketization",
        package=f"{_ROW_PRESERVING}.time_bucketization",
        group="TimeBucketizationFeatureGroup",
        loop="bucketize",
        kept_hook="_compute_bucket",
        removed_hooks=("_compute_bucket", "_assert_source_column_is_timestamp"),
        feature_name="timestamp__floor_1_day",
        context={},
        implementations=_impls("time_bucketization", "TimeBucketization"),
        mixin_owners=(("input_features", "FeatureChainParserMixin"),),
    ),
    Family(
        key="ema",
        package=f"{_ROW_PRESERVING}.ema",
        group="EmaFeatureGroup",
        loop="ema",
        kept_hook="_compute_ema",
        removed_hooks=("_compute_ema",),
        feature_name="value_float__ema_2",
        context={"order_by": "timestamp", "partition_by": ["region"]},
        implementations=_impls("ema", "Ema", skip=("duckdb", "pyarrow", "sqlite")),
        mixin_owners=(("input_features", "KeyColumnInputsMixin"),),
        frameworks=(_PANDAS, _POLARS),
        source=PandasDataOpsTestDataCreator,
    ),
    Family(
        key="resample",
        package=f"{_ROW_CHANGING}.resample",
        group="ResampleFeatureGroup",
        loop="resample",
        kept_hook="_compute_resample",
        removed_hooks=("_compute_resample",),
        feature_name="value_float__resample_1_day_mean",
        context={"time_column": "timestamp", "partition_by": ["region"]},
        implementations=_impls("resample", "Resample", skip=("sqlite",)),
        mixin_owners=(("input_features", "KeyColumnInputsMixin"),),
    ),
]

_BY_FAMILY = pytest.mark.parametrize("family", FAMILIES, ids=[f.key for f in FAMILIES])
_BY_IMPLEMENTATION = pytest.mark.parametrize(
    ("family", "impl"),
    [pytest.param(f, i, id=f"{f.key}-{i.name}") for f in FAMILIES for i in f.implementations],
)


def _owner(group: Any, method: str) -> str:
    """Name of the first class in the MRO that defines the method."""
    return str(next(c.__name__ for c in group.__mro__ if method in vars(c)))


class TestGroupShape:
    """The group is one concrete OperationFeatureGroup that owns its loop and has no per-framework hook."""

    @_BY_FAMILY
    def test_group_is_a_concrete_operation_group(self, family: Family) -> None:
        group = family.group_class()
        assert issubclass(group, OperationFeatureGroup)
        assert not inspect.isabstract(group)

    @_BY_FAMILY
    def test_group_owns_the_loop_and_no_per_framework_compute(self, family: Family) -> None:
        group = family.group_class()
        assert callable(getattr(group, family.loop))
        for hook in family.removed_hooks:
            assert not hasattr(group, hook), hook

    @_BY_FAMILY
    def test_mixins_still_win_over_the_operation_group(self, family: Family) -> None:
        group = family.group_class()
        for method, owner in family.mixin_owners:
            assert _owner(group, method) == owner, method

    @_BY_FAMILY
    def test_group_needs_an_implementation_on_the_feature_set(self, family: Family) -> None:
        """calculate_feature dispatches to the step's implementation; none on the feature set is an error."""
        fs = make_feature_set(family.feature_name, **family.context)
        with pytest.raises(ValueError, match="no implementation"):
            family.group_class().calculate_feature(PyArrowDataOpsTestDataCreator.create(), fs)


class TestImplementationShape:
    """Each backend is an Implementation naming its (group, framework) pair, not a FeatureGroup."""

    @_BY_IMPLEMENTATION
    def test_implementation_names_its_pair(self, family: Family, impl: Impl) -> None:
        implementation = family.implementation(impl)
        framework = getattr(pytest.importorskip(impl.framework[0]), impl.framework[1])
        assert issubclass(implementation, Implementation)
        assert not issubclass(implementation, FeatureGroup)
        assert implementation.feature_group is family.group_class()
        assert implementation.compute_framework is framework

    @_BY_IMPLEMENTATION
    def test_implementation_keeps_its_compute_hook(self, family: Family, impl: Impl) -> None:
        assert callable(getattr(family.implementation(impl), family.kept_hook))


class TestManifest:
    """The entry-point manifest lists the group and its installed Implementations."""

    @_BY_FAMILY
    def test_manifest_lists_the_group_and_every_installed_implementation(self, family: Family) -> None:
        feature_groups = importlib.import_module(f"{family.package}.manifest").FEATURE_GROUPS
        group = family.group_class()

        assert group in feature_groups
        assert {cls.__name__ for cls in feature_groups if issubclass(cls, Implementation)} == {
            i.name for i in family.installed()
        }
        implementations: list[Any] = [cls for cls in feature_groups if cls is not group]
        assert all(cls.feature_group is group for cls in implementations)


class TestRunThroughThePipeline:
    """Resolution and planning treat the family as one group with per-framework implementations."""

    @_BY_FAMILY
    def test_no_implementation_on_the_run_framework_is_a_resolution_error(self, family: Family) -> None:
        """With every implementation for the framework disabled the group fails resolution, naming why."""
        group = family.group_class()
        collector = family.collector()
        framework = getattr(pytest.importorskip(family.frameworks[0][0]), family.frameworks[0][1])
        disabled = {
            i
            for i in get_all_subclasses(Implementation)
            if i.__module__.startswith(family.package)
            and issubclass(group, i.feature_group)
            and issubclass(framework, i.compute_framework)
        }
        assert disabled
        collector.add_disabled_implementation_classes(disabled)

        with pytest.raises(FeatureResolutionError, match="disabled or denied") as excinfo:
            mloda.run_all([family.feature()], compute_frameworks=[framework], plugin_collector=collector)
        assert family.group in str(excinfo.value)

    @_BY_FAMILY
    def test_several_frameworks_enabled_runs_the_feature(self, family: Family) -> None:
        frameworks = [getattr(pytest.importorskip(m), n) for m, n in family.frameworks]

        results = mloda.run_all([family.feature()], compute_frameworks=frameworks, plugin_collector=family.collector())

        holders = [
            r
            for r in results
            if family.feature_name in (r.column_names if hasattr(r, "column_names") else getattr(r, "columns", []))
        ]
        assert len(holders) == 1

    @_BY_FAMILY
    def test_several_frameworks_plan_one_compute_step_with_an_implementation(self, family: Family) -> None:
        frameworks = [getattr(pytest.importorskip(m), n) for m, n in family.frameworks]
        group = family.group_class()

        plan = mloda.explain([family.feature()], compute_frameworks=frameworks, plugin_collector=family.collector())

        steps = [s for s in plan if s.step_kind == "compute" and s.feature_group is group]
        assert len(steps) == 1
        assert steps[0].implementation is not None
