"""Pipeline behaviour of the time-ops families with an Implementation per framework.

Group and Implementation shape, subtype resolution and the manifest live in each family's
``tests/test_base.py``. ffill and time_bucketization run these pipeline checks in their integration tests.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import pytest
from mloda.provider import Implementation, get_all_subclasses
from mloda.user import Feature, FeatureResolutionError, Options, PluginCollector, mloda

from mloda.testing.data_creator.pandas import PandasDataOpsTestDataCreator
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator

_ROW_PRESERVING = "mloda.community.feature_groups.data_operations.row_preserving"
_ROW_CHANGING = "mloda.community.feature_groups.data_operations.row_changing"

_PYARROW = ("mloda_plugins.compute_framework.base_implementations.pyarrow.table", "PyArrowTable")
_PANDAS = ("mloda_plugins.compute_framework.base_implementations.pandas.dataframe", "PandasDataFrame")
_POLARS = ("mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe", "PolarsLazyDataFrame")


@dataclass(frozen=True)
class Family:
    """What the pipeline tests need to know about one family."""

    key: str
    package: str
    group: str
    feature_name: str
    context: dict[str, Any]
    frameworks: tuple[tuple[str, str], ...] = (_PYARROW, _PANDAS)
    source: Any = PyArrowDataOpsTestDataCreator

    def group_class(self) -> Any:
        return getattr(importlib.import_module(f"{self.package}.base"), self.group)

    def feature(self) -> Feature:
        return Feature(self.feature_name, options=Options(context=dict(self.context)))

    def collector(self) -> PluginCollector:
        importlib.import_module(f"{self.package}.manifest")  # loads the Implementations that serve the group
        return PluginCollector.enabled_feature_groups({self.source, self.group_class()})


FAMILIES = [
    Family(
        key="sessionization",
        package=f"{_ROW_PRESERVING}.sessionization",
        group="SessionizationFeatureGroup",
        feature_name="timestamp__sessionize_30_minute",
        context={"partition_by": ["region"]},
    ),
    Family(
        key="ema",
        package=f"{_ROW_PRESERVING}.ema",
        group="EmaFeatureGroup",
        feature_name="value_float__ema_2",
        context={"order_by": "timestamp", "partition_by": ["region"]},
        frameworks=(_PANDAS, _POLARS),
        source=PandasDataOpsTestDataCreator,
    ),
    Family(
        key="resample",
        package=f"{_ROW_CHANGING}.resample",
        group="ResampleFeatureGroup",
        feature_name="value_float__resample_1_day_mean",
        context={"time_column": "timestamp", "partition_by": ["region"]},
    ),
]

_BY_FAMILY = pytest.mark.parametrize("family", FAMILIES, ids=[f.key for f in FAMILIES])


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
