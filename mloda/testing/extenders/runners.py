"""Small pipeline runners for exercising Extenders through mloda.run_all."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from mloda.provider import ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.steward import Extender
from mloda.user import PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator


def expected_value_int() -> list[Any]:
    """Expected `value_int` column values from the canonical raw test fixture."""
    return PyArrowDataOpsTestDataCreator.get_raw_data()["value_int"]


def run_value_int(*extenders: Extender) -> list[Any]:
    """Run `value_int` through the pipeline with the given extenders; return the column."""
    plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator})
    results = mloda.run_all(
        ["value_int"],
        compute_frameworks={PyArrowTable},
        plugin_collector=plugin_collector,
        function_extender=set(extenders),
    )
    for table in results:
        if isinstance(table, pa.Table) and "value_int" in table.column_names:
            column: list[Any] = table.to_pydict()["value_int"]
            return column
    raise AssertionError("No result table with value_int found")


class FailingFeatureGroup(FeatureGroup):
    """Primary-source feature group that always raises; the empty feature_name never matches a run."""

    feature_name: str = ""
    calls: int = 0

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({cls.feature_name})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        cls.calls += 1
        raise RuntimeError("inner boom")


def failing_feature_group(feature_name: str) -> type[FailingFeatureGroup]:
    """Build a fresh FailingFeatureGroup subclass per call so parallel tests never share state."""

    class _Failing(FailingFeatureGroup):
        pass

    _Failing.feature_name = feature_name
    _Failing.calls = 0
    _Failing.__name__ = f"FailingFeatureGroup_{feature_name}"
    _Failing.__qualname__ = f"FailingFeatureGroup_{feature_name}"
    return _Failing


def run_failing_feature(feature_group: type[FailingFeatureGroup], *extenders: Extender) -> Any:
    """Run feature_group.feature_name through the pipeline; calculate_feature always raises."""
    plugin_collector = PluginCollector.enabled_feature_groups({feature_group})
    return mloda.run_all(
        [feature_group.feature_name],
        compute_frameworks={PyArrowTable},
        plugin_collector=plugin_collector,
        function_extender=set(extenders),
    )
