"""Small pipeline runners for exercising Extenders through mloda.run_all."""

from __future__ import annotations

import uuid
from typing import Any

import pyarrow as pa
from mloda.provider import ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.steward import Extender, ExtenderHook
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


class CountingExtender(Extender):
    """Breaking pass-through probe that counts its own invocations."""

    def __init__(self) -> None:
        self.raise_on_error = True
        self.calls = 0
        # Above the default priority (100) so this probe always sorts downstream of a
        # default-priority host extender, regardless of set iteration order.
        self.priority = 200

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return func(*args, **kwargs)


class FailingFeatureGroup(FeatureGroup):
    """Primary-source feature group that always raises; the sentinel feature_name never matches a real request."""

    feature_name: str = "mloda_testing_never_requested"
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
    suffix = uuid.uuid4().hex[:8]
    _Failing.__name__ = f"FailingFeatureGroup_{feature_name}_{suffix}"
    _Failing.__qualname__ = f"FailingFeatureGroup_{feature_name}_{suffix}"
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
