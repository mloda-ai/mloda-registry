"""Small pipeline runners for exercising Extenders through mloda.run_all."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
from mloda.provider import ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.steward import Extender, ExtenderHook
from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader

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


def _value_int_plus_one_feature_group() -> type[FeatureGroup]:
    """Build a fresh `ValueIntPlusOne` subclass per call so parallel tests never share state."""

    class ValueIntPlusOne(FeatureGroup):
        """Adds one to `value_int`, null-safe; used to exercise two chained calculate invocations."""

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return {Feature("value_int")}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
            return {PyArrowTable}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            values = data["value_int"].to_pylist()
            return {cls.get_class_name(): [None if v is None else v + 1 for v in values]}

    return ValueIntPlusOne


def run_two_features(*extenders: Extender) -> list[Any]:
    """Run a `value_int`-plus-one feature group through the pipeline, chaining two
    FEATURE_GROUP_CALCULATE_FEATURE invocations (the data creator, then this feature group); return the plus-one
    column."""
    feature_group = _value_int_plus_one_feature_group()
    plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, feature_group})
    column_name = feature_group.get_class_name()
    results = mloda.run_all(
        [column_name],
        compute_frameworks={PyArrowTable},
        plugin_collector=plugin_collector,
        function_extender=set(extenders),
    )
    for table in results:
        if isinstance(table, pa.Table) and column_name in table.column_names:
            column: list[Any] = table.to_pydict()[column_name]
            return column
    raise AssertionError(f"No result table with {column_name} found")


def run_csv_feature(directory: Path, *extenders: Extender) -> list[Any]:
    """Write a small CSV into `directory` and run its `alpha` column through the pipeline, firing a nested
    INPUT_DATA_LOAD hook with `data_access_identity` set to the CSV's path; return the column."""
    path = directory / "data.csv"
    path.write_text("alpha,beta\n1,2\n3,4\n", encoding="utf-8")
    plugin_collector = PluginCollector.enabled_feature_groups({ReadFileFeature})
    results = mloda.run_all(
        [Feature("alpha", options={CsvReader.__name__: str(path)})],
        compute_frameworks={PyArrowTable},
        plugin_collector=plugin_collector,
        function_extender=set(extenders),
    )
    for table in results:
        if isinstance(table, pa.Table) and "alpha" in table.column_names:
            column: list[Any] = table.to_pydict()["alpha"]
            return column
    raise AssertionError("No result table with alpha found")


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
