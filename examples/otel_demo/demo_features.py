"""Demo pipeline: orders (csv, parquet or in-memory) joined with customers, then two calculate steps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.steward import Extender
from mloda.user import Feature, FeatureName, Index, JoinSpec, Link, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader
from mloda_plugins.feature_group.input_data.read_files.parquet import ParquetReader

SOURCES = ("csv", "parquet", "memory")

ORDERS = {
    "otel_demo_order_id": [10, 11, 12, 13],
    "otel_demo_customer_id": [1, 2, 3, 1],
    "otel_demo_amount": [5.0, 7.0, 9.0, 11.0],
}
CUSTOMERS = {"otel_demo_cust_id": [1, 2, 3], "otel_demo_region": ["north", "south", "north"]}

READERS = (CsvReader, ParquetReader)
FAIL_KEY = "otel_demo_fail"


def _reader_options(options: Options) -> dict[str, Any]:
    """The reader option (reader class name -> path) of a feature, to forward to the order features."""
    return {r.__name__: options.get(r.__name__) for r in READERS if r.__name__ in options}


def _pyarrow_only() -> set[type[ComputeFramework]]:
    return {PyArrowTable}


class OtelDemoCustomers(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(set(CUSTOMERS))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return _pyarrow_only()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table(CUSTOMERS)


class OtelDemoOrdersMemory(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(set(ORDERS))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return _pyarrow_only()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table(ORDERS)


class OtelDemoRegionAmount(FeatureGroup):
    """Joins orders with customers and combines region and amount."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        reader = _reader_options(options)
        return {
            Feature("otel_demo_amount", options=reader),
            Feature("otel_demo_customer_id", options=reader),
            Feature("otel_demo_region"),
        }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return _pyarrow_only()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        rows = zip(data["otel_demo_region"].to_pylist(), data["otel_demo_amount"].to_pylist())
        return {cls.get_class_name(): [f"{region}:{float(amount):.2f}" for region, amount in rows]}


class OtelDemoTaxedRegion(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("OtelDemoRegionAmount", options=_reader_options(options))}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return _pyarrow_only()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if features.get_options_key(FAIL_KEY):
            raise ValueError("otel demo: simulated failure in the last calculate step")
        return {cls.get_class_name(): [v.upper() for v in data["OtelDemoRegionAmount"].to_pylist()]}


def default_data_dir() -> Path:
    """A stable folder so Marquez sees the same dataset across runs."""
    return Path.home() / ".cache" / "mloda-otel-demo"


def write_orders(data_dir: Path, source: str) -> Path | None:
    table = pa.table(ORDERS)
    if source == "csv":
        path = data_dir / "orders.csv"
        pacsv.write_csv(table, str(path))
    elif source == "parquet":
        path = data_dir / "orders.parquet"
        pq.write_table(table, str(path))
    else:
        return None
    return path


def requested_feature(source: str, data_dir: Path, fail: bool = False) -> Feature:
    """The one feature the pipeline requests: reader path as group option, fail flag as context option."""
    group: dict[str, Any] = {}
    if source == "csv":
        group[CsvReader.__name__] = str(data_dir / "orders.csv")
    elif source == "parquet":
        group[ParquetReader.__name__] = str(data_dir / "orders.parquet")
    return Feature("OtelDemoTaxedRegion", Options(group=group, context={FAIL_KEY: fail}))


def run_pipeline(source: str, extenders: set[Extender], data_dir: Path, fail: bool = False) -> list[Any]:
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}, expected one of {SOURCES}")
    write_orders(data_dir, source)
    left: type[FeatureGroup] = OtelDemoOrdersMemory if source == "memory" else ReadFileFeature
    link = Link.inner(
        JoinSpec(left, Index(("otel_demo_customer_id",))), JoinSpec(OtelDemoCustomers, Index(("otel_demo_cust_id",)))
    )
    groups: set[type[FeatureGroup]] = {left, OtelDemoCustomers, OtelDemoRegionAmount, OtelDemoTaxedRegion}
    results = mloda.run_all(
        [requested_feature(source, data_dir, fail)],
        compute_frameworks=[PyArrowTable],
        plugin_collector=PluginCollector.enabled_feature_groups(groups),
        links={link},
        function_extender=extenders,
    )
    return list(results[0].column("OtelDemoTaxedRegion").to_pylist())
