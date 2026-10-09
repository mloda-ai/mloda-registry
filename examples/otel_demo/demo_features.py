"""Demo pipeline: orders (csv, parquet or in-memory) joined with customers, then two calculate steps."""

from __future__ import annotations

import tempfile
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

# Per-run settings; the demo runs one pipeline at a time.
_STATE: dict[str, Any] = {"reader": {}, "fail": False}


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
        reader = _STATE["reader"]
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
        return {Feature("OtelDemoRegionAmount")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return _pyarrow_only()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if _STATE["fail"]:
            raise ValueError("otel demo: simulated failure in the last calculate step")
        return {cls.get_class_name(): [v.upper() for v in data["OtelDemoRegionAmount"].to_pylist()]}


def default_data_dir() -> Path:
    """A stable folder so Marquez sees the same dataset across runs."""
    return Path(tempfile.gettempdir()) / "mloda-otel-demo"


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


def run_pipeline(source: str, extenders: set[Extender], data_dir: Path, fail: bool = False) -> list[Any]:
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}, expected one of {SOURCES}")
    path = write_orders(data_dir, source)
    reader: dict[str, Any] = {}
    if source == "csv":
        reader[CsvReader.__name__] = str(path)
    elif source == "parquet":
        reader[ParquetReader.__name__] = str(path)
    _STATE.update(reader=reader, fail=fail)

    left: type[FeatureGroup] = OtelDemoOrdersMemory if source == "memory" else ReadFileFeature
    link = Link.inner(
        JoinSpec(left, Index(("otel_demo_customer_id",))), JoinSpec(OtelDemoCustomers, Index(("otel_demo_cust_id",)))
    )
    groups: set[type[FeatureGroup]] = {left, OtelDemoCustomers, OtelDemoRegionAmount, OtelDemoTaxedRegion}
    try:
        results = mloda.run_all(
            ["OtelDemoTaxedRegion"],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups(groups),
            links={link},
            function_extender=extenders,
        )
    finally:
        _STATE.update(reader={}, fail=False)
    return list(results[0].column("OtelDemoTaxedRegion").to_pylist())
