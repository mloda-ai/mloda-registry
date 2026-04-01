"""Shared helpers and framework mixins for data-operations tests.

Provides:
- ``extract_column``: Extract a column from any framework result as a Python list.
- ``make_feature_set``: Build a FeatureSet with optional partition_by/order_by.
- Framework mixins (``PyArrowTestMixin``, ``PandasTestMixin``, etc.) that implement
  the abstract adapter methods required by ``DataOpsTestBase`` subclasses, so each
  concrete test file only needs to specify ``implementation_class``.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def extract_column(result: Any, column_name: str) -> list[Any]:
    """Extract a column from a result object as a Python list.

    Handles pa.Table (direct .column() access), relation types
    (DuckdbRelation, SqliteRelation) that expose .to_arrow_table(),
    Polars LazyFrames that expose .collect(), and pandas DataFrames.
    """
    if isinstance(result, pa.Table):
        return list(result.column(column_name).to_pylist())
    if hasattr(result, "to_arrow_table"):
        arrow_table = result.to_arrow_table()
        return list(arrow_table.column(column_name).to_pylist())
    if hasattr(result, "collect"):
        df = result.collect()
        return list(df[column_name].to_list())
    return list(result[column_name])


def make_feature_set(
    feature_name: str,
    partition_by: list[str] | None = None,
    order_by: str | None = None,
) -> FeatureSet:
    """Build a FeatureSet with optional partition_by and order_by options."""
    context: dict[str, Any] = {}
    if partition_by is not None:
        context["partition_by"] = partition_by
    if order_by is not None:
        context["order_by"] = order_by
    feature = Feature(feature_name, options=Options(context=context))
    fs = FeatureSet()
    fs.add(feature)
    return fs


# ---------------------------------------------------------------------------
# Framework mixins
# ---------------------------------------------------------------------------


class PyArrowTestMixin:
    """Mixin implementing adapter methods for PyArrow."""

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return arrow_table

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return list(result.column(column_name).to_pylist())

    def get_row_count(self, result: Any) -> int:
        return int(result.num_rows)

    def get_expected_type(self) -> Any:
        return pa.Table


class PandasTestMixin:
    """Mixin implementing adapter methods for Pandas.

    Requires ``pandas`` to be importable at class-definition time.
    Concrete test modules should guard with ``pytest.importorskip("pandas")``.
    """

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return arrow_table.to_pandas()

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        import pandas as pd

        series = result[column_name]
        return [None if pd.isna(v) else v for v in series.tolist()]

    def get_row_count(self, result: Any) -> int:
        return len(result)

    def get_expected_type(self) -> Any:
        import pandas as pd

        return pd.DataFrame


class DuckdbTestMixin:
    """Mixin implementing adapter methods for DuckDB.

    Requires ``duckdb`` to be importable. Concrete test modules should guard
    with ``duckdb = pytest.importorskip("duckdb")``.

    Creates ``self.conn`` in ``setup_method`` before delegating to the parent.
    """

    def setup_method(self) -> None:
        import duckdb

        self.conn = duckdb.connect()
        super().setup_method()  # type: ignore[misc]

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        return DuckdbRelation.from_arrow(self.conn, arrow_table)

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return list(result.to_arrow_table().column(column_name).to_pylist())

    def get_row_count(self, result: Any) -> int:
        return int(result.to_arrow_table().num_rows)

    def get_expected_type(self) -> Any:
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        return DuckdbRelation


class SqliteTestMixin:
    """Mixin implementing adapter methods for SQLite.

    Creates ``self.conn`` in ``setup_method`` before delegating to the parent.
    """

    def setup_method(self) -> None:
        import sqlite3

        self.conn = sqlite3.connect(":memory:")
        super().setup_method()  # type: ignore[misc]

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

        return SqliteRelation.from_arrow(self.conn, arrow_table)

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return list(result.to_arrow_table().column(column_name).to_pylist())

    def get_row_count(self, result: Any) -> int:
        return len(result)

    def get_expected_type(self) -> Any:
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

        return SqliteRelation


class PolarsLazyTestMixin:
    """Mixin implementing adapter methods for Polars LazyFrame.

    Requires ``polars`` to be importable. Concrete test modules should guard
    with ``pytest.importorskip("polars")``.
    """

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        import polars as pl

        return pl.from_arrow(arrow_table).lazy()

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        collected = result.collect()
        return list(collected[column_name].to_list())

    def get_row_count(self, result: Any) -> int:
        return int(result.collect().height)

    def get_expected_type(self) -> Any:
        import polars as pl

        return pl.LazyFrame
