"""Cross-framework comparison tests for window aggregation implementations.

Runs each aggregation through all available framework implementations and asserts
that every framework produces the same results as the PyArrow reference.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pyarrow as pa
import pytest

pytest.importorskip("pandas")
pytest.importorskip("polars")
duckdb = pytest.importorskip("duckdb")

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.duckdb_window_aggregation import (
    DuckdbWindowAggregation,
)
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pandas_window_aggregation import (
    PandasWindowAggregation,
)
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.polars_lazy_window_aggregation import (
    PolarsLazyWindowAggregation,
)
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
    PyArrowWindowAggregation,
)
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.sqlite_window_aggregation import (
    SqliteWindowAggregation,
)
from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation import (
    CrossFrameworkComparisonBase,
    FrameworkConfig,
)


class TestCrossFrameworkComparison(CrossFrameworkComparisonBase):
    """Compare all 5 framework implementations against PyArrow reference."""

    def setup_method(self) -> None:
        self.conn_sqlite = sqlite3.connect(":memory:")
        self.conn_duckdb = duckdb.connect()
        super().setup_method()

    def teardown_method(self) -> None:
        self.conn_sqlite.close()
        self.conn_duckdb.close()

    @classmethod
    def pyarrow_implementation_class(cls) -> Any:
        return PyArrowWindowAggregation

    def get_frameworks(self) -> list[FrameworkConfig]:
        return [
            FrameworkConfig("pandas", PandasWindowAggregation),
            FrameworkConfig("polars_lazy", PolarsLazyWindowAggregation),
            FrameworkConfig(
                "sqlite",
                SqliteWindowAggregation,
                lambda t: SqliteRelation.from_arrow(self.conn_sqlite, t),
            ),
            FrameworkConfig(
                "duckdb",
                DuckdbWindowAggregation,
                lambda t: DuckdbRelation.from_arrow(self.conn_duckdb, t),
            ),
        ]
