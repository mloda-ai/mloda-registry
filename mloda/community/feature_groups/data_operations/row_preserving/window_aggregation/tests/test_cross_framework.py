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

from mloda.testing.data_creator import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation import (
    extract_column,
    make_feature_set,
)
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


@pytest.fixture
def arrow_table() -> pa.Table:
    """Return the shared 12-row test dataset as a PyArrow Table."""
    return PyArrowDataOpsTestDataCreator.create()


class TestCrossFrameworkComparison:
    """Compare all 6 framework implementations against PyArrow reference."""

    def _run_all_frameworks(
        self,
        arrow_table: pa.Table,
        feature_name: str,
        partition_by: list[str],
    ) -> dict[str, list[Any]]:
        """Run the given feature through all frameworks and return results keyed by name."""
        fs = make_feature_set(feature_name, partition_by)
        results: dict[str, list[Any]] = {}

        # PyArrow (reference)
        result = PyArrowWindowAggregation.calculate_feature(arrow_table, fs)
        results["pyarrow"] = extract_column(result, feature_name)

        # Pandas (accepts pa.Table)
        result = PandasWindowAggregation.calculate_feature(arrow_table, fs)
        results["pandas"] = extract_column(result, feature_name)

        # Polars Lazy (accepts pa.Table)
        result = PolarsLazyWindowAggregation.calculate_feature(arrow_table, fs)
        results["polars_lazy"] = extract_column(result, feature_name)

        # SQLite (accepts SqliteRelation)
        conn_sqlite = sqlite3.connect(":memory:")
        sqlite_data = SqliteRelation.from_arrow(conn_sqlite, arrow_table)
        result = SqliteWindowAggregation.calculate_feature(sqlite_data, fs)
        results["sqlite"] = extract_column(result, feature_name)
        conn_sqlite.close()

        # DuckDB (accepts DuckdbRelation)
        conn_duckdb = duckdb.connect()
        duckdb_data = DuckdbRelation.from_arrow(conn_duckdb, arrow_table)
        result = DuckdbWindowAggregation.calculate_feature(duckdb_data, fs)
        results["duckdb"] = extract_column(result, feature_name)
        conn_duckdb.close()

        return results

    def _assert_matches_reference(
        self,
        results: dict[str, list[Any]],
        use_approx: bool = False,
    ) -> None:
        """Assert every framework result matches the PyArrow reference."""
        reference = results["pyarrow"]
        for name, values in results.items():
            if name == "pyarrow":
                continue
            assert len(values) == len(reference), f"{name}: row count {len(values)} != reference {len(reference)}"
            if use_approx:
                for i, (ref_val, fw_val) in enumerate(zip(reference, values)):
                    if ref_val is None:
                        assert fw_val is None, f"{name} row {i}: expected None, got {fw_val}"
                    else:
                        assert fw_val == pytest.approx(ref_val, rel=1e-6), (
                            f"{name} row {i}: {fw_val} != reference {ref_val}"
                        )
            else:
                assert values == reference, f"{name} produced {values}, expected {reference}"

    def test_sum_cross_framework(self, arrow_table: pa.Table) -> None:
        """Sum of value_int partitioned by region: all frameworks must match PyArrow."""
        results = self._run_all_frameworks(arrow_table, "value_int__sum_groupby", ["region"])
        self._assert_matches_reference(results)

    def test_avg_cross_framework(self, arrow_table: pa.Table) -> None:
        """Average of value_int partitioned by region: all frameworks must match PyArrow."""
        results = self._run_all_frameworks(arrow_table, "value_int__avg_groupby", ["region"])
        self._assert_matches_reference(results, use_approx=True)

    def test_count_cross_framework(self, arrow_table: pa.Table) -> None:
        """Count of value_int partitioned by region: all frameworks must match PyArrow."""
        results = self._run_all_frameworks(arrow_table, "value_int__count_groupby", ["region"])
        self._assert_matches_reference(results)

    def test_min_cross_framework(self, arrow_table: pa.Table) -> None:
        """Minimum of value_int partitioned by region: all frameworks must match PyArrow."""
        results = self._run_all_frameworks(arrow_table, "value_int__min_groupby", ["region"])
        self._assert_matches_reference(results)

    def test_max_cross_framework(self, arrow_table: pa.Table) -> None:
        """Maximum of value_int partitioned by region: all frameworks must match PyArrow."""
        results = self._run_all_frameworks(arrow_table, "value_int__max_groupby", ["region"])
        self._assert_matches_reference(results)
