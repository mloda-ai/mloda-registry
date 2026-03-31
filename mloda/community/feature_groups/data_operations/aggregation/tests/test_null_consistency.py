"""Cross-framework null consistency tests for column aggregation.

Verifies all five framework implementations (PyArrow, Pandas, Polars,
DuckDB, SQLite) produce identical results for null handling edge cases.
PyArrow is the golden reference.

Covers:
- All-null columns (score): sum/min/max/avg should produce None, count should produce 0
- Partial-null columns (value_int, value_float): aggregation should skip nulls
- Multi-null columns (amount): consistent null-skip behavior

See: https://github.com/mloda-ai/mloda-registry/issues/74
"""

from __future__ import annotations

import sqlite3
from typing import Any, Generator

import pyarrow as pa
import pytest

duckdb = pytest.importorskip("duckdb")
pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.aggregation.duckdb_aggregation import (
    DuckdbColumnAggregation,
)
from mloda.community.feature_groups.data_operations.aggregation.pandas_aggregation import (
    PandasColumnAggregation,
)
from mloda.community.feature_groups.data_operations.aggregation.polars_lazy_aggregation import (
    PolarsLazyColumnAggregation,
)
from mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation import (
    PyArrowColumnAggregation,
)
from mloda.community.feature_groups.data_operations.aggregation.sqlite_aggregation import (
    SqliteColumnAggregation,
)
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.null_consistency import (
    FrameworkAdapter,
    assert_all_frameworks_agree,
    make_feature_set,
)
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation


# ---------------------------------------------------------------------------
# Extract helpers (per-framework column extraction)
# ---------------------------------------------------------------------------


def _extract_pyarrow(result: Any, col: str) -> list[Any]:
    return list(result.column(col).to_pylist())


def _extract_pandas(result: Any, col: str) -> list[Any]:
    return [None if pd.isna(v) else v for v in result[col].tolist()]


def _extract_polars(result: Any, col: str) -> list[Any]:
    return list(result.collect()[col].to_list())


def _extract_arrow_backed(result: Any, col: str) -> list[Any]:
    return list(result.to_arrow_table().column(col).to_pylist())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def arrow_table() -> pa.Table:
    return PyArrowDataOpsTestDataCreator.create()


@pytest.fixture
def all_adapters(arrow_table: pa.Table) -> Generator[list[FrameworkAdapter], None, None]:
    """All 5 framework adapters for column aggregation."""
    conn_duckdb = duckdb.connect()
    conn_sqlite = sqlite3.connect(":memory:")

    adapters = [
        FrameworkAdapter("pyarrow", PyArrowColumnAggregation, arrow_table, _extract_pyarrow),
        FrameworkAdapter("pandas", PandasColumnAggregation, arrow_table.to_pandas(), _extract_pandas),
        FrameworkAdapter("polars", PolarsLazyColumnAggregation, pl.from_arrow(arrow_table).lazy(), _extract_polars),
        FrameworkAdapter(
            "duckdb",
            DuckdbColumnAggregation,
            DuckdbRelation.from_arrow(conn_duckdb, arrow_table),
            _extract_arrow_backed,
        ),
        FrameworkAdapter(
            "sqlite",
            SqliteColumnAggregation,
            SqliteRelation.from_arrow(conn_sqlite, arrow_table),
            _extract_arrow_backed,
        ),
    ]

    yield adapters

    conn_duckdb.close()
    conn_sqlite.close()


# ---------------------------------------------------------------------------
# Tests: all-null column (score)
# ---------------------------------------------------------------------------


class TestAllNullColumnConsistency:
    """All frameworks must agree on aggregation of an entirely null column (score)."""

    def test_sum_all_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Sum of all-null column: PyArrow/Polars/DuckDB/SQLite return None.

        NOTE: Known divergence. Pandas returns 0 (identity element convention:
        sum of zero valid values = 0, via skipna=True with min_count=0 default).
        All other frameworks return None (SQL null-propagation convention).
        """
        fs = make_feature_set("score__sum_aggr")
        non_pandas = [a for a in all_adapters if a.name != "pandas"]
        results = assert_all_frameworks_agree(non_pandas, fs, "score__sum_aggr")
        assert all(v is None for v in results["pyarrow"])
        # Verify Pandas divergence is documented, not silent
        pandas_adapter = next(a for a in all_adapters if a.name == "pandas")
        raw = pandas_adapter.calculate(fs)
        pandas_col = pandas_adapter.extract(raw, "score__sum_aggr")
        assert all(v == 0 for v in pandas_col), "Pandas sum(all-null) should return 0"

    def test_min_all_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Min of all-null column should produce None for every row."""
        fs = make_feature_set("score__min_aggr")
        results = assert_all_frameworks_agree(all_adapters, fs, "score__min_aggr")
        assert all(v is None for v in results["pyarrow"])

    def test_max_all_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Max of all-null column should produce None for every row."""
        fs = make_feature_set("score__max_aggr")
        results = assert_all_frameworks_agree(all_adapters, fs, "score__max_aggr")
        assert all(v is None for v in results["pyarrow"])

    def test_avg_all_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Avg of all-null column should produce None for every row."""
        fs = make_feature_set("score__avg_aggr")
        results = assert_all_frameworks_agree(all_adapters, fs, "score__avg_aggr")
        assert all(v is None for v in results["pyarrow"])

    def test_count_all_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Count of all-null column should produce 0 for every row."""
        fs = make_feature_set("score__count_aggr")
        results = assert_all_frameworks_agree(all_adapters, fs, "score__count_aggr")
        assert all(v == 0 for v in results["pyarrow"])


# ---------------------------------------------------------------------------
# Tests: partial-null column (value_int: 1 null at row 4)
# ---------------------------------------------------------------------------


class TestPartialNullColumnConsistency:
    """All frameworks must agree on aggregation of a column with some nulls."""

    def test_sum_partial_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """value_int has 1 null. All frameworks should skip it and agree on sum=225."""
        fs = make_feature_set("value_int__sum_aggr")
        results = assert_all_frameworks_agree(all_adapters, fs, "value_int__sum_aggr")
        assert all(v == 225 for v in results["pyarrow"])

    def test_count_partial_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """value_int has 1 null. Count of non-null values should be 11."""
        fs = make_feature_set("value_int__count_aggr")
        results = assert_all_frameworks_agree(all_adapters, fs, "value_int__count_aggr")
        assert all(v == 11 for v in results["pyarrow"])

    def test_min_partial_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """value_int has 1 null. Min should be -10."""
        fs = make_feature_set("value_int__min_aggr")
        results = assert_all_frameworks_agree(all_adapters, fs, "value_int__min_aggr")
        assert all(v == -10 for v in results["pyarrow"])

    def test_max_partial_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """value_int has 1 null. Max should be 60."""
        fs = make_feature_set("value_int__max_aggr")
        results = assert_all_frameworks_agree(all_adapters, fs, "value_int__max_aggr")
        assert all(v == 60 for v in results["pyarrow"])

    def test_avg_partial_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """value_int has 1 null. Avg should be 225/11."""
        fs = make_feature_set("value_int__avg_aggr")
        assert_all_frameworks_agree(all_adapters, fs, "value_int__avg_aggr", use_approx=True)


# ---------------------------------------------------------------------------
# Tests: multi-null column (value_float: nulls at rows 2 and 11)
# ---------------------------------------------------------------------------


class TestMultiNullColumnConsistency:
    """All frameworks must agree on aggregation of a column with multiple nulls."""

    def test_sum_multi_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """value_float has 2 nulls. All frameworks should skip them."""
        fs = make_feature_set("value_float__sum_aggr")
        assert_all_frameworks_agree(all_adapters, fs, "value_float__sum_aggr", use_approx=True)

    def test_count_multi_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """value_float has 2 nulls. Count should be 10."""
        fs = make_feature_set("value_float__count_aggr")
        results = assert_all_frameworks_agree(all_adapters, fs, "value_float__count_aggr")
        assert all(v == 10 for v in results["pyarrow"])

    def test_min_multi_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """value_float has 2 nulls. Min should be -3.14."""
        fs = make_feature_set("value_float__min_aggr")
        assert_all_frameworks_agree(all_adapters, fs, "value_float__min_aggr", use_approx=True)

    def test_max_multi_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """value_float has 2 nulls. Max should be 100.0."""
        fs = make_feature_set("value_float__max_aggr")
        assert_all_frameworks_agree(all_adapters, fs, "value_float__max_aggr", use_approx=True)

    def test_avg_multi_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """value_float has 2 nulls. Avg should use only non-null values."""
        fs = make_feature_set("value_float__avg_aggr")
        assert_all_frameworks_agree(all_adapters, fs, "value_float__avg_aggr", use_approx=True)


# ---------------------------------------------------------------------------
# Tests: amount column (nulls at rows 1 and 7)
# ---------------------------------------------------------------------------


class TestAmountNullConsistency:
    """All frameworks must agree on aggregation of amount (2 nulls at different positions)."""

    def test_sum_amount(self, all_adapters: list[FrameworkAdapter]) -> None:
        """amount has nulls at rows 1 and 7. All frameworks should skip them."""
        fs = make_feature_set("amount__sum_aggr")
        assert_all_frameworks_agree(all_adapters, fs, "amount__sum_aggr", use_approx=True)

    def test_count_amount(self, all_adapters: list[FrameworkAdapter]) -> None:
        """amount has 2 nulls. Count should be 10."""
        fs = make_feature_set("amount__count_aggr")
        results = assert_all_frameworks_agree(all_adapters, fs, "amount__count_aggr")
        assert all(v == 10 for v in results["pyarrow"])
