"""Cross-framework null consistency tests for group aggregation.

Verifies all five framework implementations (PyArrow, Pandas, Polars,
DuckDB, SQLite) produce identical results for null handling edge cases
in grouped aggregation (reduce to one row per group).

Covers:
- Null group keys (region=None forms its own group)
- Partial-null values within groups (group B has null at row 4)
- All-null column aggregation per group (score is entirely null)
- Multi-key partitions with null keys

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

from mloda.community.feature_groups.data_operations.group_aggregation.duckdb_group_aggregation import (
    DuckdbGroupAggregation,
)
from mloda.community.feature_groups.data_operations.group_aggregation.pandas_group_aggregation import (
    PandasGroupAggregation,
)
from mloda.community.feature_groups.data_operations.group_aggregation.polars_lazy_group_aggregation import (
    PolarsLazyGroupAggregation,
)
from mloda.community.feature_groups.data_operations.group_aggregation.pyarrow_group_aggregation import (
    PyArrowGroupAggregation,
)
from mloda.community.feature_groups.data_operations.group_aggregation.sqlite_group_aggregation import (
    SqliteGroupAggregation,
)
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.null_consistency import (
    FrameworkAdapter,
    assert_all_frameworks_agree_grouped,
    make_feature_set,
)
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation


# ---------------------------------------------------------------------------
# Extract helpers
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
    """All 5 framework adapters for group aggregation."""
    conn_duckdb = duckdb.connect()
    conn_sqlite = sqlite3.connect(":memory:")

    adapters = [
        FrameworkAdapter("pyarrow", PyArrowGroupAggregation, arrow_table, _extract_pyarrow),
        FrameworkAdapter("pandas", PandasGroupAggregation, arrow_table.to_pandas(), _extract_pandas),
        FrameworkAdapter("polars", PolarsLazyGroupAggregation, pl.from_arrow(arrow_table).lazy(), _extract_polars),
        FrameworkAdapter(
            "duckdb", DuckdbGroupAggregation, DuckdbRelation.from_arrow(conn_duckdb, arrow_table), _extract_arrow_backed
        ),
        FrameworkAdapter(
            "sqlite", SqliteGroupAggregation, SqliteRelation.from_arrow(conn_sqlite, arrow_table), _extract_arrow_backed
        ),
    ]

    yield adapters

    conn_duckdb.close()
    conn_sqlite.close()


# ---------------------------------------------------------------------------
# Tests: null group keys
# ---------------------------------------------------------------------------


class TestNullGroupKeyConsistency:
    """All frameworks must treat region=None as its own group."""

    def test_null_key_group_exists_sum(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Row 11 has region=None, value_int=-10. The None group sum should be -10."""
        fs = make_feature_set("value_int__sum_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "value_int__sum_grouped", "region")
        assert results["pyarrow"][None] == -10

    def test_null_key_group_exists_count(self, all_adapters: list[FrameworkAdapter]) -> None:
        """None group has 1 non-null value_int. Count should be 1."""
        fs = make_feature_set("value_int__count_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "value_int__count_grouped", "region")
        assert results["pyarrow"][None] == 1

    def test_null_key_group_exists_min(self, all_adapters: list[FrameworkAdapter]) -> None:
        """None group has value_int=-10. Min should be -10."""
        fs = make_feature_set("value_int__min_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "value_int__min_grouped", "region")
        assert results["pyarrow"][None] == -10

    def test_null_key_group_exists_max(self, all_adapters: list[FrameworkAdapter]) -> None:
        """None group has value_int=-10. Max should be -10."""
        fs = make_feature_set("value_int__max_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "value_int__max_grouped", "region")
        assert results["pyarrow"][None] == -10

    def test_group_count_includes_null_key(self, all_adapters: list[FrameworkAdapter]) -> None:
        """There should be exactly 4 groups: A, B, C, and None."""
        fs = make_feature_set("value_int__sum_grouped", context={"partition_by": ["region"]})
        for adapter in all_adapters:
            raw = adapter.calculate(fs)
            keys = adapter.extract(raw, "region")
            assert len(keys) == 4, f"{adapter.name}: expected 4 groups, got {len(keys)}"


# ---------------------------------------------------------------------------
# Tests: partial-null values within groups
# ---------------------------------------------------------------------------


class TestPartialNullWithinGroupConsistency:
    """Group B has value_int=[None, 50, 30, 60]. All frameworks should skip the null."""

    def test_group_b_sum_skips_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """B sum: 50+30+60=140 (null at row 4 skipped)."""
        fs = make_feature_set("value_int__sum_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "value_int__sum_grouped", "region")
        assert results["pyarrow"]["B"] == 140

    def test_group_b_count_excludes_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """B count: 3 non-null values (null at row 4 excluded)."""
        fs = make_feature_set("value_int__count_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "value_int__count_grouped", "region")
        assert results["pyarrow"]["B"] == 3

    def test_group_b_avg_skips_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """B avg: 140/3 (null at row 4 skipped)."""
        fs = make_feature_set("value_int__avg_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(
            all_adapters,
            fs,
            "value_int__avg_grouped",
            "region",
            use_approx=True,
        )
        assert results["pyarrow"]["B"] == pytest.approx(140.0 / 3.0, rel=1e-6)

    def test_group_b_min_skips_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """B min: 30 (null excluded)."""
        fs = make_feature_set("value_int__min_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "value_int__min_grouped", "region")
        assert results["pyarrow"]["B"] == 30

    def test_group_b_max_skips_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """B max: 60 (null excluded)."""
        fs = make_feature_set("value_int__max_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "value_int__max_grouped", "region")
        assert results["pyarrow"]["B"] == 60


# ---------------------------------------------------------------------------
# Tests: all-null column across groups
# ---------------------------------------------------------------------------


class TestAllNullColumnGroupConsistency:
    """score column is all-null. Every group should get None for sum/min/max/avg."""

    def test_all_null_sum_per_group(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Sum of all-null column per group should be None for every group."""
        fs = make_feature_set("score__sum_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "score__sum_grouped", "region")
        for key, val in results["pyarrow"].items():
            assert val is None, f"group {key!r}: expected None, got {val}"

    def test_all_null_count_per_group(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Count of all-null column per group should be 0 for every group."""
        fs = make_feature_set("score__count_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "score__count_grouped", "region")
        for key, val in results["pyarrow"].items():
            assert val == 0, f"group {key!r}: expected 0, got {val}"

    def test_all_null_min_per_group(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Min of all-null column per group should be None."""
        fs = make_feature_set("score__min_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "score__min_grouped", "region")
        for key, val in results["pyarrow"].items():
            assert val is None, f"group {key!r}: expected None, got {val}"

    def test_all_null_max_per_group(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Max of all-null column per group should be None."""
        fs = make_feature_set("score__max_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "score__max_grouped", "region")
        for key, val in results["pyarrow"].items():
            assert val is None, f"group {key!r}: expected None, got {val}"

    def test_all_null_avg_per_group(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Avg of all-null column per group should be None."""
        fs = make_feature_set("score__avg_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "score__avg_grouped", "region")
        for key, val in results["pyarrow"].items():
            assert val is None, f"group {key!r}: expected None, got {val}"


# ---------------------------------------------------------------------------
# Tests: multi-key partition with null keys
# ---------------------------------------------------------------------------


class TestMultiKeyNullConsistency:
    """Partition by [region, category] where category=None at row 6."""

    def test_multi_key_null_category_sum(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Group (B, None) should have sum=30 (row 6: value_int=30)."""
        fs = make_feature_set("value_int__sum_grouped", context={"partition_by": ["region", "category"]})
        for adapter in all_adapters:
            raw = adapter.calculate(fs)
            region_col = adapter.extract(raw, "region")
            category_col = adapter.extract(raw, "category")
            value_col = adapter.extract(raw, "value_int__sum_grouped")
            result_map = {(region_col[i], category_col[i]): value_col[i] for i in range(len(region_col))}
            assert result_map[("B", None)] == 30, (
                f"{adapter.name}: group (B, None) sum={result_map.get(('B', None))}, expected 30"
            )

    def test_multi_key_null_region_sum(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Group (None, X) should have sum=-10 (row 11: value_int=-10)."""
        fs = make_feature_set("value_int__sum_grouped", context={"partition_by": ["region", "category"]})
        for adapter in all_adapters:
            raw = adapter.calculate(fs)
            region_col = adapter.extract(raw, "region")
            category_col = adapter.extract(raw, "category")
            value_col = adapter.extract(raw, "value_int__sum_grouped")
            result_map = {(region_col[i], category_col[i]): value_col[i] for i in range(len(region_col))}
            assert result_map[(None, "X")] == -10, (
                f"{adapter.name}: group (None, X) sum={result_map.get((None, 'X'))}, expected -10"
            )

    def test_multi_key_group_with_null_value(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Group (B, X) has rows 4 (null) and 7 (60). Sum should be 60."""
        fs = make_feature_set("value_int__sum_grouped", context={"partition_by": ["region", "category"]})
        for adapter in all_adapters:
            raw = adapter.calculate(fs)
            region_col = adapter.extract(raw, "region")
            category_col = adapter.extract(raw, "category")
            value_col = adapter.extract(raw, "value_int__sum_grouped")
            result_map = {(region_col[i], category_col[i]): value_col[i] for i in range(len(region_col))}
            assert result_map[("B", "X")] == 60, (
                f"{adapter.name}: group (B, X) sum={result_map.get(('B', 'X'))}, expected 60"
            )
