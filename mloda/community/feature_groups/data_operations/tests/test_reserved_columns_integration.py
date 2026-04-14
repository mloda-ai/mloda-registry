"""Integration tests: every guarded ``_compute_*`` raises on a colliding input column.

These tests exercise the public ``_compute_*`` entry of each implementation
that uses ``__mloda_`` helper columns. The input always carries one user
column whose name starts with the reserved prefix, and each test asserts:

1. The raised exception is a ``ValueError`` (via ``pytest.raises``).
2. The message names the reserved prefix and the colliding column.
3. The framework label and operation label are surfaced.

Operation arguments beyond the input data are valid stubs; the guard fires
before the implementation reaches the argument-validation logic, so the
specific values do not matter for the assertion. Together with the unit
tests in ``test_reserved_columns.py``, these tests prove the guard is
wired in at every entry point that adds a ``__mloda_`` helper column.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from mloda.community.feature_groups.data_operations.aggregation.polars_lazy_aggregation import PolarsLazyAggregation
from mloda.community.feature_groups.data_operations.row_preserving.binning.duckdb_binning import DuckdbBinning
from mloda.community.feature_groups.data_operations.row_preserving.binning.sqlite_binning import SqliteBinning
from mloda.community.feature_groups.data_operations.row_preserving.datetime.sqlite_datetime import (
    SqliteDateTimeExtraction,
)
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.duckdb_frame_aggregate import (
    DuckdbFrameAggregate,
)
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pandas_frame_aggregate import (
    PandasFrameAggregate,
)
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.polars_lazy_frame_aggregate import (
    PolarsLazyFrameAggregate,
)
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.sqlite_frame_aggregate import (
    SqliteFrameAggregate,
)
from mloda.community.feature_groups.data_operations.row_preserving.offset.duckdb_offset import DuckdbOffset
from mloda.community.feature_groups.data_operations.row_preserving.offset.pandas_offset import PandasOffset
from mloda.community.feature_groups.data_operations.row_preserving.offset.polars_lazy_offset import PolarsLazyOffset
from mloda.community.feature_groups.data_operations.row_preserving.offset.sqlite_offset import SqliteOffset
from mloda.community.feature_groups.data_operations.row_preserving.percentile.polars_lazy_percentile import (
    PolarsLazyPercentile,
)
from mloda.community.feature_groups.data_operations.row_preserving.rank.duckdb_rank import DuckdbRank
from mloda.community.feature_groups.data_operations.row_preserving.rank.polars_lazy_rank import PolarsLazyRank
from mloda.community.feature_groups.data_operations.row_preserving.rank.sqlite_rank import SqliteRank
from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.polars_lazy_scalar_aggregate import (
    PolarsLazyScalarAggregate,
)
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

# A user column with the reserved prefix; the literal name does not matter as
# long as it starts with ``__mloda_``.
COLLIDING = "__mloda_user_col__"

# An uppercase variant that collides only under case-insensitive identifier
# resolution (SQLite and DuckDB unquoted identifiers).
COLLIDING_UPPER = "__MLODA_USER_COL__"


def _arrow_fixture_with(colliding: str) -> pa.Table:
    """Minimal valid input that carries an arbitrary colliding user column."""
    return pa.table(
        {
            "region": ["A", "A", "B"],
            "value": [1.0, 2.0, 3.0],
            "ts": [0, 1, 2],
            colliding: [9, 9, 9],
        }
    )


def _arrow_fixture() -> pa.Table:
    """Minimal valid input that also carries a colliding user column."""
    return _arrow_fixture_with(COLLIDING)


def _pandas_fixture() -> pd.DataFrame:
    return _arrow_fixture().to_pandas()


def _polars_fixture() -> pl.LazyFrame:
    df = pl.from_arrow(_arrow_fixture())
    assert isinstance(df, pl.DataFrame)
    return df.lazy()


def _duckdb_fixture(conn: Any) -> Any:
    from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

    return DuckdbRelation.from_arrow(conn, _arrow_fixture())


def _sqlite_fixture(conn: Any) -> Any:
    from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

    return SqliteRelation.from_arrow(conn, _arrow_fixture())


def _sqlite_fixture_upper(conn: Any) -> Any:
    """SQLite relation carrying an uppercase colliding user column."""
    from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

    return SqliteRelation.from_arrow(conn, _arrow_fixture_with(COLLIDING_UPPER))


@pytest.fixture
def duckdb_conn() -> Any:
    duckdb = pytest.importorskip("duckdb")
    return duckdb.connect()


@pytest.fixture
def sqlite_conn() -> Any:
    import sqlite3

    return sqlite3.connect(":memory:")


# ----------------------------------------------------------------------
# Pandas
# ----------------------------------------------------------------------


class TestPandasGuards:
    def test_frame_aggregate(self) -> None:
        with pytest.raises(ValueError, match=r"Pandas frame aggregate.*'__mloda_user_col__'"):
            PandasFrameAggregate._compute_frame(
                _pandas_fixture(), "value__sum_rolling_3", "value", ["region"], "ts", "sum", "rolling", 3
            )

    def test_window_aggregation(self) -> None:
        with pytest.raises(ValueError, match=r"Pandas window aggregation.*'__mloda_user_col__'"):
            PandasWindowAggregation._compute_window(_pandas_fixture(), "value__sum_agg", "value", ["region"], "sum")

    def test_offset(self) -> None:
        with pytest.raises(ValueError, match=r"Pandas offset.*'__mloda_user_col__'"):
            PandasOffset._compute_offset(_pandas_fixture(), "value__lag_1", "value", ["region"], "ts", "lag_1")


# ----------------------------------------------------------------------
# Polars Lazy
# ----------------------------------------------------------------------


class TestPolarsLazyGuards:
    def test_frame_aggregate(self) -> None:
        with pytest.raises(ValueError, match=r"Polars frame aggregate.*'__mloda_user_col__'"):
            PolarsLazyFrameAggregate._compute_frame(
                _polars_fixture(), "value__sum_rolling_3", "value", ["region"], "ts", "sum", "rolling", 3
            )

    def test_window_aggregation(self) -> None:
        with pytest.raises(ValueError, match=r"Polars window aggregation.*'__mloda_user_col__'"):
            PolarsLazyWindowAggregation._compute_window(_polars_fixture(), "value__sum_agg", "value", ["region"], "sum")

    def test_offset(self) -> None:
        with pytest.raises(ValueError, match=r"Polars offset.*'__mloda_user_col__'"):
            PolarsLazyOffset._compute_offset(_polars_fixture(), "value__lag_1", "value", ["region"], "ts", "lag_1")

    def test_rank(self) -> None:
        with pytest.raises(ValueError, match=r"Polars rank.*'__mloda_user_col__'"):
            PolarsLazyRank._compute_rank(_polars_fixture(), "value__row_number", ["region"], "ts", "row_number")

    def test_aggregation(self) -> None:
        with pytest.raises(ValueError, match=r"Polars aggregation.*'__mloda_user_col__'"):
            PolarsLazyAggregation._compute_group(_polars_fixture(), "value__sum_agg", "value", ["region"], "sum")

    def test_scalar_aggregate(self) -> None:
        with pytest.raises(ValueError, match=r"Polars scalar aggregate.*'__mloda_user_col__'"):
            PolarsLazyScalarAggregate._compute_aggregation(_polars_fixture(), "value__sum_scalar", "value", "sum")

    def test_percentile(self) -> None:
        with pytest.raises(ValueError, match=r"Polars percentile.*'__mloda_user_col__'"):
            PolarsLazyPercentile._compute_percentile(_polars_fixture(), "value__pctl_50", "value", ["region"], 0.5)


# ----------------------------------------------------------------------
# DuckDB
# ----------------------------------------------------------------------


class TestDuckDBGuards:
    def test_frame_aggregate(self, duckdb_conn: Any) -> None:
        with pytest.raises(ValueError, match=r"DuckDB frame aggregate.*'__mloda_user_col__'"):
            DuckdbFrameAggregate._compute_frame(
                _duckdb_fixture(duckdb_conn),
                "value__sum_rolling_3",
                "value",
                ["region"],
                "ts",
                "sum",
                "rolling",
                3,
            )

    def test_window_aggregation(self, duckdb_conn: Any) -> None:
        with pytest.raises(ValueError, match=r"DuckDB window aggregation.*'__mloda_user_col__'"):
            DuckdbWindowAggregation._compute_window(
                _duckdb_fixture(duckdb_conn), "value__sum_agg", "value", ["region"], "sum"
            )

    def test_offset(self, duckdb_conn: Any) -> None:
        with pytest.raises(ValueError, match=r"DuckDB offset.*'__mloda_user_col__'"):
            DuckdbOffset._compute_offset(
                _duckdb_fixture(duckdb_conn), "value__lag_1", "value", ["region"], "ts", "lag_1"
            )

    def test_rank(self, duckdb_conn: Any) -> None:
        with pytest.raises(ValueError, match=r"DuckDB rank.*'__mloda_user_col__'"):
            DuckdbRank._compute_rank(_duckdb_fixture(duckdb_conn), "value__row_number", ["region"], "ts", "row_number")

    def test_binning(self, duckdb_conn: Any) -> None:
        with pytest.raises(ValueError, match=r"DuckDB binning.*'__mloda_user_col__'"):
            DuckdbBinning._compute_binning(_duckdb_fixture(duckdb_conn), "value__bin_3", "value", "bin", 3)


# ----------------------------------------------------------------------
# SQLite
# ----------------------------------------------------------------------


class TestSQLiteGuards:
    def test_frame_aggregate(self, sqlite_conn: Any) -> None:
        with pytest.raises(ValueError, match=r"SQLite frame aggregate.*'__mloda_user_col__'"):
            SqliteFrameAggregate._compute_frame(
                _sqlite_fixture(sqlite_conn),
                "value__sum_rolling_3",
                "value",
                ["region"],
                "ts",
                "sum",
                "rolling",
                3,
            )

    def test_window_aggregation(self, sqlite_conn: Any) -> None:
        with pytest.raises(ValueError, match=r"SQLite window aggregation.*'__mloda_user_col__'"):
            SqliteWindowAggregation._compute_window(
                _sqlite_fixture(sqlite_conn), "value__sum_agg", "value", ["region"], "sum"
            )

    def test_offset(self, sqlite_conn: Any) -> None:
        with pytest.raises(ValueError, match=r"SQLite offset.*'__mloda_user_col__'"):
            SqliteOffset._compute_offset(
                _sqlite_fixture(sqlite_conn), "value__lag_1", "value", ["region"], "ts", "lag_1"
            )

    def test_rank(self, sqlite_conn: Any) -> None:
        with pytest.raises(ValueError, match=r"SQLite rank.*'__mloda_user_col__'"):
            SqliteRank._compute_rank(_sqlite_fixture(sqlite_conn), "value__row_number", ["region"], "ts", "row_number")

    def test_binning(self, sqlite_conn: Any) -> None:
        with pytest.raises(ValueError, match=r"SQLite binning.*'__mloda_user_col__'"):
            SqliteBinning._compute_binning(_sqlite_fixture(sqlite_conn), "value__bin_3", "value", "bin", 3)

    def test_datetime(self, sqlite_conn: Any) -> None:
        with pytest.raises(ValueError, match=r"SQLite datetime.*'__mloda_user_col__'"):
            SqliteDateTimeExtraction._compute_datetime(_sqlite_fixture(sqlite_conn), "ts__year", "ts", "year")

    def test_string(self, sqlite_conn: Any) -> None:
        from mloda.community.feature_groups.data_operations.string.sqlite_string import SqliteStringOps

        with pytest.raises(ValueError, match=r"SQLite string.*'__mloda_user_col__'"):
            SqliteStringOps._compute_string(_sqlite_fixture(sqlite_conn), "region__trim", "region", "trim")

    def test_window_aggregation_uppercase_collision(self, sqlite_conn: Any) -> None:
        """Uppercase user columns collide with SQLite's ``__mloda_rn__`` helper
        because unquoted identifiers are case-insensitive in SQLite."""
        with pytest.raises(ValueError, match=r"SQLite window aggregation.*'__MLODA_USER_COL__'"):
            SqliteWindowAggregation._compute_window(
                _sqlite_fixture_upper(sqlite_conn), "value__sum_agg", "value", ["region"], "sum"
            )


# ----------------------------------------------------------------------
# PyArrow
# ----------------------------------------------------------------------


class TestPyArrowGuards:
    def test_window_aggregation(self) -> None:
        with pytest.raises(ValueError, match=r"PyArrow window aggregation.*'__mloda_user_col__'"):
            PyArrowWindowAggregation._compute_window(_arrow_fixture(), "value__sum_agg", "value", ["region"], "sum")
