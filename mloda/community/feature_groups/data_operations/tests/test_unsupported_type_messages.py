"""End-to-end checks that every ``_compute_*`` implementation produces an
unsupported-type error whose message lists the concrete set of valid types.

These tests guard against regressions of the user-guidance contract
introduced for audit item #14: an unsupported ``agg_type`` (or
``frame_type``) must raise ``ValueError`` whose message *both* shows the
rejected value *and* enumerates the types the caller could have used.

The tests import only what the specific framework needs and skip when
that framework's optional dependency is missing, so running the suite
in a slim environment still covers everything that is installable.
"""

from __future__ import annotations

import re
import sqlite3

import pytest


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _assert_valid_error(err: BaseException, agg_type: str, *expected_supported: str) -> None:
    """Assert the raised error follows the contract shared across frameworks."""
    msg = str(err)
    assert repr(agg_type) in msg, f"rejected value {agg_type!r} missing from {msg!r}"
    assert "Supported types:" in msg, f"missing 'Supported types:' prefix in {msg!r}"
    for value in expected_supported:
        assert re.search(rf"\b{re.escape(value)}\b", msg), f"{value!r} missing from {msg!r}"


# ---------------------------------------------------------------------------
# Aggregation (group-by)
# ---------------------------------------------------------------------------


class TestAggregationErrors:
    def test_pandas_aggregation_error(self) -> None:
        pd = pytest.importorskip("pandas")
        from mloda.community.feature_groups.data_operations.aggregation.pandas_aggregation import (
            PandasAggregation,
        )

        df = pd.DataFrame({"region": ["a", "b"], "val": [1, 2]})
        with pytest.raises(ValueError) as exc:
            PandasAggregation._compute_group(df, "f", "val", ["region"], "not_a_real_agg")
        _assert_valid_error(exc.value, "not_a_real_agg", "sum", "count", "mean")
        assert "for Pandas" in str(exc.value)

    def test_pyarrow_aggregation_error(self) -> None:
        pa = pytest.importorskip("pyarrow")
        from mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation import (
            PyArrowAggregation,
        )

        tbl = pa.table({"region": ["a", "b"], "val": [1, 2]})
        with pytest.raises(ValueError) as exc:
            PyArrowAggregation._compute_group(tbl, "f", "val", ["region"], "not_a_real_agg")
        _assert_valid_error(exc.value, "not_a_real_agg", "sum", "count", "first")
        assert "for PyArrow" in str(exc.value)

    def test_duckdb_aggregation_error(self) -> None:
        duckdb = pytest.importorskip("duckdb")
        pa = pytest.importorskip("pyarrow")
        from mloda.community.feature_groups.data_operations.aggregation.duckdb_aggregation import (
            DuckdbAggregation,
        )
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import (
            DuckdbRelation,
        )

        arrow = pa.table({"region": ["a", "b"], "val": [1, 2]})
        conn = duckdb.connect(":memory:")
        try:
            rel = DuckdbRelation.from_arrow(conn, arrow)
            with pytest.raises(ValueError) as exc:
                DuckdbAggregation._compute_group(rel, "f", "val", ["region"], "not_a_real_agg")
            _assert_valid_error(exc.value, "not_a_real_agg", "sum", "median", "mode")
            assert "for DuckDB" in str(exc.value)
        finally:
            conn.close()

    def test_sqlite_aggregation_error(self) -> None:
        pa = pytest.importorskip("pyarrow")
        from mloda.community.feature_groups.data_operations.aggregation.sqlite_aggregation import (
            SqliteAggregation,
        )
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import (
            SqliteRelation,
        )

        conn = sqlite3.connect(":memory:")
        try:
            arrow = pa.table({"region": ["a", "b"], "val": [1, 2]})
            rel = SqliteRelation.from_arrow(conn, arrow)
            with pytest.raises(ValueError) as exc:
                SqliteAggregation._compute_group(rel, "f", "val", ["region"], "median")
            _assert_valid_error(exc.value, "median", "sum", "avg", "count", "min", "max")
            assert "for SQLite" in str(exc.value)
        finally:
            conn.close()

    def test_polars_lazy_aggregation_error(self) -> None:
        pl = pytest.importorskip("polars")
        from mloda.community.feature_groups.data_operations.aggregation.polars_lazy_aggregation import (
            PolarsLazyAggregation,
        )

        lf = pl.LazyFrame({"region": ["a", "b"], "val": [1, 2]})
        with pytest.raises(ValueError) as exc:
            PolarsLazyAggregation._compute_group(lf, "f", "val", ["region"], "not_a_real_agg")
        _assert_valid_error(exc.value, "not_a_real_agg", "sum", "count", "first")
        assert "for Polars" in str(exc.value)


# ---------------------------------------------------------------------------
# Scalar aggregate
# ---------------------------------------------------------------------------


class TestScalarAggregateErrors:
    def test_pandas_scalar_aggregate_error(self) -> None:
        pd = pytest.importorskip("pandas")
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.pandas_scalar_aggregate import (
            PandasScalarAggregate,
        )

        df = pd.DataFrame({"val": [1, 2, 3]})
        with pytest.raises(ValueError) as exc:
            PandasScalarAggregate._compute_aggregation(df, "f", "val", "not_a_real_agg")
        _assert_valid_error(exc.value, "not_a_real_agg", "sum", "median", "std_samp")
        assert "for Pandas" in str(exc.value)

    def test_pyarrow_scalar_aggregate_error(self) -> None:
        pa = pytest.importorskip("pyarrow")
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.pyarrow_scalar_aggregate import (
            PyArrowScalarAggregate,
        )

        tbl = pa.table({"val": [1, 2, 3]})
        with pytest.raises(ValueError) as exc:
            PyArrowScalarAggregate._compute_aggregation(tbl, "f", "val", "not_a_real_agg")
        _assert_valid_error(exc.value, "not_a_real_agg", "sum", "median", "std_samp")
        assert "for PyArrow" in str(exc.value)

    def test_sqlite_scalar_aggregate_error(self) -> None:
        pa = pytest.importorskip("pyarrow")
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.sqlite_scalar_aggregate import (
            SqliteScalarAggregate,
        )
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import (
            SqliteRelation,
        )

        conn = sqlite3.connect(":memory:")
        try:
            arrow = pa.table({"val": [1, 2, 3]})
            rel = SqliteRelation.from_arrow(conn, arrow)
            with pytest.raises(ValueError) as exc:
                SqliteScalarAggregate._compute_aggregation(rel, "f", "val", "median")
            _assert_valid_error(exc.value, "median", "sum", "avg", "count")
            assert "for SQLite" in str(exc.value)
        finally:
            conn.close()

    def test_duckdb_scalar_aggregate_error(self) -> None:
        duckdb = pytest.importorskip("duckdb")
        pa = pytest.importorskip("pyarrow")
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.duckdb_scalar_aggregate import (
            DuckdbScalarAggregate,
        )
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import (
            DuckdbRelation,
        )

        arrow = pa.table({"val": [1, 2, 3]})
        conn = duckdb.connect(":memory:")
        try:
            rel = DuckdbRelation.from_arrow(conn, arrow)
            with pytest.raises(ValueError) as exc:
                DuckdbScalarAggregate._compute_aggregation(rel, "f", "val", "not_a_real_agg")
            _assert_valid_error(exc.value, "not_a_real_agg", "sum", "median")
            assert "for DuckDB" in str(exc.value)
        finally:
            conn.close()

    def test_polars_lazy_scalar_aggregate_error(self) -> None:
        pl = pytest.importorskip("polars")
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.polars_lazy_scalar_aggregate import (
            PolarsLazyScalarAggregate,
        )

        lf = pl.LazyFrame({"val": [1, 2, 3]})
        with pytest.raises(ValueError) as exc:
            PolarsLazyScalarAggregate._compute_aggregation(lf, "f", "val", "not_a_real_agg")
        _assert_valid_error(exc.value, "not_a_real_agg", "sum", "median")
        assert "for Polars" in str(exc.value)


# ---------------------------------------------------------------------------
# Window aggregation
# ---------------------------------------------------------------------------


class TestWindowAggregationErrors:
    def test_pandas_window_error(self) -> None:
        pd = pytest.importorskip("pandas")
        from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pandas_window_aggregation import (
            PandasWindowAggregation,
        )

        df = pd.DataFrame({"region": ["a", "b"], "val": [1, 2]})
        with pytest.raises(ValueError) as exc:
            PandasWindowAggregation._compute_window(df, "f", "val", ["region"], "not_a_real_agg")
        _assert_valid_error(exc.value, "not_a_real_agg", "sum", "count")
        assert "for Pandas" in str(exc.value)

    def test_pyarrow_window_error(self) -> None:
        pa = pytest.importorskip("pyarrow")
        from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
            PyArrowWindowAggregation,
        )

        tbl = pa.table({"region": ["a", "b"], "val": [1, 2]})
        with pytest.raises(ValueError) as exc:
            PyArrowWindowAggregation._compute_window(tbl, "f", "val", ["region"], "not_a_real_agg")
        _assert_valid_error(exc.value, "not_a_real_agg", "sum", "first")
        assert "for PyArrow" in str(exc.value)

    def test_sqlite_window_error(self) -> None:
        pa = pytest.importorskip("pyarrow")
        from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.sqlite_window_aggregation import (
            SqliteWindowAggregation,
        )
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import (
            SqliteRelation,
        )

        conn = sqlite3.connect(":memory:")
        try:
            arrow = pa.table({"region": ["a", "b"], "val": [1, 2]})
            rel = SqliteRelation.from_arrow(conn, arrow)
            with pytest.raises(ValueError) as exc:
                SqliteWindowAggregation._compute_window(rel, "f", "val", ["region"], "median")
            _assert_valid_error(exc.value, "median", "sum", "avg", "count")
            assert "for SQLite" in str(exc.value)
        finally:
            conn.close()

    def test_polars_lazy_window_error(self) -> None:
        pl = pytest.importorskip("polars")
        from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.polars_lazy_window_aggregation import (
            PolarsLazyWindowAggregation,
        )

        lf = pl.LazyFrame({"region": ["a", "b"], "val": [1, 2]})
        with pytest.raises(ValueError) as exc:
            PolarsLazyWindowAggregation._compute_window(lf, "f", "val", ["region"], "not_a_real_agg")
        _assert_valid_error(exc.value, "not_a_real_agg", "sum", "first")
        assert "for Polars" in str(exc.value)


# ---------------------------------------------------------------------------
# Frame aggregate (agg_type AND frame_type errors)
# ---------------------------------------------------------------------------


class TestFrameAggregateErrors:
    def test_pandas_frame_agg_type_error(self) -> None:
        pd = pytest.importorskip("pandas")
        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pandas_frame_aggregate import (
            PandasFrameAggregate,
        )

        df = pd.DataFrame({"region": ["a", "b"], "val": [1, 2], "ts": [1, 2]})
        with pytest.raises(ValueError) as exc:
            PandasFrameAggregate._compute_frame(
                df, "f", "val", ["region"], "ts", "not_a_real_agg", "rolling", frame_size=2
            )
        _assert_valid_error(exc.value, "not_a_real_agg", "sum", "std", "median")
        assert "for Pandas frame aggregate" in str(exc.value)

    def test_pandas_frame_type_error(self) -> None:
        pd = pytest.importorskip("pandas")
        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pandas_frame_aggregate import (
            PandasFrameAggregate,
        )

        df = pd.DataFrame({"region": ["a", "b"], "val": [1, 2], "ts": [1, 2]})
        with pytest.raises(ValueError) as exc:
            PandasFrameAggregate._compute_frame(df, "f", "val", ["region"], "ts", "sum", "not_a_real_frame")
        _assert_valid_error(exc.value, "not_a_real_frame", "rolling", "cumulative", "expanding")
        assert "Unsupported frame type for Pandas" in str(exc.value)

    def test_sqlite_frame_agg_type_error(self) -> None:
        pa = pytest.importorskip("pyarrow")
        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.sqlite_frame_aggregate import (
            SqliteFrameAggregate,
        )
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import (
            SqliteRelation,
        )

        conn = sqlite3.connect(":memory:")
        try:
            arrow = pa.table({"region": ["a", "b"], "val": [1, 2], "ts": [1, 2]})
            rel = SqliteRelation.from_arrow(conn, arrow)
            with pytest.raises(ValueError) as exc:
                SqliteFrameAggregate._compute_frame(
                    rel, "f", "val", ["region"], "ts", "median", "rolling", frame_size=2
                )
            _assert_valid_error(exc.value, "median", "sum", "avg", "count")
            assert "for SQLite frame aggregate" in str(exc.value)
        finally:
            conn.close()

    def test_duckdb_frame_agg_type_error(self) -> None:
        duckdb = pytest.importorskip("duckdb")
        pa = pytest.importorskip("pyarrow")
        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.duckdb_frame_aggregate import (
            DuckdbFrameAggregate,
        )
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import (
            DuckdbRelation,
        )

        arrow = pa.table({"region": ["a", "b"], "val": [1, 2], "ts": [1, 2]})
        conn = duckdb.connect(":memory:")
        try:
            rel = DuckdbRelation.from_arrow(conn, arrow)
            with pytest.raises(ValueError) as exc:
                DuckdbFrameAggregate._compute_frame(
                    rel, "f", "val", ["region"], "ts", "not_a_real_agg", "rolling", frame_size=2
                )
            _assert_valid_error(exc.value, "not_a_real_agg", "sum", "median", "std")
            assert "for DuckDB frame aggregate" in str(exc.value)
        finally:
            conn.close()

    def test_polars_lazy_frame_cumulative_error(self) -> None:
        pl = pytest.importorskip("polars")
        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.polars_lazy_frame_aggregate import (
            PolarsLazyFrameAggregate,
        )

        lf = pl.LazyFrame({"region": ["a", "b"], "val": [1, 2], "ts": [1, 2]})
        with pytest.raises(ValueError) as exc:
            PolarsLazyFrameAggregate._compute_frame(lf, "f", "val", ["region"], "ts", "median", "cumulative")
        _assert_valid_error(exc.value, "median", "sum", "avg")
        assert "for Polars cumulative/expanding" in str(exc.value)

    def test_polars_lazy_frame_rolling_error(self) -> None:
        pl = pytest.importorskip("polars")
        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.polars_lazy_frame_aggregate import (
            PolarsLazyFrameAggregate,
        )

        lf = pl.LazyFrame({"region": ["a", "b"], "val": [1, 2], "ts": [1, 2]})
        with pytest.raises(ValueError) as exc:
            PolarsLazyFrameAggregate._compute_frame(lf, "f", "val", ["region"], "ts", "mode", "rolling", frame_size=2)
        _assert_valid_error(exc.value, "mode", "sum", "std")
        assert "for Polars rolling" in str(exc.value)


# ---------------------------------------------------------------------------
# Test reference implementations
# ---------------------------------------------------------------------------


class TestReferenceAggregationHelperError:
    def test_reference_aggregation_helper_error(self) -> None:
        from mloda.testing.feature_groups.data_operations.aggregation_helpers import aggregate

        with pytest.raises(ValueError) as exc:
            aggregate([1, 2, 3], "not_a_real_agg")
        _assert_valid_error(exc.value, "not_a_real_agg", "sum", "median", "mode")


def test_error_messages_use_single_quoted_repr() -> None:
    """All messages quote the rejected value via !r for robustness.

    This regression-guards the contract: if a caller ever refactors the
    helper to use ``str()`` instead of ``repr()``, empty-string and
    whitespace values would vanish silently.
    """
    from mloda.community.feature_groups.data_operations.errors import (
        unsupported_agg_type_error,
    )

    err = unsupported_agg_type_error(" ", {"sum"})
    assert "' '" in str(err)
