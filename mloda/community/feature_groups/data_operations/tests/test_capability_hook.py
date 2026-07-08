"""Per-op, per-framework capability checks via ``supports_compute_framework`` (issue #247).

mloda core 0.9.0 evaluates ``FeatureGroup.supports_compute_framework(feature_name, options,
compute_framework)`` per feature at match time. The data_operations backends must override
it so that operations a backend cannot compute (e.g. ``median`` on SQLite) are rejected at
match time instead of failing later inside ``calculate_feature``. Backends stay conservative:
anything they cannot parse into an operation keeps the default ``True``.

The tests import only what the specific framework needs and skip when that framework's
optional dependency is missing.
"""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.options import Options


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _config_aggregation_options(agg_type: str) -> Options:
    """Build config-based aggregation options carrying the aggregation_type discriminator."""
    return Options(
        context={
            "aggregation_type": agg_type,
            "in_features": "value",
            "partition_by": ["region"],
        }
    )


def _time_frame_options(frame_unit: str) -> Options:
    """Build config-based time-frame options for the given frame_unit."""
    return Options(
        context={
            "aggregation_type": "sum",
            "frame_type": "time",
            "frame_size": 3,
            "frame_unit": frame_unit,
            "in_features": "value",
            "partition_by": ["region"],
            "order_by": "ts",
        }
    )


# ---------------------------------------------------------------------------
# Aggregation (group-by)
# ---------------------------------------------------------------------------


class TestAggregationCapability:
    def test_sqlite_rejects_median(self) -> None:
        """SQLite cannot compute median natively, so the hook must reject it."""
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

        from mloda.community.feature_groups.data_operations.aggregation.sqlite_aggregation import (
            SqliteAggregation,
        )

        result = SqliteAggregation.supports_compute_framework("value__median_agg", Options(), SqliteFramework)
        assert result is False

    @pytest.mark.parametrize("feature_name", ["value__sum_agg", "value__mean_agg"])
    def test_sqlite_accepts_supported_types(self, feature_name: str) -> None:
        """SQLite supports sum and mean (avg alias in _SQLITE_AGG_FUNCS), so the hook allows them."""
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

        from mloda.community.feature_groups.data_operations.aggregation.sqlite_aggregation import (
            SqliteAggregation,
        )

        result = SqliteAggregation.supports_compute_framework(feature_name, Options(), SqliteFramework)
        assert result is True

    @pytest.mark.parametrize("feature_name", ["value__median_agg", "value__mode_agg"])
    def test_pyarrow_rejects_unsupported_types(self, feature_name: str) -> None:
        """PyArrow group_by has no median or mode kernel, so the hook must reject them."""
        pytest.importorskip("pyarrow")
        from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

        from mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation import (
            PyArrowAggregation,
        )

        result = PyArrowAggregation.supports_compute_framework(feature_name, Options(), PyArrowTable)
        assert result is False

    def test_pyarrow_accepts_nunique(self) -> None:
        """PyArrow supports nunique via count_distinct, so the hook allows it."""
        pytest.importorskip("pyarrow")
        from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

        from mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation import (
            PyArrowAggregation,
        )

        result = PyArrowAggregation.supports_compute_framework("value__nunique_agg", Options(), PyArrowTable)
        assert result is True

    @pytest.mark.parametrize("feature_name", ["value__median_agg", "value__mode_agg"])
    def test_pandas_is_unrestricted(self, feature_name: str) -> None:
        """Pandas supports the full aggregation-type table, so the hook allows everything."""
        pytest.importorskip("pandas")
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        from mloda.community.feature_groups.data_operations.aggregation.pandas_aggregation import (
            PandasAggregation,
        )

        result = PandasAggregation.supports_compute_framework(feature_name, Options(), PandasDataFrame)
        assert result is True

    def test_sqlite_rejects_config_based_median(self) -> None:
        """Config-based features carry the agg type in options; SQLite must reject median there too."""
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

        from mloda.community.feature_groups.data_operations.aggregation.sqlite_aggregation import (
            SqliteAggregation,
        )

        options = _config_aggregation_options("median")
        result = SqliteAggregation.supports_compute_framework("median_result", options, SqliteFramework)
        assert result is False

    def test_sqlite_conservative_default_for_unparsable_name(self) -> None:
        """A name encoding no parsable agg type keeps the default True (conservative)."""
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

        from mloda.community.feature_groups.data_operations.aggregation.sqlite_aggregation import (
            SqliteAggregation,
        )

        result = SqliteAggregation.supports_compute_framework("totally_unrelated", Options(), SqliteFramework)
        assert result is True


# ---------------------------------------------------------------------------
# Window aggregation
# ---------------------------------------------------------------------------


class TestWindowAggregationCapability:
    def test_sqlite_rejects_median_window(self) -> None:
        """SQLite window functions cannot compute median, so the hook must reject it."""
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

        from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.sqlite_window_aggregation import (
            SqliteWindowAggregation,
        )

        result = SqliteWindowAggregation.supports_compute_framework("value__median_window", Options(), SqliteFramework)
        assert result is False

    def test_pyarrow_rejects_mode_window(self) -> None:
        """PyArrow window aggregation has no mode kernel, so the hook must reject it."""
        pytest.importorskip("pyarrow")
        from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

        from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
            PyArrowWindowAggregation,
        )

        result = PyArrowWindowAggregation.supports_compute_framework("value__mode_window", Options(), PyArrowTable)
        assert result is False

    def test_duckdb_accepts_median_window(self) -> None:
        """DuckDB computes median in window frames, so the hook allows it."""
        pytest.importorskip("duckdb")
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework

        from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.duckdb_window_aggregation import (
            DuckdbWindowAggregation,
        )

        result = DuckdbWindowAggregation.supports_compute_framework("value__median_window", Options(), DuckDBFramework)
        assert result is True


# ---------------------------------------------------------------------------
# Scalar aggregate
# ---------------------------------------------------------------------------


class TestScalarAggregateCapability:
    def test_sqlite_rejects_median_scalar(self) -> None:
        """SQLite cannot compute a median scalar, so the hook must reject it."""
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.sqlite_scalar_aggregate import (
            SqliteScalarAggregate,
        )

        result = SqliteScalarAggregate.supports_compute_framework("value__median_scalar", Options(), SqliteFramework)
        assert result is False

    def test_sqlite_accepts_sum_scalar(self) -> None:
        """SQLite supports sum, so the hook allows it."""
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.sqlite_scalar_aggregate import (
            SqliteScalarAggregate,
        )

        result = SqliteScalarAggregate.supports_compute_framework("value__sum_scalar", Options(), SqliteFramework)
        assert result is True

    def test_pandas_accepts_median_scalar(self) -> None:
        """Pandas computes median natively, so the hook allows it."""
        pytest.importorskip("pandas")
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.pandas_scalar_aggregate import (
            PandasScalarAggregate,
        )

        result = PandasScalarAggregate.supports_compute_framework("value__median_scalar", Options(), PandasDataFrame)
        assert result is True


# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------


class TestRankCapability:
    def test_pandas_rejects_percent_rank(self) -> None:
        """Pandas percent_rank diverges from SQL semantics, so the hook must reject it."""
        pytest.importorskip("pandas")
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        from mloda.community.feature_groups.data_operations.row_preserving.rank.pandas_rank import (
            PandasRank,
        )

        result = PandasRank.supports_compute_framework("value__percent_rank_ranked", Options(), PandasDataFrame)
        assert result is False

    def test_pandas_accepts_dense_rank(self) -> None:
        """Pandas supports dense_rank, so the hook allows it."""
        pytest.importorskip("pandas")
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        from mloda.community.feature_groups.data_operations.row_preserving.rank.pandas_rank import (
            PandasRank,
        )

        result = PandasRank.supports_compute_framework("value__dense_rank_ranked", Options(), PandasDataFrame)
        assert result is True

    def test_duckdb_accepts_percent_rank(self) -> None:
        """DuckDB supports SQL percent_rank natively, so the hook allows it."""
        pytest.importorskip("duckdb")
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework

        from mloda.community.feature_groups.data_operations.row_preserving.rank.duckdb_rank import (
            DuckdbRank,
        )

        result = DuckdbRank.supports_compute_framework("value__percent_rank_ranked", Options(), DuckDBFramework)
        assert result is True


# ---------------------------------------------------------------------------
# Frame aggregate (frame_type / time_unit capability, not agg types)
# ---------------------------------------------------------------------------


class TestFrameAggregateCapability:
    def test_pandas_rejects_month_time_unit(self) -> None:
        """Pandas SUPPORTED_TIME_UNITS excludes month, so the hook must reject a month time frame."""
        pytest.importorskip("pandas")
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pandas_frame_aggregate import (
            PandasFrameAggregate,
        )

        options = _time_frame_options("month")
        result = PandasFrameAggregate.supports_compute_framework("value_time_frame", options, PandasDataFrame)
        assert result is False

    def test_pandas_accepts_day_time_unit(self) -> None:
        """Pandas supports day time frames, so the hook allows the same probe with unit day."""
        pytest.importorskip("pandas")
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pandas_frame_aggregate import (
            PandasFrameAggregate,
        )

        options = _time_frame_options("day")
        result = PandasFrameAggregate.supports_compute_framework("value_time_frame", options, PandasDataFrame)
        assert result is True

    def test_pandas_accepts_rolling_name(self) -> None:
        """Pandas supports rolling row-count frames, so the hook allows the string-based name."""
        pytest.importorskip("pandas")
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pandas_frame_aggregate import (
            PandasFrameAggregate,
        )

        result = PandasFrameAggregate.supports_compute_framework("value__sum_rolling_3", Options(), PandasDataFrame)
        assert result is True

    def test_duckdb_accepts_month_time_unit(self) -> None:
        """DuckDB supports calendar time units, so the hook allows the month probe."""
        pytest.importorskip("duckdb")
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.duckdb_frame_aggregate import (
            DuckdbFrameAggregate,
        )

        options = _time_frame_options("month")
        result = DuckdbFrameAggregate.supports_compute_framework("value_time_frame", options, DuckDBFramework)
        assert result is True


# ---------------------------------------------------------------------------
# Integration: resolve_feature surfaces the capability split
# ---------------------------------------------------------------------------


class TestResolveFeatureIntegration:
    def test_resolve_feature_splits_frameworks_for_median_scalar(self) -> None:
        """resolve_feature must list SqliteFramework as unsupported and PandasDataFrame as supported.

        resolve_feature evaluates matching under empty Options, and the group-by
        aggregation family requires partition_by to match, so the scalar aggregate
        family (matching string-based with empty Options) is the integration probe
        for the SQLite-rejects-median capability.
        """
        pytest.importorskip("pandas")
        from mloda.steward import resolve_feature

        # Importing the production classes registers them as FeatureGroup subclasses.
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.pandas_scalar_aggregate import (  # noqa: F401
            PandasScalarAggregate,
        )
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.sqlite_scalar_aggregate import (  # noqa: F401
            SqliteScalarAggregate,
        )

        result = resolve_feature("value__median_scalar")

        assert "SqliteFramework" in result.unsupported_compute_frameworks
        assert "PandasDataFrame" in result.supported_compute_frameworks
