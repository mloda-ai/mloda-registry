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


def _config_frame_options(agg_type: str, frame_type: str, frame_size: int = 3) -> Options:
    """Build config-based frame-aggregate options carrying aggregation_type and frame_type discriminators."""
    return Options(
        context={
            "aggregation_type": agg_type,
            "frame_type": frame_type,
            "frame_size": frame_size,
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
    def test_pandas_accepts_percent_rank(self) -> None:
        """Pandas computes percent_rank with SQL semantics ((rank-1)/(count-1)), so the hook allows it."""
        pytest.importorskip("pandas")
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        from mloda.community.feature_groups.data_operations.row_preserving.rank.pandas_rank import (
            PandasRank,
        )

        result = PandasRank.supports_compute_framework("value__percent_rank_ranked", Options(), PandasDataFrame)
        assert result is True

    def test_pandas_accepts_dense_rank(self) -> None:
        """Pandas supports dense_rank, so the hook allows it."""
        pytest.importorskip("pandas")
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        from mloda.community.feature_groups.data_operations.row_preserving.rank.pandas_rank import (
            PandasRank,
        )

        result = PandasRank.supports_compute_framework("value__dense_rank_ranked", Options(), PandasDataFrame)
        assert result is True

    def test_polars_lazy_accepts_percent_rank(self) -> None:
        """Polars lazy supports percent_rank, so the hook allows it."""
        pytest.importorskip("polars")
        from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

        from mloda.community.feature_groups.data_operations.row_preserving.rank.polars_lazy_rank import (
            PolarsLazyRank,
        )

        result = PolarsLazyRank.supports_compute_framework("value__percent_rank_ranked", Options(), PolarsLazyDataFrame)
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

    def test_sqlite_accepts_percent_rank(self) -> None:
        """SQLite supports SQL percent_rank natively, so the hook allows it."""
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

        from mloda.community.feature_groups.data_operations.row_preserving.rank.sqlite_rank import (
            SqliteRank,
        )

        result = SqliteRank.supports_compute_framework("value__percent_rank_ranked", Options(), SqliteFramework)
        assert result is True

    def test_base_default_is_unrestricted(self) -> None:
        """No rank backend restricts today: the base supported_rank_types() default is None."""
        from mloda.community.feature_groups.data_operations.row_preserving.rank.base import RankFeatureGroup

        assert RankFeatureGroup.supported_rank_types() is None

    def test_restricting_subclass_rejects_unlisted_named_types(self) -> None:
        """The hook mechanism: a subclass restricting supported_rank_types rejects unlisted named types."""
        pytest.importorskip("pandas")
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        from mloda.community.feature_groups.data_operations.row_preserving.rank.base import RankFeatureGroup

        class _RestrictedRank(RankFeatureGroup):
            @classmethod
            def supported_rank_types(cls) -> frozenset[str] | None:
                return frozenset({"row_number", "rank", "dense_rank"})

        rejected = _RestrictedRank.supports_compute_framework("value__percent_rank_ranked", Options(), PandasDataFrame)
        accepted = _RestrictedRank.supports_compute_framework("value__dense_rank_ranked", Options(), PandasDataFrame)
        assert rejected is False
        assert accepted is True


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

    # -- Aggregation-type axis (issue #296) ---------------------------------
    # supports_compute_framework must also reject aggregation types a backend
    # cannot compute, not just unsupported frame types / time units. Today the
    # hook ignores the agg axis, so unsupported agg types slip through match
    # time and only fail later inside _compute_frame.

    @pytest.mark.parametrize(
        "feature_name",
        [
            "value__median_rolling_3",
            "value__std_rolling_3",
            "value__var_rolling_3",
            "value__cumstd",
            "value__expanding_median",
        ],
    )
    def test_sqlite_rejects_unsupported_agg_at_match_time(self, feature_name: str) -> None:
        """SQLite frame aggregate supports only sum/avg/count/min/max; std/var/median must be rejected at match time."""
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.sqlite_frame_aggregate import (
            SqliteFrameAggregate,
        )

        result = SqliteFrameAggregate.supports_compute_framework(feature_name, Options(), SqliteFramework)
        assert result is False

    @pytest.mark.parametrize("agg_type", ["median", "std", "var"])
    def test_sqlite_rejects_unsupported_agg_config_based(self, agg_type: str) -> None:
        """Config-based features carry the agg type in options; SQLite must reject std/var/median there too."""
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.sqlite_frame_aggregate import (
            SqliteFrameAggregate,
        )

        options = _config_frame_options(agg_type, "rolling")
        result = SqliteFrameAggregate.supports_compute_framework("value_frame", options, SqliteFramework)
        assert result is False

    @pytest.mark.parametrize(
        "feature_name",
        [
            "value__sum_rolling_3",
            "value__avg_rolling_3",
            "value__count_rolling_3",
            "value__min_rolling_3",
            "value__max_rolling_3",
        ],
    )
    def test_sqlite_accepts_supported_agg_at_match_time(self, feature_name: str) -> None:
        """SQLite frame aggregate supports sum/avg/count/min/max: the hook must not over-reject them."""
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.sqlite_frame_aggregate import (
            SqliteFrameAggregate,
        )

        result = SqliteFrameAggregate.supports_compute_framework(feature_name, Options(), SqliteFramework)
        assert result is True

    @pytest.mark.parametrize("agg_type", ["sum", "avg", "count", "min", "max"])
    def test_sqlite_accepts_supported_agg_config_based(self, agg_type: str) -> None:
        """Config-based features carry the agg type in options; SQLite must accept sum/avg/count/min/max there too."""
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.sqlite_frame_aggregate import (
            SqliteFrameAggregate,
        )

        options = _config_frame_options(agg_type, "rolling")
        result = SqliteFrameAggregate.supports_compute_framework("value_frame", options, SqliteFramework)
        assert result is True

    @pytest.mark.parametrize(
        "feature_name",
        [
            "value__cumstd",
            "value__cumvar",
            "value__cummedian",
            "value__expanding_std",
            "value__expanding_var",
            "value__expanding_median",
        ],
    )
    def test_polars_rejects_unsupported_agg_for_cumulative_expanding_at_match_time(self, feature_name: str) -> None:
        """Polars cumulative/expanding frames support only sum/min/max/count/avg; std/var/median must be rejected."""
        pytest.importorskip("polars")
        from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.polars_lazy_frame_aggregate import (
            PolarsLazyFrameAggregate,
        )

        result = PolarsLazyFrameAggregate.supports_compute_framework(feature_name, Options(), PolarsLazyDataFrame)
        assert result is False

    @pytest.mark.parametrize("frame_type", ["cumulative", "expanding"])
    @pytest.mark.parametrize("agg_type", ["std", "var", "median"])
    def test_polars_rejects_unsupported_agg_for_cumulative_expanding_config_based(
        self, agg_type: str, frame_type: str
    ) -> None:
        """Config-based cumulative/expanding features must reject std/var/median on Polars too."""
        pytest.importorskip("polars")
        from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.polars_lazy_frame_aggregate import (
            PolarsLazyFrameAggregate,
        )

        options = _config_frame_options(agg_type, frame_type)
        result = PolarsLazyFrameAggregate.supports_compute_framework("value_frame", options, PolarsLazyDataFrame)
        assert result is False

    @pytest.mark.parametrize(
        "feature_name",
        [
            "value__median_rolling_3",
            "value__std_rolling_3",
            "value__var_rolling_3",
            "value__median_7_day_window",
        ],
    )
    def test_polars_accepts_std_var_median_for_rolling_and_time(self, feature_name: str) -> None:
        """Polars rolling/time frames support std/var/median: the hook must not reject those combinations."""
        pytest.importorskip("polars")
        from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.polars_lazy_frame_aggregate import (
            PolarsLazyFrameAggregate,
        )

        result = PolarsLazyFrameAggregate.supports_compute_framework(feature_name, Options(), PolarsLazyDataFrame)
        assert result is True

    @pytest.mark.parametrize("agg_type", ["std", "var", "median"])
    def test_polars_accepts_std_var_median_for_rolling_config_based(self, agg_type: str) -> None:
        """Config-based rolling features must accept std/var/median on Polars too."""
        pytest.importorskip("polars")
        from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.polars_lazy_frame_aggregate import (
            PolarsLazyFrameAggregate,
        )

        options = _config_frame_options(agg_type, "rolling")
        result = PolarsLazyFrameAggregate.supports_compute_framework("value_frame", options, PolarsLazyDataFrame)
        assert result is True

    def test_pandas_accepts_median_rolling(self) -> None:
        """Pandas supports the full frame agg-type table, so median rolling must not be rejected (regression guard)."""
        pytest.importorskip("pandas")
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pandas_frame_aggregate import (
            PandasFrameAggregate,
        )

        result = PandasFrameAggregate.supports_compute_framework("value__median_rolling_3", Options(), PandasDataFrame)
        assert result is True

    def test_duckdb_accepts_median_rolling(self) -> None:
        """DuckDB supports the full frame agg-type table, so median rolling must not be rejected (regression guard)."""
        pytest.importorskip("duckdb")
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.duckdb_frame_aggregate import (
            DuckdbFrameAggregate,
        )

        result = DuckdbFrameAggregate.supports_compute_framework("value__median_rolling_3", Options(), DuckDBFramework)
        assert result is True

    def test_sqlite_conservative_default_when_frame_type_unresolved(self) -> None:
        """A name encoding no parsable frame/agg keeps the default True (conservative)."""
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.sqlite_frame_aggregate import (
            SqliteFrameAggregate,
        )

        result = SqliteFrameAggregate.supports_compute_framework("totally_unrelated", Options(), SqliteFramework)
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

    def test_capability_split_rejects_sqlite_for_median_rolling_frame(self) -> None:
        """The capability split must reject SqliteFramework for a median rolling frame while keeping Pandas/DuckDB.

        resolve_feature cannot be the probe here: frame aggregate's
        ``match_feature_group_criteria`` requires partition_by/order_by, which are absent
        under the empty Options resolve_feature evaluates matching with (the same reason the
        median-scalar test above uses the scalar family instead of the group-by family). This
        test exercises ``split_frameworks_by_capability`` directly, which is exactly the
        mechanism resolve_feature uses internally to derive
        supported/unsupported_compute_frameworks.
        """
        pytest.importorskip("pandas")
        pytest.importorskip("duckdb")
        from mloda.core.prepare.identify_feature_group import split_frameworks_by_capability

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.duckdb_frame_aggregate import (
            DuckdbFrameAggregate,
        )
        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pandas_frame_aggregate import (
            PandasFrameAggregate,
        )
        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.sqlite_frame_aggregate import (
            SqliteFrameAggregate,
        )

        supported, rejected = split_frameworks_by_capability(
            [SqliteFrameAggregate, PandasFrameAggregate, DuckdbFrameAggregate],
            "value__median_rolling_3",
            Options(),
        )
        supported_names = {c.get_class_name() for c in supported}
        rejected_names = {c.get_class_name() for c in rejected}

        assert "SqliteFramework" in rejected_names
        assert "PandasDataFrame" in supported_names
        assert "DuckDBFramework" in supported_names
