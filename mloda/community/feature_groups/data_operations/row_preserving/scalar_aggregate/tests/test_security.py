"""Scalar-aggregate-specific security tests: compute-level rejection, backend
allowlist completeness, and SQL utility safety.

Generic match-validation tests live in ``test_base.py`` via
``MatchValidationTestBase``.
"""

from __future__ import annotations

import pytest

from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator

from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.base import (
    AGGREGATION_TYPES,
)
from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.pyarrow_scalar_aggregate import (
    PyArrowScalarAggregate,
)


class TestScalarAggregateComputeRejection:
    """Verify that invalid types reaching compute raise ValueError."""

    def test_pandas_rejects_unknown_type_at_compute(self) -> None:
        pytest.importorskip("pandas")
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.pandas_scalar_aggregate import (
            PandasScalarAggregate,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        df = arrow_table.to_pandas()
        with pytest.raises(ValueError, match="[Uu]nsupported"):
            PandasScalarAggregate._compute_aggregation(df, "result_col", "value_int", "evil_type")

    def test_pyarrow_rejects_unknown_type_at_compute(self) -> None:
        arrow_table = PyArrowDataOpsTestDataCreator.create()
        with pytest.raises(ValueError, match="[Uu]nsupported"):
            PyArrowScalarAggregate._compute_aggregation(arrow_table, "result_col", "value_int", "evil_type")


class TestAllowlistCompleteness:
    """Verify that every type in AGGREGATION_TYPES is covered by every backend."""

    def test_duckdb_covers_all_types(self) -> None:
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.duckdb_scalar_aggregate import (
            _DUCKDB_AGG_FUNCS,
        )

        for agg_type in AGGREGATION_TYPES:
            assert agg_type in _DUCKDB_AGG_FUNCS, f"DuckDB backend missing aggregation type: {agg_type}"

    def test_pandas_covers_all_types(self) -> None:
        pytest.importorskip("pandas")
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.pandas_scalar_aggregate import (
            PandasScalarAggregate,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        df = arrow_table.to_pandas()
        for agg_type in AGGREGATION_TYPES:
            result = PandasScalarAggregate._compute_aggregation(df, f"test_{agg_type}", "value_int", agg_type)
            assert f"test_{agg_type}" in result.columns

    def test_pyarrow_covers_all_types(self) -> None:
        arrow_table = PyArrowDataOpsTestDataCreator.create()
        for agg_type in AGGREGATION_TYPES:
            result = PyArrowScalarAggregate._compute_aggregation(arrow_table, f"test_{agg_type}", "value_int", agg_type)
            assert f"test_{agg_type}" in result.column_names

    def test_polars_covers_all_types(self) -> None:
        polars = pytest.importorskip("polars")
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.polars_lazy_scalar_aggregate import (
            PolarsLazyScalarAggregate,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        lf = polars.from_arrow(arrow_table).lazy()
        for agg_type in AGGREGATION_TYPES:
            result = PolarsLazyScalarAggregate._compute_aggregation(lf, f"test_{agg_type}", "value_int", agg_type)
            collected = result.collect()
            assert f"test_{agg_type}" in collected.columns

    def test_sqlite_covers_supported_types(self) -> None:
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.sqlite_scalar_aggregate import (
            _SQLITE_AGG_FUNCS,
        )

        basic_types = {"sum", "min", "max", "avg", "mean", "count"}
        for agg_type in basic_types:
            assert agg_type in _SQLITE_AGG_FUNCS, f"SQLite backend missing basic aggregation type: {agg_type}"


class TestSqlUtilities:
    """Verify SQL utility functions handle edge cases safely."""

    def test_sqlite_quote_ident_handles_double_quotes(self) -> None:
        from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

        assert quote_ident('col"name') == '"col""name"'
        assert quote_ident("normal_col") == '"normal_col"'
        assert quote_ident("col'; DROP TABLE--") == '"col\'; DROP TABLE--"'
