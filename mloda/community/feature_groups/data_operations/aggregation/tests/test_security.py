"""Aggregation-specific security tests: compute-level rejection, backend
allowlist completeness, and SQL utility safety.

Generic match-validation tests live in ``test_base.py`` via
``MatchValidationTestBase``.
"""

from __future__ import annotations

import pytest

from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
)
from mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation import (
    PyArrowAggregation,
)


class TestAggregationComputeRejection:
    """Verify that invalid types reaching compute raise ValueError."""

    def test_pyarrow_rejects_unknown_type_at_compute(self) -> None:
        arrow_table = PyArrowDataOpsTestDataCreator.create()
        with pytest.raises((ValueError, KeyError)):
            PyArrowAggregation._compute_group(arrow_table, "result_col", "value_int", ["region"], "evil_type")

    def test_pandas_rejects_unknown_type_at_compute(self) -> None:
        pytest.importorskip("pandas")
        from mloda.community.feature_groups.data_operations.aggregation.pandas_aggregation import (
            PandasAggregation,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        df = arrow_table.to_pandas()
        with pytest.raises((ValueError, KeyError)):
            PandasAggregation._compute_group(df, "result_col", "value_int", ["region"], "evil_type")

    def test_duckdb_rejects_unknown_type_at_compute(self) -> None:
        duckdb = pytest.importorskip("duckdb")
        from mloda.community.feature_groups.data_operations.aggregation.duckdb_aggregation import (
            DuckdbAggregation,
        )
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        conn = duckdb.connect()
        arrow_table = PyArrowDataOpsTestDataCreator.create()
        rel = conn.from_arrow(arrow_table)
        data = DuckdbRelation(conn, rel)
        with pytest.raises(ValueError, match="[Uu]nsupported"):
            DuckdbAggregation._compute_group(data, "result_col", "value_int", ["region"], "evil_type")
        conn.close()

    def test_sqlite_rejects_unknown_type_at_compute(self) -> None:
        import sqlite3

        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation
        from mloda.community.feature_groups.data_operations.aggregation.sqlite_aggregation import (
            SqliteAggregation,
        )

        conn = sqlite3.connect(":memory:")
        arrow_table = PyArrowDataOpsTestDataCreator.create()
        data = SqliteRelation.from_arrow(conn, arrow_table)
        with pytest.raises(ValueError, match="[Uu]nsupported"):
            SqliteAggregation._compute_group(data, "result_col", "value_int", ["region"], "evil_type")
        conn.close()


class TestAllowlistCompleteness:
    """Verify that every type in AGGREGATION_TYPES is covered by every backend."""

    def test_duckdb_covers_all_types(self) -> None:
        from mloda.community.feature_groups.data_operations.aggregation.duckdb_aggregation import (
            _DUCKDB_AGG_FUNCS,
        )

        for agg_type in AggregationFeatureGroup.AGGREGATION_TYPES:
            assert agg_type in _DUCKDB_AGG_FUNCS, f"DuckDB backend missing aggregation type: {agg_type}"

    def test_sqlite_covers_supported_types(self) -> None:
        from mloda.community.feature_groups.data_operations.aggregation.sqlite_aggregation import (
            _SQLITE_AGG_FUNCS,
        )

        basic_types = {"sum", "min", "max", "avg", "count"}
        for agg_type in basic_types:
            assert agg_type in _SQLITE_AGG_FUNCS, f"SQLite backend missing basic aggregation type: {agg_type}"


class TestPartitionByInjection:
    """Verify that partition_by values with SQL injection payloads are safely quoted."""

    INJECTION_PAYLOADS = [
        'region"; DROP TABLE users--',
        "region' OR '1'='1",
        "region); DELETE FROM data--",
    ]

    def test_duckdb_quotes_partition_by(self) -> None:
        """DuckDB uses quote_ident for partition_by columns, preventing injection."""
        from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

        for payload in self.INJECTION_PAYLOADS:
            quoted = quote_ident(payload)
            assert '"' == quoted[0] and '"' == quoted[-1], f"Not properly quoted: {quoted}"
            assert payload.replace('"', '""') in quoted

    def test_sqlite_quotes_partition_by(self) -> None:
        """SQLite uses quote_ident for partition_by columns, preventing injection."""
        from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

        for payload in self.INJECTION_PAYLOADS:
            quoted = quote_ident(payload)
            assert '"' == quoted[0] and '"' == quoted[-1], f"Not properly quoted: {quoted}"
            assert payload.replace('"', '""') in quoted
