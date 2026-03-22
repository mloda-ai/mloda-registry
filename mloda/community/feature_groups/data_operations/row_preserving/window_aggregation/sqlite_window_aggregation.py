"""SQLite implementation for window aggregation feature groups.

Uses SqliteFramework and SqliteRelation from mloda core. SQL-native aggregations
(sum, avg, count, min, max) run as window functions via data.select(_raw_sql=...).
Unsupported aggregations (std, var, median, mode, nunique, first, last) fall back
to Python by materializing via to_arrow_table(), computing the aggregate, and
appending the result column via data.append_column().
"""

from __future__ import annotations

from typing import Any, Set, Type, Union

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)

# Aggregation types that SQLite supports natively in window functions.
_SQLITE_AGG_FUNCS: dict[str, str] = {
    "sum": "SUM",
    "avg": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
}


class SqliteWindowAggregation(WindowAggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {SqliteFramework}

    @classmethod
    def _compute_window(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
    ) -> SqliteRelation:
        """Delegate to SQL path for native aggs, Python fallback otherwise."""
        if agg_type in _SQLITE_AGG_FUNCS:
            return cls._compute_window_sql(data, feature_name, source_col, partition_by, agg_type)
        return cls._compute_window_python(data, feature_name, source_col, partition_by, agg_type)

    @classmethod
    def _compute_window_sql(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
    ) -> SqliteRelation:
        """Execute the aggregation as a SQL window function.

        Computes the window aggregate via a SQL query that includes a row-number
        column to preserve original row order, extracts the result values in
        order, and appends them to the original relation.
        """
        agg_func = _SQLITE_AGG_FUNCS[agg_type]
        quoted_source = quote_ident(source_col)
        partition_clause = ", ".join(quote_ident(col) for col in partition_by)
        quoted_feature = quote_ident(feature_name)
        qrn = quote_ident("__mloda_rn__")

        # Compute window aggregate with row ordering in a single query.
        sql = (
            f"SELECT {agg_func}({quoted_source}) OVER (PARTITION BY {partition_clause}) AS {quoted_feature}, "  # nosec B608
            f"ROW_NUMBER() OVER () AS {qrn} "
            f"FROM {quote_ident(data.table_name)} ORDER BY {qrn}"
        )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()

        result_values = [row[0] for row in rows]
        return data.append_column(feature_name, result_values)

    @classmethod
    def _compute_window_python(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
    ) -> SqliteRelation:
        """Compute aggregation using Python fallback (materialize, compute, append)."""
        arrow_table = data.to_arrow_table()
        num_rows = arrow_table.num_rows

        # Build group keys per row.
        keys: list[tuple[Any, ...]] = []
        for i in range(num_rows):
            key = tuple(arrow_table.column(col)[i].as_py() for col in partition_by)
            keys.append(key)

        # Collect source values per group.
        groups: dict[tuple[Any, ...], list[Any]] = {}
        for i in range(num_rows):
            key = keys[i]
            val = arrow_table.column(source_col)[i].as_py()
            if key not in groups:
                groups[key] = []
            groups[key].append(val)

        # Compute aggregate per group.
        agg_results: dict[tuple[Any, ...], Any] = {}
        for key, values in groups.items():
            agg_results[key] = cls._aggregate(values, agg_type)

        # Broadcast back to every row.
        result_values = [agg_results[keys[i]] for i in range(num_rows)]

        return data.append_column(feature_name, result_values)
