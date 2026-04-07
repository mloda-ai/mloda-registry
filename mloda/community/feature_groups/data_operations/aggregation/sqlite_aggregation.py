"""SQLite implementation for aggregation feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation, _next_table_name

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
)

# Aggregation types that SQLite supports natively.
_SQLITE_AGG_FUNCS: dict[str, str] = {
    "sum": "SUM",
    "avg": "AVG",
    "mean": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
}


class SqliteAggregation(AggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SqliteFramework}

    @classmethod
    def _compute_group(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> SqliteRelation:
        """Execute the aggregation as a SQL GROUP BY query."""
        agg_func = _SQLITE_AGG_FUNCS.get(agg_type)
        if agg_func is None:
            raise ValueError(f"Unsupported aggregation type for SQLite: {agg_type}")

        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)
        partition_cols = ", ".join(quote_ident(col) for col in partition_by)

        source_sql = quoted_source
        if mask_spec is not None:
            from mloda.community.feature_groups.data_operations.mask_utils import build_sql_case_when

            source_sql = build_sql_case_when(mask_spec, quoted_source)

        new_name = _next_table_name()
        sql = (
            f"CREATE TEMP VIEW {quote_ident(new_name)} AS "  # nosec
            f"SELECT {partition_cols}, "
            f"{agg_func}({source_sql}) AS {quoted_feature} "
            f"FROM {quote_ident(data.table_name)} "
            f"GROUP BY {partition_cols}"
        )
        data.connection.execute(sql)
        return SqliteRelation(data.connection, new_name, _is_view=True)
