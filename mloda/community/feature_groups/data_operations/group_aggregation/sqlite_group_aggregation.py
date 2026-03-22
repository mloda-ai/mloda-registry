"""SQLite implementation for group aggregation feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation, _next_table_name

from mloda.community.feature_groups.data_operations.group_aggregation.base import (
    GroupAggregationFeatureGroup,
)

# Aggregation types that SQLite supports natively.
_SQLITE_AGG_FUNCS: dict[str, str] = {
    "sum": "SUM",
    "avg": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
}


class SqliteGroupAggregation(GroupAggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {SqliteFramework}

    @classmethod
    def _compute_group(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
    ) -> SqliteRelation:
        """Execute the aggregation as a SQL GROUP BY query."""
        agg_func = _SQLITE_AGG_FUNCS.get(agg_type)
        if agg_func is None:
            raise ValueError(f"Unsupported aggregation type for SQLite: {agg_type}")

        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)
        partition_cols = ", ".join(quote_ident(col) for col in partition_by)

        new_name = _next_table_name()
        sql = (
            f"CREATE TEMP VIEW {quote_ident(new_name)} AS "  # nosec B608
            f"SELECT {partition_cols}, "
            f"{agg_func}({quoted_source}) AS {quoted_feature} "
            f"FROM {quote_ident(data.table_name)} "
            f"GROUP BY {partition_cols}"
        )
        data.connection.execute(sql)
        return SqliteRelation(data.connection, new_name, _is_view=True)
