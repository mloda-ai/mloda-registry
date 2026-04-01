"""SQLite implementation for window aggregation feature groups."""

from __future__ import annotations

from typing import Any, Optional, Set, Type, Union

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
    "mean": "AVG",
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
        order_by: Optional[str] = None,
    ) -> SqliteRelation:
        """Execute the aggregation as a SQL window function."""
        agg_func = _SQLITE_AGG_FUNCS.get(agg_type)
        if agg_func is None:
            raise ValueError(f"Unsupported aggregation type for SQLite: {agg_type}")

        quoted_source = quote_ident(source_col)
        partition_clause = ", ".join(quote_ident(col) for col in partition_by)
        quoted_feature = quote_ident(feature_name)
        qrn = quote_ident("__mloda_rn__")

        # Safety: no user-supplied values, only quote_ident()-quoted identifiers
        # and whitelisted SQL keywords. PEP 249 qmark parametrization does not
        # apply (identifiers cannot be parameterized per the SQL standard).
        sql = " ".join(
            [
                "SELECT",
                f"{agg_func}({quoted_source}) OVER (PARTITION BY {partition_clause}) AS {quoted_feature},",
                f"ROW_NUMBER() OVER (ORDER BY rowid) AS {qrn}",
                "FROM",
                f"{quote_ident(data.table_name)}",
                "ORDER BY",
                qrn,
            ]
        )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()

        result_values = [row[0] for row in rows]
        return data.append_column(feature_name, result_values)
