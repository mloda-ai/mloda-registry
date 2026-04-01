"""SQLite implementation for single-column global aggregate broadcast."""

from __future__ import annotations

from typing import Set, Type, Union

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.base import (
    ScalarAggregateFeatureGroup,
)

_SQLITE_AGG_FUNCS: dict[str, str] = {
    "sum": "SUM",
    "min": "MIN",
    "max": "MAX",
    "avg": "AVG",
    "mean": "AVG",
    "count": "COUNT",
}


class SqliteScalarAggregate(ScalarAggregateFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {SqliteFramework}

    @classmethod
    def _compute_aggregation(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        agg_type: str,
    ) -> SqliteRelation:
        agg_func = _SQLITE_AGG_FUNCS.get(agg_type)
        if agg_func is None:
            raise ValueError(f"Unsupported aggregation type for SQLite: {agg_type}")

        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)

        sql = " ".join(
            [
                "SELECT",
                f"{agg_func}({quoted_source}) OVER () AS {quoted_feature}",
                "FROM",
                f"{quote_ident(data.table_name)}",
            ]
        )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()

        result_values = [row[0] for row in rows]
        return data.append_column(feature_name, result_values)
