"""SQLite implementation for frame aggregate feature groups."""

from __future__ import annotations

from typing import Any, Optional, Set, Type, Union

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
)

_SQLITE_AGG_FUNCS: dict[str, str] = {
    "sum": "SUM",
    "avg": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
}


class SqliteFrameAggregate(FrameAggregateFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {SqliteFramework}

    @classmethod
    def _compute_frame(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        agg_type: str,
        frame_type: str,
        frame_size: Optional[int] = None,
        frame_unit: Optional[str] = None,
    ) -> SqliteRelation:
        agg_func = _SQLITE_AGG_FUNCS.get(agg_type)
        if agg_func is None:
            raise ValueError(f"Unsupported aggregation type for SQLite frame aggregate: {agg_type}")

        quoted_source = quote_ident(source_col)
        quoted_order = quote_ident(order_by)
        partition_clause = ", ".join(quote_ident(col) for col in partition_by)
        quoted_feature = quote_ident(feature_name)
        qrn = quote_ident("__mloda_rn__")

        null_sort = f"CASE WHEN {quoted_order} IS NULL THEN 1 ELSE 0 END"
        order_clause = f"{null_sort}, {quoted_order}"

        if frame_type in ("cumulative", "expanding"):
            frame_clause = "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        elif frame_type == "rolling":
            window_size = int(frame_size) if frame_size is not None else 1  # nosec B608
            frame_clause = f"ROWS BETWEEN {window_size - 1} PRECEDING AND CURRENT ROW"
        elif frame_type == "time":
            raise ValueError("SQLite does not support RANGE-based time windows natively")
        else:
            raise ValueError(f"Unsupported frame type for SQLite: {frame_type}")

        # nosec B608: all identifiers use quote_ident(), agg_func from whitelist
        sql = " ".join(  # nosec B608
            [
                "SELECT",
                f"{agg_func}({quoted_source}) OVER",  # nosec B608
                f"(PARTITION BY {partition_clause} ORDER BY {order_clause} {frame_clause})",  # nosec B608
                f"AS {quoted_feature},",  # nosec B608
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
