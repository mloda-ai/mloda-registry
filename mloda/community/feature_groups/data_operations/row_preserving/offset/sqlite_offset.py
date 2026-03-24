"""SQLite implementation for offset feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.offset.base import (
    OffsetFeatureGroup,
)


class SqliteOffset(OffsetFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {SqliteFramework}

    @classmethod
    def _compute_offset(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        offset_type: str,
    ) -> SqliteRelation:
        quoted_source = quote_ident(source_col)
        quoted_order = quote_ident(order_by)
        quoted_feature = quote_ident(feature_name)
        partition_clause = ", ".join(quote_ident(col) for col in partition_by)
        qrn = quote_ident("__mloda_rn__")

        null_sort = f"CASE WHEN {quoted_order} IS NULL THEN 1 ELSE 0 END"
        order_clause = f"{null_sort}, {quoted_order}"
        window_clause = f"PARTITION BY {partition_clause} ORDER BY {order_clause}"  # nosec B608

        if offset_type.startswith("lag_"):
            offset_n = offset_type[len("lag_") :]
            offset_expr = f"LAG({quoted_source}, {offset_n})"
        elif offset_type.startswith("lead_"):
            offset_n = offset_type[len("lead_") :]
            offset_expr = f"LEAD({quoted_source}, {offset_n})"
        elif offset_type == "first_value":
            # SQLite doesn't support IGNORE NULLS. Use subquery to get first non-null.
            partition_match = " AND ".join(f"t2.{quote_ident(col)} IS t1.{quote_ident(col)}" for col in partition_by)
            offset_expr = (
                f"(SELECT t2.{quoted_source} FROM {quote_ident(data.table_name)} t2 "  # nosec B608
                f"WHERE {partition_match} AND t2.{quoted_source} IS NOT NULL "
                f"ORDER BY CASE WHEN t2.{quoted_order} IS NULL THEN 1 ELSE 0 END, t2.{quoted_order} LIMIT 1)"
            )
            sql = (
                f"SELECT {offset_expr} AS {quoted_feature}, "  # nosec B608
                f"ROW_NUMBER() OVER (ORDER BY rowid) AS {qrn} "
                f"FROM {quote_ident(data.table_name)} t1 ORDER BY {qrn}"
            )
            cursor = data.connection.execute(sql)
            rows = cursor.fetchall()
            result_values = [row[0] for row in rows]
            return data.append_column(feature_name, result_values)
        elif offset_type == "last_value":
            partition_match = " AND ".join(f"t2.{quote_ident(col)} IS t1.{quote_ident(col)}" for col in partition_by)
            offset_expr = (
                f"(SELECT t2.{quoted_source} FROM {quote_ident(data.table_name)} t2 "  # nosec B608
                f"WHERE {partition_match} AND t2.{quoted_source} IS NOT NULL "
                f"ORDER BY CASE WHEN t2.{quoted_order} IS NULL THEN 1 ELSE 0 END DESC, t2.{quoted_order} DESC LIMIT 1)"
            )
            sql = (
                f"SELECT {offset_expr} AS {quoted_feature}, "  # nosec B608
                f"ROW_NUMBER() OVER (ORDER BY rowid) AS {qrn} "
                f"FROM {quote_ident(data.table_name)} t1 ORDER BY {qrn}"
            )
            cursor = data.connection.execute(sql)
            rows = cursor.fetchall()
            result_values = [row[0] for row in rows]
            return data.append_column(feature_name, result_values)
        else:
            raise ValueError(f"Unsupported offset type for SQLite: {offset_type}")

        sql = (
            f"SELECT {offset_expr} OVER ({window_clause}) AS {quoted_feature}, "  # nosec B608
            f"ROW_NUMBER() OVER (ORDER BY rowid) AS {qrn} "
            f"FROM {quote_ident(data.table_name)} ORDER BY {qrn}"
        )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()

        result_values = [row[0] for row in rows]
        return data.append_column(feature_name, result_values)
