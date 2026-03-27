"""SQLite implementation for offset feature groups."""

from __future__ import annotations

from typing import Set, Type, Union

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
        elif offset_type in ("first_value", "last_value"):
            return cls._compute_first_last(data, feature_name, source_col, partition_by, order_by, offset_type)
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

    @classmethod
    def _compute_first_last(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        offset_type: str,
    ) -> SqliteRelation:
        """Compute first_value or last_value using a correlated subquery.

        SQLite does not support IGNORE NULLS in window functions, so we use a
        correlated subquery that selects the first (or last) non-null value
        within each partition.
        """
        quoted_source = quote_ident(source_col)
        quoted_order = quote_ident(order_by)
        quoted_feature = quote_ident(feature_name)
        qrn = quote_ident("__mloda_rn__")

        partition_match = " AND ".join(f"t2.{quote_ident(col)} IS t1.{quote_ident(col)}" for col in partition_by)
        null_sort = f"CASE WHEN t2.{quoted_order} IS NULL THEN 1 ELSE 0 END"

        if offset_type == "first_value":
            sort_clause = f"{null_sort}, t2.{quoted_order}"
        else:
            sort_clause = f"{null_sort} DESC, t2.{quoted_order} DESC"

        subquery = (
            f"(SELECT t2.{quoted_source} FROM {quote_ident(data.table_name)} t2 "  # nosec B608
            f"WHERE {partition_match} AND t2.{quoted_source} IS NOT NULL "
            f"ORDER BY {sort_clause} LIMIT 1)"
        )
        sql = (
            f"SELECT {subquery} AS {quoted_feature}, "  # nosec B608
            f"ROW_NUMBER() OVER (ORDER BY rowid) AS {qrn} "
            f"FROM {quote_ident(data.table_name)} t1 ORDER BY {qrn}"
        )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()
        result_values = [row[0] for row in rows]
        return data.append_column(feature_name, result_values)
