"""SQLite implementation of gap-threshold sessionization.

SQLite has no native timestamp type and no ``.query()`` relational method, so
the session ids are computed by running raw SQL directly against the relation's
table (mirroring ``sqlite_time_bucketization``), fetching them in ``rowid``
order, and appending the result column via ``append_column``.

Floating-point note
-------------------

SQLite expresses timestamp gaps with ``julianday`` differences, which carry
float noise (a true 1800.0s gap can compute as 1800.0000134). The gap seconds
are therefore ``ROUND``-ed to whole seconds before the ``> threshold``
comparison, so the exact-boundary case (gap == threshold must stay in the SAME
session under the strict ``gap > threshold`` rule) does not flip on noise.
"""

from __future__ import annotations

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.sessionization.base import (
    SessionizationFeatureGroup,
)


class SqliteSessionization(SessionizationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SqliteFramework}

    @classmethod
    def _assert_source_column_present(cls, data: SqliteRelation, order_col: str) -> None:
        if order_col not in data.columns:
            raise ValueError(
                f"Source column {order_col!r} is not present in the SQLite table; available: {data.columns}."
            )

    @classmethod
    def _compute_session(
        cls,
        data: SqliteRelation,
        feature_name: str,
        order_col: str,
        threshold_seconds: int,
        partition_by: list[str],
    ) -> SqliteRelation:
        # Safety: every identifier (order column, partition columns, table name)
        # is quoted via quote_ident(); the only interpolated literal is the
        # integer threshold_seconds. No user-controlled string is inlined unquoted.
        q_order = quote_ident(order_col)
        q_table = quote_ident(data.table_name)
        partition_cols = [quote_ident(c) for c in partition_by]

        partition_clause = f"PARTITION BY {', '.join(partition_cols)} " if partition_cols else ""
        window_def = f"w AS ({partition_clause}ORDER BY {q_order})"
        order_keys = ", ".join([*partition_cols, q_order])

        gap_seconds = f"ROUND((julianday({q_order}) - julianday(LAG({q_order}) OVER w)) * 86400.0)"
        is_new_expr = (
            f"CASE WHEN LAG({q_order}) OVER w IS NULL OR {gap_seconds} > {threshold_seconds} THEN 1 ELSE 0 END"
        )

        inner_cols = ", ".join([*partition_cols, q_order])
        sql = (
            f"SELECT sid FROM ("  # nosec
            f"SELECT __rid, "
            f"(SUM(is_new) OVER (ORDER BY {order_keys} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) - 1) AS sid "
            f"FROM ("
            f"SELECT rowid AS __rid, {inner_cols}, {is_new_expr} AS is_new "
            f"FROM {q_table} "
            f"WINDOW {window_def}"
            f")"
            f") ORDER BY __rid"
        )
        cursor = data.connection.execute(sql)
        values = [int(row[0]) for row in cursor.fetchall()]

        return data.append_column(feature_name, values)
