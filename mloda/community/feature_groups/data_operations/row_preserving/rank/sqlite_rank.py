"""SQLite implementation for rank feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation, _next_table_name

from mloda.community.feature_groups.data_operations.row_preserving.rank.base import (
    RankFeatureGroup,
)

# Rank functions natively supported by SQLite.
_SQLITE_RANK_FUNCS: dict[str, str] = {
    "row_number": "ROW_NUMBER()",
    "rank": "RANK()",
    "dense_rank": "DENSE_RANK()",
    "percent_rank": "PERCENT_RANK()",
}


class SqliteRank(RankFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {SqliteFramework}

    @classmethod
    def _compute_rank(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        rank_type: str,
    ) -> SqliteRelation:
        """Execute the rank as a SQL window function with NULLS LAST."""
        quoted_order = quote_ident(order_by)
        quoted_feature = quote_ident(feature_name)
        partition_clause = ", ".join(quote_ident(col) for col in partition_by)
        qrn = quote_ident("__mloda_rn__")

        # NullPolicy.NULLS_LAST: add CASE WHEN to ORDER BY
        null_sort = f"CASE WHEN {quoted_order} IS NULL THEN 1 ELSE 0 END"
        order_clause = f"{null_sort}, {quoted_order}"

        if rank_type.startswith("ntile_"):
            ntile_n = int(rank_type[len("ntile_") :])
            rank_expr = f"NTILE({ntile_n})"
        else:
            rank_func = _SQLITE_RANK_FUNCS.get(rank_type)
            if rank_func is None:
                raise ValueError(f"Unsupported rank type for SQLite: {rank_type}")
            rank_expr = rank_func

        # nosec B608: all identifiers use quote_ident(), rank_expr from whitelist
        sql = (
            f"SELECT {rank_expr} OVER "  # nosec B608
            f"(PARTITION BY {partition_clause} ORDER BY {order_clause}) AS {quoted_feature}, "
            f"ROW_NUMBER() OVER (ORDER BY rowid) AS {qrn} "
            f"FROM {quote_ident(data.table_name)} ORDER BY {qrn}"
        )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()

        result_values = [row[0] for row in rows]
        return data.append_column(feature_name, result_values)
