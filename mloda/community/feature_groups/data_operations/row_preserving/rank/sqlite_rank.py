"""SQLite implementation for rank feature groups."""

from __future__ import annotations


import pyarrow as pa

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.rank.base import (
    RankFeatureGroup,
)


class _BoolCastRelation(SqliteRelation):
    """Thin wrapper that casts specified columns from int to boolean in to_arrow_table.

    SQLite has no native boolean type, so boolean comparison results (0/1) are
    stored as INTEGER.  This wrapper ensures the Arrow table produced by
    ``to_arrow_table()`` uses ``pa.bool_()`` for the designated columns so that
    downstream ``to_pylist()`` returns Python ``True``/``False``.
    This ensures parity with the PyArrow reference, which returns native bool.
    """

    def __init__(self, base: SqliteRelation, bool_columns: set[str]) -> None:
        super().__init__(base.connection, base.table_name, _is_view=True)
        self._bool_columns = bool_columns

    def to_arrow_table(self) -> pa.Table:
        table = super().to_arrow_table()
        for col_name in self._bool_columns:
            if col_name in table.column_names:
                idx = table.column_names.index(col_name)
                bool_array = pa.array([bool(v) for v in table.column(col_name).to_pylist()])
                table = table.set_column(idx, col_name, bool_array)
        return table


# Rank functions natively supported by SQLite.
_SQLITE_RANK_FUNCS: dict[str, str] = {
    "row_number": "ROW_NUMBER()",
    "rank": "RANK()",
    "dense_rank": "DENSE_RANK()",
    "percent_rank": "PERCENT_RANK()",
}


class SqliteRank(RankFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SqliteFramework}

    @classmethod
    def _compute_rank(
        cls,
        data: SqliteRelation,
        feature_name: str,
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
        elif rank_type.startswith(("top_", "bottom_")):
            is_top = rank_type.startswith("top_")
            prefix = "top_" if is_top else "bottom_"
            n_val = int(rank_type[len(prefix) :])
            direction = "DESC" if is_top else "ASC"
            top_bottom_order = f"{null_sort}, {quoted_order} {direction}"
            rank_expr = f"(ROW_NUMBER() OVER (PARTITION BY {partition_clause} ORDER BY {top_bottom_order}) <= {n_val})"
        else:
            rank_func = _SQLITE_RANK_FUNCS.get(rank_type)
            if rank_func is None:
                raise ValueError(f"Unsupported rank type for SQLite: {rank_type}")
            rank_expr = rank_func

        # Safety: all identifiers use quote_ident(), rank_expr from whitelist
        if rank_type.startswith(("top_", "bottom_")):
            # rank_expr already contains full window expression with boolean comparison
            sql = (
                f"SELECT {rank_expr} AS {quoted_feature}, "  # nosec
                f"ROW_NUMBER() OVER (ORDER BY rowid) AS {qrn} "
                f"FROM {quote_ident(data.table_name)} ORDER BY {qrn}"
            )
        else:
            sql = (
                f"SELECT {rank_expr} OVER "  # nosec
                f"(PARTITION BY {partition_clause} ORDER BY {order_clause}) AS {quoted_feature}, "
                f"ROW_NUMBER() OVER (ORDER BY rowid) AS {qrn} "
                f"FROM {quote_ident(data.table_name)} ORDER BY {qrn}"
            )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()

        result_values = [row[0] for row in rows]
        result = data.append_column(feature_name, result_values)

        if rank_type.startswith(("top_", "bottom_")):
            return _BoolCastRelation(result, {feature_name})
        return result
