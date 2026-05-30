"""SQLite implementation for rank feature groups."""

from __future__ import annotations


import pyarrow as pa

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name, quote_ident
from mloda_plugins.compute_framework.base_implementations.sql.sql_window import OrderBy
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
        # NullPolicy.NULLS_LAST: ``OrderBy(order_by, nulls="last")`` renders
        # ``ORDER BY ... NULLS LAST``, equivalent to the old
        # ``CASE WHEN order IS NULL THEN 1 ELSE 0 END, order`` sort key.
        original_cols = list(data.columns)
        rn = pick_helper_column_name(taken=set(data.columns) | {feature_name})
        rel = data.with_row_number(rn, order_by=["rowid"])

        if rank_type.startswith(("top_", "bottom_")):
            # The window result is wrapped in a boolean comparison, so compute
            # ROW_NUMBER() into a helper column, then apply the wrapper via a raw
            # projection (Pattern W). ``descending`` follows the top/bottom direction.
            is_top = rank_type.startswith("top_")
            prefix = "top_" if is_top else "bottom_"
            n_val = int(rank_type[len(prefix) :])
            helper = pick_helper_column_name(taken=set(data.columns) | {feature_name, rn})
            rel = rel.window(
                "ROW_NUMBER()",
                helper,
                partition_by=partition_by,
                order_by=[OrderBy(order_by, descending=is_top, nulls="last")],
            )
            qhelper = quote_ident(helper)
            proj = (
                ", ".join(quote_ident(c) for c in original_cols)
                + f", {quote_ident(rn)}, ({qhelper} <= {n_val}) AS {quote_ident(feature_name)}"
            )
            rel = rel.select(_raw_sql=proj)
            rel = rel.order(rn)
            result = rel.select(*original_cols, feature_name)
            return _BoolCastRelation(result, {feature_name})

        if rank_type.startswith("ntile_"):
            ntile_n = int(rank_type[len("ntile_") :])
            rank_func = f"NTILE({ntile_n})"
        else:
            standard_func = _SQLITE_RANK_FUNCS.get(rank_type)
            if standard_func is None:
                raise ValueError(f"Unsupported rank type for SQLite: {rank_type}")
            rank_func = standard_func

        rel = rel.window(
            rank_func,
            feature_name,
            partition_by=partition_by,
            order_by=[OrderBy(order_by, nulls="last")],
        )
        rel = rel.order(rn)
        return rel.select(*original_cols, feature_name)
