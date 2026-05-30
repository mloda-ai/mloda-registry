"""SQLite implementation for offset feature groups."""

from __future__ import annotations

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name, quote_ident
from mloda_plugins.compute_framework.base_implementations.sql.sql_window import OrderBy
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.offset.base import (
    OffsetFeatureGroup,
)


class SqliteOffset(OffsetFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
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
        if offset_type in ("first_value", "last_value"):
            return cls._compute_first_last(data, feature_name, source_col, partition_by, order_by, offset_type)

        quoted_source = quote_ident(source_col)

        # NullPolicy.NULLS_LAST: ``OrderBy(order_by, nulls="last")`` renders
        # ``ORDER BY ... NULLS LAST``, equivalent to the old
        # ``CASE WHEN order IS NULL THEN 1 ELSE 0 END, order`` sort key.
        order_spec = [OrderBy(order_by, nulls="last")]

        original_cols = list(data.columns)
        rn = pick_helper_column_name(taken=set(data.columns) | {feature_name})
        rel = data.with_row_number(rn, order_by=["rowid"])

        if offset_type.startswith("pct_change_"):
            # The window result is wrapped in a CASE expression, so compute LAG into a
            # helper column, then apply the wrapper via a raw projection (Pattern W).
            offset_n = int(offset_type[len("pct_change_") :])
            prev = pick_helper_column_name(taken=set(data.columns) | {feature_name, rn})
            rel = rel.window(
                f"LAG({quoted_source}, {offset_n})",
                prev,
                partition_by=partition_by,
                order_by=order_spec,
            )
            qhelper = quote_ident(prev)
            wrapper = (
                f"CASE WHEN {qhelper} IS NOT NULL AND {qhelper} != 0 "
                f"THEN ({quoted_source} - {qhelper}) * 1.0 / {qhelper} END"
            )
            proj = (
                ", ".join(quote_ident(c) for c in original_cols)
                + f", {quote_ident(rn)}, {wrapper} AS {quote_ident(feature_name)}"
            )
            rel = rel.select(_raw_sql=proj)
            rel = rel.order(rn)
            return rel.select(*original_cols, feature_name)

        if offset_type.startswith("lag_"):
            offset_n = int(offset_type[len("lag_") :])
            func = f"LAG({quoted_source}, {offset_n})"
        elif offset_type.startswith("lead_"):
            offset_n = int(offset_type[len("lead_") :])
            func = f"LEAD({quoted_source}, {offset_n})"
        elif offset_type.startswith("diff_"):
            offset_n = int(offset_type[len("diff_") :])
            # OVER binds to LAG only, so the subtraction stays outside the window.
            func = f"{quoted_source} - LAG({quoted_source}, {offset_n})"
        else:
            supported = "lag, lead, diff, pct_change, first_value, last_value"
            raise ValueError(f"Unsupported offset type for SQLite: {offset_type}. Supported types: {supported}")

        rel = rel.window(func, feature_name, partition_by=partition_by, order_by=order_spec)
        rel = rel.order(rn)
        return rel.select(*original_cols, feature_name)

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
        """SQLite lacks IGNORE NULLS in window functions, so select the first
        (or last) non-null value per partition via a correlated subquery."""
        quoted_source = quote_ident(source_col)
        quoted_order = quote_ident(order_by)
        quoted_feature = quote_ident(feature_name)

        partition_match = " AND ".join(f"t2.{quote_ident(col)} IS t1.{quote_ident(col)}" for col in partition_by)
        null_sort = f"CASE WHEN t2.{quoted_order} IS NULL THEN 1 ELSE 0 END"

        if offset_type == "first_value":
            sort_clause = f"{null_sort}, t2.{quoted_order}"
        else:
            sort_clause = f"{null_sort} DESC, t2.{quoted_order} DESC"

        subquery = (
            f"(SELECT t2.{quoted_source} FROM {quote_ident(data.table_name)} t2 "  # nosec
            f"WHERE {partition_match} AND t2.{quoted_source} IS NOT NULL "
            f"ORDER BY {sort_clause} LIMIT 1)"
        )
        sql = (
            f"SELECT {subquery} AS {quoted_feature} "  # nosec
            f"FROM {quote_ident(data.table_name)} t1 ORDER BY rowid"
        )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()
        result_values = [row[0] for row in rows]
        return data.append_column(feature_name, result_values)
