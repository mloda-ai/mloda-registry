"""SQLite implementation for offset feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name, quote_ident
from mloda_plugins.compute_framework.base_implementations.sql.sql_window import OrderBy
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.offset.base import (
    OffsetFeatureGroup,
)


class SqliteOffset(Implementation[OffsetFeatureGroup]):
    feature_group = OffsetFeatureGroup
    compute_framework = SqliteFramework

    @classmethod
    def calculate(
        cls, group: type[OffsetFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.offset(data, features, compute_offset=cls._compute_offset)

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

        if offset_type.startswith("pct_change_"):
            # The window result is wrapped in a CASE expression, so compute LAG into a
            # helper column, then apply the wrapper via a raw projection (Pattern W).
            offset_n = int(offset_type[len("pct_change_") :])
            prev = pick_helper_column_name(taken=set(data.columns) | {feature_name})
            rel = data.window(
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
            proj = ", ".join(quote_ident(c) for c in original_cols) + f", {wrapper} AS {quote_ident(feature_name)}"
            return rel.select(_raw_sql=proj)

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

        return data.window(func, feature_name, partition_by=partition_by, order_by=order_spec)

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

        rn = pick_helper_column_name(taken=set(data.columns) | {feature_name})
        q_rn = quote_ident(rn)
        tagged = data.with_row_number(rn)
        q_table = quote_ident(tagged.table_name)

        partition_match = " AND ".join(f"t2.{quote_ident(col)} IS t1.{quote_ident(col)}" for col in partition_by)
        null_sort = f"CASE WHEN t2.{quoted_order} IS NULL THEN 1 ELSE 0 END"

        if offset_type == "first_value":
            sort_clause = f"{null_sort}, t2.{quoted_order}, t2.{q_rn}"
        else:
            sort_clause = f"{null_sort} DESC, t2.{quoted_order} DESC, t2.{q_rn} DESC"

        subquery = (
            f"(SELECT t2.{quoted_source} FROM {q_table} t2 "  # nosec
            f"WHERE {partition_match} AND t2.{quoted_source} IS NOT NULL "
            f"ORDER BY {sort_clause} LIMIT 1)"
        )
        # ORDER BY the input-position tag so the fetched values line up with the input rows.
        sql = (
            f"SELECT {subquery} AS {quoted_feature} "  # nosec
            f"FROM {q_table} t1 ORDER BY t1.{q_rn}"
        )
        try:
            result_values = [row[0] for row in data.connection.execute(sql).fetchall()]
        finally:
            tagged.drop()
        return data.append_column(feature_name, result_values)
