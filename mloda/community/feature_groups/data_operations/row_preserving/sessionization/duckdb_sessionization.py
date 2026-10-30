"""DuckDB implementation of gap-threshold sessionization.

The session id is computed natively with two stacked window functions: an
inner ``LAG``/``date_diff`` produces the per-row ``is_new`` flag (1 when the
row begins a new session), and an outer running ``SUM`` over the same ordering
turns the flags into a 0-based session id. ``date_diff('second', ...)`` is
exact integer second arithmetic, so the strict ``gap > threshold`` boundary
needs no rounding.
"""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name, quote_ident

from mloda.community.feature_groups.data_operations.duckdb_helpers import preserve_row_order
from mloda.community.feature_groups.data_operations.row_preserving.sessionization.base import (
    SessionizationFeatureGroup,
)


class DuckdbSessionization(Implementation[SessionizationFeatureGroup]):
    feature_group = SessionizationFeatureGroup
    compute_framework = DuckDBFramework

    @classmethod
    def calculate(
        cls,
        group: type[SessionizationFeatureGroup],
        data: Any,
        features: FeatureSet,
        framework: ComputeFramework | None,
    ) -> Any:
        return group.sessionize(data, features, compute_session=cls._compute_session)

    @classmethod
    def _compute_session(
        cls,
        data: DuckdbRelation,
        feature_name: str,
        order_col: str,
        threshold_seconds: int,
        partition_by: list[str],
    ) -> DuckdbRelation:
        def compute(rel: DuckdbRelation, rn: str) -> DuckdbRelation:
            is_new = pick_helper_column_name(taken=set(rel.columns) | {feature_name})

            # Safety: every identifier (column names, helper is_new, feature name,
            # view name) is quoted via quote_ident(); the only interpolated literal
            # is the integer threshold_seconds. No user-controlled string is inlined unquoted.
            q_order = quote_ident(order_col)
            q_is_new = quote_ident(is_new)
            q_feature = quote_ident(feature_name)
            partition_cols = [quote_ident(c) for c in partition_by]

            partition_clause = f"PARTITION BY {', '.join(partition_cols)} " if partition_cols else ""
            window_def = f"w AS ({partition_clause}ORDER BY {q_order})"
            order_keys = ", ".join([*partition_cols, q_order])

            is_new_expr = (
                f"CASE WHEN LAG({q_order}) OVER w IS NULL "
                f"OR date_diff('second', LAG({q_order}) OVER w, {q_order}) > {threshold_seconds} "
                f"THEN 1 ELSE 0 END"
            )
            session_expr = (
                f"CAST(SUM({q_is_new}) OVER ("
                f"ORDER BY {order_keys} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
                f") - 1 AS BIGINT)"
            )

            select_cols = ", ".join(quote_ident(c) for c in rel.columns)
            sql = (
                f"SELECT {select_cols}, {session_expr} AS {q_feature} "  # nosec
                f"FROM (SELECT *, {is_new_expr} AS {q_is_new} FROM session_src WINDOW {window_def})"
            )
            return rel.query("session_src", sql)

        return preserve_row_order(data, feature_name, compute)
