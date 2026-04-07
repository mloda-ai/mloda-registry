"""DuckDB implementation for window aggregation feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)

# All aggregation types are natively supported by DuckDB window functions.
_DUCKDB_AGG_FUNCS: dict[str, str] = {
    "sum": "SUM",
    "avg": "AVG",
    "mean": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
    "std": "STDDEV_POP",
    "var": "VAR_POP",
    "std_pop": "STDDEV_POP",
    "std_samp": "STDDEV_SAMP",
    "var_pop": "VAR_POP",
    "var_samp": "VAR_SAMP",
    "median": "MEDIAN",
    "mode": "MODE",
    "nunique": "COUNT_DISTINCT",  # handled specially: COUNT(DISTINCT col) syntax
    "first": "FIRST_VALUE",
    "last": "LAST_VALUE",
}

_RN_COL = "__mloda_rn__"


class DuckdbWindowAggregation(WindowAggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {DuckDBFramework}

    @classmethod
    def _compute_window(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        order_by: str | None = None,
    ) -> DuckdbRelation:
        # Safety: _raw_sql is composed entirely from quote_ident()-quoted identifiers
        # and hardcoded SQL function names from _DUCKDB_AGG_FUNCS. No user-controlled
        # strings are interpolated without quoting.
        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)
        partition_clause = ", ".join(quote_ident(col) for col in partition_by)

        if agg_type == "nunique":
            raw_sql = f"*, COUNT(DISTINCT {quoted_source}) OVER (PARTITION BY {partition_clause}) AS {quoted_feature}"
            result: DuckdbRelation = data.select(_raw_sql=raw_sql)
            return result

        if agg_type in ("first", "last"):
            return cls._compute_first_last(data, feature_name, source_col, partition_by, agg_type, order_by)

        agg_func = _DUCKDB_AGG_FUNCS[agg_type]
        raw_sql = f"*, {agg_func}({quoted_source}) OVER (PARTITION BY {partition_clause}) AS {quoted_feature}"
        result = data.select(_raw_sql=raw_sql)
        return result

    @classmethod
    def _compute_first_last(
        cls,
        data: DuckdbRelation,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        order_by: str | None = None,
    ) -> DuckdbRelation:
        """Compute FIRST_VALUE/LAST_VALUE with ORDER BY for deterministic results.

        Uses ROW_NUMBER to tag original row positions, computes the window
        function with an explicit UNBOUNDED frame and ORDER BY clause,
        then restores original row order.
        """
        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)
        partition_clause = ", ".join(quote_ident(col) for col in partition_by)
        qrn = quote_ident(_RN_COL)
        agg_func = _DUCKDB_AGG_FUNCS[agg_type]

        order_clause = f"ORDER BY {quote_ident(order_by)}" if order_by else ""

        # Step 1: tag rows with their original position
        rel = data._relation.project(f"*, ROW_NUMBER() OVER () AS {qrn}")

        # Step 2: compute with full frame and ORDER BY for deterministic results
        rel = rel.project(
            f"*, {agg_func}({quoted_source} IGNORE NULLS) "
            f"OVER (PARTITION BY {partition_clause} {order_clause} "
            f"ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS {quoted_feature}"
        )

        # Step 3: restore original row order, drop helper column
        rel = rel.order(qrn)
        keep = ", ".join(quote_ident(c) for c in rel.columns if c != _RN_COL)
        rel = rel.project(keep)

        return DuckdbRelation(data.connection, rel)
