"""DuckDB implementation for window aggregation feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import build_sql_case_when
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
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> DuckdbRelation:
        # Safety: _raw_sql is composed entirely from quote_ident()-quoted identifiers
        # and hardcoded SQL function names from _DUCKDB_AGG_FUNCS. No user-controlled
        # strings are interpolated without quoting.
        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)
        partition_clause = ", ".join(quote_ident(col) for col in partition_by)

        source_sql = quoted_source
        if mask_spec is not None:
            source_sql = build_sql_case_when(mask_spec, quoted_source)

        if agg_type == "nunique":
            raw_sql = f"*, COUNT(DISTINCT {source_sql}) OVER (PARTITION BY {partition_clause}) AS {quoted_feature}"
            result: DuckdbRelation = data.select(_raw_sql=raw_sql)
            return result

        if agg_type in ("first", "last"):
            return cls._compute_first_last(data, feature_name, source_sql, partition_by, agg_type, order_by)

        agg_func = _DUCKDB_AGG_FUNCS.get(agg_type)
        if agg_func is None:
            raise unsupported_agg_type_error(agg_type, _DUCKDB_AGG_FUNCS.keys(), framework="DuckDB")
        raw_sql = f"*, {agg_func}({source_sql}) OVER (PARTITION BY {partition_clause}) AS {quoted_feature}"
        result = data.select(_raw_sql=raw_sql)
        return result

    @classmethod
    def _compute_first_last(
        cls,
        data: DuckdbRelation,
        feature_name: str,
        source_sql: str,
        partition_by: list[str],
        agg_type: str,
        order_by: str | None = None,
    ) -> DuckdbRelation:
        """Compute FIRST_VALUE/LAST_VALUE with ORDER BY for deterministic results.

        PyArrow parity: PyArrow group_by().aggregate() sees the entire
        partition at once. DuckDB default ordered-window frame is
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW, which makes
        LAST_VALUE return the current row instead of the partition-wide
        last. Explicit ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED
        FOLLOWING ensures full-partition visibility.

        Uses ROW_NUMBER to tag original row positions, computes the window
        function with the explicit UNBOUNDED frame, then restores original
        row order.

        *source_sql* is a quoted identifier or a ``CASE WHEN`` expression
        when a mask is active.
        """
        quoted_feature = quote_ident(feature_name)
        partition_clause = ", ".join(quote_ident(col) for col in partition_by)
        qrn = quote_ident(_RN_COL)
        agg_func = _DUCKDB_AGG_FUNCS[agg_type]

        order_clause = f"ORDER BY {quote_ident(order_by)}" if order_by else ""

        # Step 1: tag rows with their original position
        rel = data._relation.project(f"*, ROW_NUMBER() OVER () AS {qrn}")

        # Step 2: compute with full frame and ORDER BY for deterministic results
        rel = rel.project(
            f"*, {agg_func}({source_sql} IGNORE NULLS) "
            f"OVER (PARTITION BY {partition_clause} {order_clause} "
            f"ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS {quoted_feature}"
        )

        # Step 3: restore original row order, drop helper column
        rel = rel.order(qrn)
        keep = ", ".join(quote_ident(c) for c in rel.columns if c != _RN_COL)
        rel = rel.project(keep)

        return DuckdbRelation(data.connection, rel)
