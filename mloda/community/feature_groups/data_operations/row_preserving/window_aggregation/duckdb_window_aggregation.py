"""DuckDB implementation for window aggregation feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

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
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
    "std": "STDDEV_SAMP",
    "var": "VAR_SAMP",
    "median": "MEDIAN",
    "mode": "MODE",
    "nunique": "COUNT_DISTINCT",  # handled specially: COUNT(DISTINCT col) syntax
    "first": "FIRST_VALUE",
    "last": "LAST_VALUE",
}

_RN_COL = "__mloda_rn__"


class DuckdbWindowAggregation(WindowAggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {DuckDBFramework}

    @classmethod
    def _compute_window(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
    ) -> DuckdbRelation:
        # Safety: _raw_sql is composed entirely from quote_ident()-quoted identifiers
        # and hardcoded SQL function names from _DUCKDB_AGG_FUNCS. No user-controlled
        # strings are interpolated without quoting.
        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)
        partition_clause = ", ".join(quote_ident(col) for col in partition_by)

        if agg_type == "nunique":
            raw_sql = f"*, COUNT(DISTINCT {quoted_source}) OVER (PARTITION BY {partition_clause}) AS {quoted_feature}"  # nosec B608
            result: DuckdbRelation = data.select(_raw_sql=raw_sql)
            return result

        if agg_type in ("first", "last"):
            return cls._compute_first_last(data, feature_name, source_col, partition_by, agg_type)

        agg_func = _DUCKDB_AGG_FUNCS[agg_type]
        raw_sql = f"*, {agg_func}({quoted_source}) OVER (PARTITION BY {partition_clause}) AS {quoted_feature}"  # nosec B608
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
    ) -> DuckdbRelation:
        """Compute FIRST_VALUE/LAST_VALUE with correct whole-partition semantics.

        FIRST_VALUE and LAST_VALUE are frame-sensitive SQL window functions.
        The default frame (ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        makes LAST_VALUE return the current row's value instead of the
        partition-wide last. An explicit UNBOUNDED frame fixes this, but
        DuckDB may reorder rows when a frame clause is present.

        Solution: tag rows with ROW_NUMBER before the window function,
        compute with the full frame, then ORDER BY the tag to restore
        the original row order.
        """
        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)
        partition_clause = ", ".join(quote_ident(col) for col in partition_by)
        qrn = quote_ident(_RN_COL)
        agg_func = _DUCKDB_AGG_FUNCS[agg_type]

        # Step 1: tag rows with their original position
        rel = data._relation.project(f"*, ROW_NUMBER() OVER () AS {qrn}")  # nosec B608

        # Step 2: compute with full frame (may reorder rows)
        rel = rel.project(  # nosec B608
            f"*, {agg_func}({quoted_source} IGNORE NULLS) "
            f"OVER (PARTITION BY {partition_clause} "
            f"ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS {quoted_feature}"
        )

        # Step 3: restore original row order, drop helper column
        rel = rel.order(qrn)
        keep = ", ".join(quote_ident(c) for c in rel.columns if c != _RN_COL)
        rel = rel.project(keep)

        return DuckdbRelation(data._connection, rel)
