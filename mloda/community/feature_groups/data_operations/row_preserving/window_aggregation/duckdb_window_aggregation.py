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


class DuckdbWindowAggregation(WindowAggregationFeatureGroup):
    """All operations are fully lazy, composing SQL window functions via
    DuckdbRelation.select(_raw_sql=...) without materializing intermediate results.
    """

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
            raw_sql = f"*, COUNT(DISTINCT {quoted_source}) OVER (PARTITION BY {partition_clause}) AS {quoted_feature}"
        elif agg_type in ("first", "last"):
            agg_func = _DUCKDB_AGG_FUNCS[agg_type]
            raw_sql = (
                f"*, {agg_func}({quoted_source} IGNORE NULLS) "
                f"OVER (PARTITION BY {partition_clause}) AS {quoted_feature}"
            )
        else:
            agg_func = _DUCKDB_AGG_FUNCS[agg_type]
            raw_sql = f"*, {agg_func}({quoted_source}) OVER (PARTITION BY {partition_clause}) AS {quoted_feature}"

        result: DuckdbRelation = data.select(_raw_sql=raw_sql)
        return result
