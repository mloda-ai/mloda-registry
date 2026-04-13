"""DuckDB implementation for aggregation feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
)
from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import build_sql_case_when

# All aggregation types natively supported by DuckDB.
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
    "first": "FIRST",
    "last": "LAST",
}


class DuckdbAggregation(AggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {DuckDBFramework}

    @classmethod
    def _compute_group(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> DuckdbRelation:
        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)
        partition_cols = ", ".join(quote_ident(col) for col in partition_by)

        source_sql = quoted_source
        if mask_spec is not None:
            source_sql = build_sql_case_when(mask_spec, quoted_source)

        if agg_type == "nunique":
            agg_expr = f"COUNT(DISTINCT {source_sql})"
        else:
            agg_func = _DUCKDB_AGG_FUNCS.get(agg_type)
            if agg_func is None:
                raise unsupported_agg_type_error(agg_type, _DUCKDB_AGG_FUNCS.keys(), framework="DuckDB")
            agg_expr = f"{agg_func}({source_sql})"
            if agg_type in ("first", "last"):
                agg_expr += f" FILTER (WHERE {source_sql} IS NOT NULL)"

        # Use lazy relation methods (aggregate + order) instead of eager query()
        # so DuckDB defers execution until the result is consumed.
        rel = data._relation.aggregate(f"{partition_cols}, {agg_expr} AS {quoted_feature}", partition_cols)
        rel = rel.order(partition_cols)
        return DuckdbRelation(data.connection, rel)
