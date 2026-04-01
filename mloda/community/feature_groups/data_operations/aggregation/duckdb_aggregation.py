"""DuckDB implementation for aggregation feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
)

# All aggregation types natively supported by DuckDB.
_DUCKDB_AGG_FUNCS: dict[str, str] = {
    "sum": "SUM",
    "avg": "AVG",
    "mean": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
    "std": "STDDEV_SAMP",
    "var": "VAR_SAMP",
    "median": "MEDIAN",
    "mode": "MODE",
    "nunique": "COUNT_DISTINCT",  # handled specially: COUNT(DISTINCT col) syntax
    "first": "FIRST",
    "last": "LAST",
}


class DuckdbAggregation(AggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {DuckDBFramework}

    @classmethod
    def _compute_group(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
    ) -> DuckdbRelation:
        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)
        partition_cols = ", ".join(quote_ident(col) for col in partition_by)

        if agg_type == "nunique":
            agg_expr = f"COUNT(DISTINCT {quoted_source})"
        else:
            agg_func = _DUCKDB_AGG_FUNCS.get(agg_type)
            if agg_func is None:
                raise ValueError(f"Unsupported aggregation type for DuckDB: {agg_type}")
            agg_expr = f"{agg_func}({quoted_source})"

        # Use lazy relation methods (aggregate + order) instead of eager query()
        # so DuckDB defers execution until the result is consumed.
        rel = data._relation.aggregate(f"{partition_cols}, {agg_expr} AS {quoted_feature}", partition_cols)
        rel = rel.order(partition_cols)
        return DuckdbRelation(data.connection, rel)
