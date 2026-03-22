"""DuckDB implementation for group aggregation feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.group_aggregation.base import (
    GroupAggregationFeatureGroup,
)

# All aggregation types natively supported by DuckDB.
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
    "nunique": "COUNT_DISTINCT",
    "first": "FIRST",
    "last": "LAST",
}


class DuckdbGroupAggregation(GroupAggregationFeatureGroup):
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

        # Use relation.query() to execute GROUP BY against the relation.
        # The virtual table name "__t" binds to data._relation in the query.
        # ORDER BY ensures deterministic row order (DuckDB re-executes lazily).
        sql = (
            f"SELECT {partition_cols}, {agg_expr} AS {quoted_feature} "  # nosec B608
            f"FROM __t GROUP BY {partition_cols} ORDER BY {partition_cols}"
        )
        new_rel = data._relation.query("__t", sql)
        return DuckdbRelation(data._connection, new_rel)
