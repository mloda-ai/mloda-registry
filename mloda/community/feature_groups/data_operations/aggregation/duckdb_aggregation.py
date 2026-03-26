"""DuckDB implementation for column aggregation feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.aggregation.base import (
    ColumnAggregationFeatureGroup,
)

_DUCKDB_AGG_FUNCS: dict[str, str] = {
    "sum": "SUM",
    "min": "MIN",
    "max": "MAX",
    "avg": "AVG",
    "mean": "AVG",
    "count": "COUNT",
    "std": "STDDEV_POP",
    "var": "VAR_POP",
    "median": "MEDIAN",
}


class DuckdbColumnAggregation(ColumnAggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {DuckDBFramework}

    @classmethod
    def _compute_aggregation(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        agg_type: str,
    ) -> DuckdbRelation:
        agg_func = _DUCKDB_AGG_FUNCS.get(agg_type)
        if agg_func is None:
            raise ValueError(f"Unsupported aggregation type for DuckDB: {agg_type}")

        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)

        raw_sql = f"*, {agg_func}({quoted_source}) OVER () AS {quoted_feature}"
        result: DuckdbRelation = data.select(_raw_sql=raw_sql)
        return result
