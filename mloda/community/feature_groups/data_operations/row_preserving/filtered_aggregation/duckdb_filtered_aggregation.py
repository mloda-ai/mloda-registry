"""DuckDB implementation for filtered aggregation feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.row_preserving.filtered_aggregation.base import (
    FilteredAggregationFeatureGroup,
)

# All basic aggregation types are natively supported by DuckDB window functions.
_DUCKDB_AGG_FUNCS: dict[str, str] = {
    "sum": "SUM",
    "avg": "AVG",
    "mean": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
}


def _quote_value(value: Any) -> str:
    """Convert a validated Python scalar to a safe SQL literal string.

    Only called after match_feature_group_criteria validates the type
    (str, int, float, or bool). Re-casts int and float as defense-in-depth
    against injection via subclass overrides of __str__.
    """
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(int(value))
    if isinstance(value, float):
        return str(float(value))
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    raise ValueError(f"Unsupported filter_value type: {type(value)}")


class DuckdbFilteredAggregation(FilteredAggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {DuckDBFramework}

    @classmethod
    def _compute_filtered(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        filter_column: str,
        filter_value: Any,
    ) -> DuckdbRelation:
        agg_func = _DUCKDB_AGG_FUNCS.get(agg_type)
        if agg_func is None:
            raise ValueError(f"Unsupported aggregation type for DuckDB: {agg_type}")

        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)
        quoted_filter_col = quote_ident(filter_column)
        partition_clause = ", ".join(quote_ident(col) for col in partition_by)

        # Safety: identifiers are quote_ident()-quoted, agg_func comes from
        # the hardcoded _DUCKDB_AGG_FUNCS whitelist, and filter_value is
        # type-validated in match_feature_group_criteria (str/int/float/bool)
        # then re-cast/escaped by _quote_value as defense-in-depth.
        quoted_value = _quote_value(filter_value)
        raw_sql = (  # nosec B608
            f"*, {agg_func}(CASE WHEN {quoted_filter_col} = {quoted_value} "
            f"THEN {quoted_source} END) "
            f"OVER (PARTITION BY {partition_clause}) AS {quoted_feature}"
        )
        result: DuckdbRelation = data.select(_raw_sql=raw_sql)
        return result
