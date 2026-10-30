"""DuckDB implementation for aggregation feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name, quote_ident

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
    AggregationSpec,
)
from mloda.community.feature_groups.data_operations.duckdb_helpers import nan_policy_agg_sql
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


class DuckdbAggregation(Implementation[AggregationFeatureGroup]):
    feature_group = AggregationFeatureGroup
    compute_framework = DuckDBFramework

    @classmethod
    def calculate(
        cls, group: type[AggregationFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.aggregate(data, features, compute_group=cls._compute_group)

    @classmethod
    def _compute_group(cls, data: Any, specs: list[AggregationSpec], partition_by: list[str]) -> DuckdbRelation:
        for spec in specs:
            if spec.agg_type not in _DUCKDB_AGG_FUNCS:
                raise unsupported_agg_type_error(spec.agg_type, _DUCKDB_AGG_FUNCS.keys(), framework="DuckDB")

        partition_cols = ", ".join(quote_ident(col) for col in partition_by)

        # mode/first/last need input order, which DuckDB 2 only guarantees via a row tag.
        qrn: str | None = None
        if any(spec.agg_type in ("mode", "first", "last") for spec in specs):
            rn = pick_helper_column_name(taken=set(data.columns) | {spec.feature_name for spec in specs})
            qrn = quote_ident(rn)
            data = data.with_row_number(rn)

        agg_parts: list[str] = []
        for spec in specs:
            source_sql = quote_ident(spec.source_col)
            if spec.mask_spec is not None:
                source_sql = build_sql_case_when(DuckDBFramework.mask_engine(), data, spec.mask_spec, source_sql)

            if spec.agg_type == "nunique":
                agg_expr = f"COUNT(DISTINCT {source_sql})"
            else:
                agg_func = _DUCKDB_AGG_FUNCS[spec.agg_type]
                agg_expr = nan_policy_agg_sql(data, spec.source_col, source_sql, spec.agg_type, agg_func, qrn)
                if spec.agg_type == "mode":
                    agg_expr += ".v"
                if spec.agg_type in ("first", "last"):
                    agg_expr = f"{agg_func}({source_sql} ORDER BY {qrn}) FILTER (WHERE {source_sql} IS NOT NULL)"
            agg_parts.append(f"{agg_expr} AS {quote_ident(spec.feature_name)}")

        # Use lazy relation methods (aggregate + order) instead of eager query()
        # so DuckDB defers execution until the result is consumed.
        select_expr = ", ".join(agg_parts)
        if partition_cols:
            select_expr = f"{partition_cols}, {select_expr}"
        rel: DuckdbRelation = data.aggregate(select_expr, partition_cols)
        if partition_cols:
            rel = rel.order(partition_cols)
        return rel
