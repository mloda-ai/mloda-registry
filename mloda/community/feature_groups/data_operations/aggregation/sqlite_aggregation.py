"""SQLite implementation for aggregation feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation, _next_table_name

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
    AggregationSpec,
)
from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import build_sql_case_when

# Aggregation types that SQLite supports natively.
_SQLITE_AGG_FUNCS: dict[str, str] = {
    "sum": "SUM",
    "avg": "AVG",
    "mean": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
}


class SqliteAggregation(Implementation[AggregationFeatureGroup]):
    feature_group = AggregationFeatureGroup
    compute_framework = SqliteFramework
    supported = frozenset(_SQLITE_AGG_FUNCS)
    divergent = {
        "sum": "skips NaN",
        "avg": "skips NaN",
        "mean": "skips NaN",
        "count": "does not count NaN",
    }

    @classmethod
    def calculate(
        cls, group: type[AggregationFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.aggregate(data, features, compute_group=cls._compute_group)

    @classmethod
    def _compute_group(
        cls, data: SqliteRelation, specs: list[AggregationSpec], partition_by: list[str]
    ) -> SqliteRelation:
        for spec in specs:
            if spec.agg_type not in _SQLITE_AGG_FUNCS:
                raise unsupported_agg_type_error(spec.agg_type, _SQLITE_AGG_FUNCS.keys(), framework="SQLite")

        partition_cols = ", ".join(quote_ident(col) for col in partition_by)

        agg_parts: list[str] = []
        for spec in specs:
            source_sql = quote_ident(spec.source_col)
            if spec.mask_spec is not None:
                source_sql = build_sql_case_when(SqliteFramework.mask_engine(), data, spec.mask_spec, source_sql)
            agg_parts.append(f"{_SQLITE_AGG_FUNCS[spec.agg_type]}({source_sql}) AS {quote_ident(spec.feature_name)}")

        select_cols = f"{partition_cols}, " if partition_cols else ""
        group_by_clause = f" GROUP BY {partition_cols}" if partition_cols else ""
        new_name = _next_table_name()
        sql = (
            f"CREATE TEMP VIEW {quote_ident(new_name)} AS "  # nosec
            f"SELECT {select_cols}"
            f"{', '.join(agg_parts)} "
            f"FROM {quote_ident(data.table_name)}"
            f"{group_by_clause}"
        )
        data.connection.execute(sql)
        return SqliteRelation(data.connection, new_name, _is_view=True)
