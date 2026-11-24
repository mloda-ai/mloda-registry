"""SQLite implementation for aggregation feature groups."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from mloda.provider import Cause, ComputeFramework, Divergence, DivergenceKind, FeatureSet, Implementation
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
    divergences = (
        Divergence(
            kind=DivergenceKind.VARIANT,
            variant="nan_as_null",
            cause=Cause.BACKEND,
            subtypes=frozenset({"sum", "avg", "mean"}),
            reason="skips NaN",
        ),
        Divergence(
            kind=DivergenceKind.VARIANT,
            variant="nan_as_null",
            cause=Cause.BACKEND,
            subtypes=frozenset({"count"}),
            reason="does not count NaN",
        ),
    )

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

        source_types = dict(zip(data.columns, data.types, strict=True))
        agg_parts: list[str] = []
        for spec in specs:
            source_sql = quote_ident(spec.source_col)
            if spec.mask_spec is not None:
                source_sql = build_sql_case_when(SqliteFramework.mask_engine(), data, spec.mask_spec, source_sql)
            agg_expr = f"{_SQLITE_AGG_FUNCS[spec.agg_type]}({source_sql})"
            if spec.agg_type == "count":
                agg_expr = f"CAST({agg_expr} AS INTEGER)"  # the declared type survives zero rows
            elif spec.agg_type == "avg" or spec.agg_type == "mean":
                agg_expr = f"CAST({agg_expr} AS REAL)"  # the declared type survives zero rows
            else:
                source_type = source_types[spec.source_col]
                if pa.types.is_integer(source_type) or pa.types.is_floating(source_type):
                    sql_type = "INTEGER" if pa.types.is_integer(source_type) else "REAL"
                    agg_expr = f"CAST({agg_expr} AS {sql_type})"  # the declared type survives zero rows
            agg_parts.append(f"{agg_expr} AS {quote_ident(spec.feature_name)}")

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
