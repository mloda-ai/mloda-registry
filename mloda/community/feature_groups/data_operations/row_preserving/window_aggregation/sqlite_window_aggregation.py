"""SQLite implementation for window aggregation feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import build_sql_case_when
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)

# Aggregation types that SQLite supports natively in window functions.
_SQLITE_AGG_FUNCS: dict[str, str] = {
    "sum": "SUM",
    "avg": "AVG",
    "mean": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
}


class SqliteWindowAggregation(Implementation[WindowAggregationFeatureGroup]):
    feature_group = WindowAggregationFeatureGroup
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
        cls,
        group: type[WindowAggregationFeatureGroup],
        data: Any,
        features: FeatureSet,
        framework: ComputeFramework | None,
    ) -> Any:
        return group.aggregate(data, features, compute_window=cls._compute_window)

    @classmethod
    def _compute_window(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        order_by: str | None = None,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> SqliteRelation:
        agg_func = _SQLITE_AGG_FUNCS.get(agg_type)
        if agg_func is None:
            raise unsupported_agg_type_error(agg_type, _SQLITE_AGG_FUNCS.keys(), framework="SQLite")

        quoted_source = quote_ident(source_col)

        source_sql = quoted_source
        if mask_spec is not None:
            source_sql = build_sql_case_when(SqliteFramework.mask_engine(), data, mask_spec, quoted_source)

        return data.window(f"{agg_func}({source_sql})", feature_name, partition_by=partition_by)
