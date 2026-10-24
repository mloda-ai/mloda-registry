"""DuckDB implementation of resample.

Floors the time column with the shared ``floor_expr`` epoch-anchored
expression (``DATE_TRUNC`` for n=1, ``time_bucket`` with a
``DATE '1970-01-01'`` origin for n>1 -- NOT the native 2000-01-03 anchor) so
buckets align with the PyArrow oracle. SQL ``SUM`` / ``AVG`` ignore nulls and
return NULL for all-null groups, and ``COUNT(col)`` counts non-null values, so
the null semantics already match the oracle. Float ``max`` also skips NaN,
matching PyArrow's ``pc.max``.
"""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.duckdb_helpers import (
    floor_expr,
    nan_policy_agg_sql,
)
from mloda.community.feature_groups.data_operations.row_changing.resample.base import (
    RESAMPLE_AGGS,
    ResampleAgg,
    ResampleFeatureGroup,
)

# Resample agg -> DuckDB aggregate function. SUM/AVG/MIN/MAX ignore nulls and
# return NULL for all-null groups; COUNT(col) counts non-null values. Float
# max additionally skips NaN.
_DUCKDB_AGG_FUNCS: dict[str, str] = {
    "mean": "AVG",
    "sum": "SUM",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
}


class DuckdbResample(Implementation[ResampleFeatureGroup]):
    """DuckDB backend for resample."""

    feature_group = ResampleFeatureGroup
    compute_framework = DuckDBFramework

    @classmethod
    def calculate(
        cls, group: type[ResampleFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.resample(data, features, compute_resample=cls._compute_resample)

    @classmethod
    def _compute_resample(
        cls,
        data: DuckdbRelation,
        aggs: list[ResampleAgg],
        time_column: str,
        partition_by: list[str],
        n: int,
        unit: str,
    ) -> DuckdbRelation:
        for spec in aggs:
            if spec.agg not in _DUCKDB_AGG_FUNCS:
                raise ValueError(
                    f"Unsupported resample agg {spec.agg!r} for DuckDB; supported: {sorted(RESAMPLE_AGGS)}."
                )

        quoted_time = quote_ident(time_column)
        floor_sql = floor_expr(quoted_time, n, unit)

        # Group keys: partition columns first, then the floored bucket. The
        # bucket is always a key, so the group list is never empty (whole-table
        # case has an empty partition list).
        partition_quoted = [quote_ident(col) for col in partition_by]
        group_exprs = [*partition_quoted, floor_sql]

        agg_parts = [
            f"{nan_policy_agg_sql(data, spec.source_col, quote_ident(spec.source_col), spec.agg, _DUCKDB_AGG_FUNCS[spec.agg])}"
            f" AS {quote_ident(spec.feature_name)}"
            for spec in aggs
        ]
        select_parts = [*partition_quoted, f"{floor_sql} AS {quoted_time}", *agg_parts]

        agg_sql = ", ".join(select_parts)
        group_sql = ", ".join(group_exprs)
        result: DuckdbRelation = data.aggregate(agg_sql, group_sql)
        # Order deterministically so the test's two separate ``to_arrow_table()``
        # extractions (bucket column, then agg column) stay row-aligned.
        result = result.order(group_sql)
        return result
