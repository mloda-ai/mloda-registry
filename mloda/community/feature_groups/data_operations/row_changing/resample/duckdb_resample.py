"""DuckDB implementation of resample.

Floors the time column with the SAME epoch-anchored expression as
``duckdb_time_bucketization`` (``DATE_TRUNC`` for n=1, ``time_bucket`` with a
``DATE '1970-01-01'`` origin for n>1 -- NOT the native 2000-01-03 anchor) so
buckets align with the PyArrow oracle. SQL ``SUM`` / ``AVG`` ignore nulls and
return NULL for all-null groups, and ``COUNT(col)`` counts non-null values, so
the null semantics already match the oracle.
"""

from __future__ import annotations

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.row_changing.resample.base import (
    RESAMPLE_AGGS,
    ResampleFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.duckdb_time_bucketization import (
    _floor_expr,
)

# Resample agg -> DuckDB aggregate function. SUM/AVG/MIN/MAX ignore nulls and
# return NULL for all-null groups; COUNT(col) counts non-null values.
_DUCKDB_AGG_FUNCS: dict[str, str] = {
    "mean": "AVG",
    "sum": "SUM",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
}


class DuckdbResample(ResampleFeatureGroup):
    """DuckDB backend for resample."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {DuckDBFramework}

    @classmethod
    def _column_types(cls, data: DuckdbRelation) -> dict[str, str]:
        underlying = data._relation
        return dict(zip(list(underlying.columns), [str(t) for t in underlying.types]))

    @classmethod
    def _assert_time_column_present(cls, data: DuckdbRelation, time_column: str) -> None:
        types = cls._column_types(data)
        if time_column not in types:
            raise ValueError(
                f"time_column {time_column!r} is not present in the DuckDB relation; available: {list(types)}."
            )

    @classmethod
    def _assert_source_column_present(cls, data: DuckdbRelation, source_col: str) -> None:
        types = cls._column_types(data)
        if source_col not in types:
            raise ValueError(
                f"Source column {source_col!r} is not present in the DuckDB relation; available: {list(types)}."
            )

    @classmethod
    def _compute_resample(
        cls,
        data: DuckdbRelation,
        feature_name: str,
        source_col: str,
        time_column: str,
        partition_by: list[str],
        n: int,
        unit: str,
        agg: str,
    ) -> DuckdbRelation:
        agg_func = _DUCKDB_AGG_FUNCS.get(agg)
        if agg_func is None:
            raise ValueError(f"Unsupported resample agg {agg!r} for DuckDB; supported: {sorted(RESAMPLE_AGGS)}.")

        quoted_source = quote_ident(source_col)
        quoted_time = quote_ident(time_column)
        quoted_feature = quote_ident(feature_name)
        floor_expr = _floor_expr(quoted_time, n, unit)

        # Group keys: partition columns first, then the floored bucket. The
        # bucket is always a key, so the group list is never empty (whole-table
        # case has an empty partition list).
        partition_quoted = [quote_ident(col) for col in partition_by]
        group_exprs = [*partition_quoted, floor_expr]

        select_parts = [
            *partition_quoted,
            f"{floor_expr} AS {quoted_time}",
            f"{agg_func}({quoted_source}) AS {quoted_feature}",
        ]

        agg_sql = ", ".join(select_parts)
        group_sql = ", ".join(group_exprs)
        result: DuckdbRelation = data.aggregate(agg_sql, group_sql)
        # Order deterministically so the test's two separate ``to_arrow_table()``
        # extractions (bucket column, then agg column) stay row-aligned.
        result = result.order(group_sql)
        return result
