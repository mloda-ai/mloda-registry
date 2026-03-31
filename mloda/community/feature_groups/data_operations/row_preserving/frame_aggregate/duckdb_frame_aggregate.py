"""DuckDB implementation for frame aggregate feature groups."""

from __future__ import annotations

from typing import Any, Optional, Set, Type, Union

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
)

_DUCKDB_AGG_FUNCS: dict[str, str] = {
    "sum": "SUM",
    "avg": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
    "std": "STDDEV_SAMP",
    "var": "VAR_SAMP",
    "median": "MEDIAN",
}

_RN_COL = "__mloda_rn__"


class DuckDBFrameAggregate(FrameAggregateFeatureGroup):
    SUPPORTED_FRAME_TYPES = {"rolling", "cumulative", "expanding"}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {DuckDBFramework}

    @classmethod
    def _compute_frame(
        cls,
        data: DuckdbRelation,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        agg_type: str,
        frame_type: str,
        frame_size: Optional[int] = None,
        frame_unit: Optional[str] = None,
    ) -> DuckdbRelation:
        agg_func = _DUCKDB_AGG_FUNCS.get(agg_type)
        if agg_func is None:
            raise ValueError(f"Unsupported aggregation type for DuckDB frame aggregate: {agg_type}")

        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)
        partition_clause = ", ".join(quote_ident(col) for col in partition_by)
        quoted_order = quote_ident(order_by)
        qrn = quote_ident(_RN_COL)

        null_sort = f"CASE WHEN {quoted_order} IS NULL THEN 1 ELSE 0 END"
        order_clause = f"{null_sort}, {quoted_order}"

        if frame_type in ("cumulative", "expanding"):
            frame_clause = "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        elif frame_type == "rolling":
            window_size = int(frame_size) if frame_size is not None else 1
            frame_clause = f"ROWS BETWEEN {window_size - 1} PRECEDING AND CURRENT ROW"
        elif frame_type == "time":
            raise ValueError("DuckDB time-based frame windows require RANGE which needs timestamp columns")
        else:
            raise ValueError(f"Unsupported frame type for DuckDB: {frame_type}")

        # Step 1: tag rows with original position
        rel = data._relation.project(f"*, ROW_NUMBER() OVER () AS {qrn}")  # nosec B608

        # Step 2: compute window function with frame
        raw_sql = (  # nosec B608
            f"*, {agg_func}({quoted_source}) OVER "
            f"(PARTITION BY {partition_clause} ORDER BY {order_clause} {frame_clause}) "
            f"AS {quoted_feature}"
        )
        rel = rel.project(raw_sql)

        # Step 3: restore original order, drop helper
        rel = rel.order(qrn)
        keep = ", ".join(quote_ident(c) for c in rel.columns if c != _RN_COL)
        rel = rel.project(keep)

        return DuckdbRelation(data.connection, rel)
