"""DuckDB implementation for window aggregation feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name, quote_ident
from mloda_plugins.compute_framework.base_implementations.sql.sql_window import Unbounded, WindowFrame

from mloda.community.feature_groups.data_operations.duckdb_agg_constants import DUCKDB_AGG_FUNCS
from mloda.community.feature_groups.data_operations.duckdb_helpers import duckdb_drop_rn_restore
from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import build_sql_case_when
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)

# Window functions use FIRST_VALUE/LAST_VALUE (group aggregation uses FIRST/LAST).
_DUCKDB_AGG_FUNCS: dict[str, str] = {
    **DUCKDB_AGG_FUNCS,
    "first": "FIRST_VALUE",
    "last": "LAST_VALUE",
}


class DuckdbWindowAggregation(WindowAggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {DuckDBFramework}

    @classmethod
    def _compute_window(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        order_by: str | None = None,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> DuckdbRelation:
        # Safety: the projection string is composed entirely from quote_ident()-quoted
        # identifiers and hardcoded SQL function names from _DUCKDB_AGG_FUNCS. No
        # user-controlled strings are interpolated without quoting.
        quoted_source = quote_ident(source_col)

        source_sql = quoted_source
        if mask_spec is not None:
            source_sql = build_sql_case_when(mask_spec, quoted_source)

        if agg_type == "nunique":
            result: DuckdbRelation = data.window(
                f"COUNT(DISTINCT {source_sql})", feature_name, partition_by=partition_by
            )
            return result

        if agg_type in ("first", "last"):
            return cls._compute_first_last(data, feature_name, source_sql, partition_by, agg_type, order_by)

        agg_func = _DUCKDB_AGG_FUNCS.get(agg_type)
        if agg_func is None:
            raise unsupported_agg_type_error(agg_type, _DUCKDB_AGG_FUNCS.keys(), framework="DuckDB")
        result = data.window(f"{agg_func}({source_sql})", feature_name, partition_by=partition_by)
        return result

    @classmethod
    def _compute_first_last(
        cls,
        data: DuckdbRelation,
        feature_name: str,
        source_sql: str,
        partition_by: list[str],
        agg_type: str,
        order_by: str | None = None,
    ) -> DuckdbRelation:
        """PyArrow parity: DuckDB's default ordered-window frame is
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW, which makes
        LAST_VALUE return the current row instead of the partition-wide
        last. Explicit UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING restores
        full-partition visibility to match PyArrow group_by().aggregate()."""
        rn = pick_helper_column_name(taken=set(data.columns) | {feature_name})
        agg_func = _DUCKDB_AGG_FUNCS[agg_type]

        # Step 1: tag rows with their original position
        rel = data.with_row_number(rn)

        # Step 2: compute with full frame and ORDER BY for deterministic results
        rel = rel.window(
            f"{agg_func}({source_sql} IGNORE NULLS)",
            feature_name,
            partition_by=partition_by,
            order_by=([order_by] if order_by else ()),
            frame=WindowFrame("rows", Unbounded(), Unbounded()),
        )

        # Step 3: restore original row order, drop helper column
        return duckdb_drop_rn_restore(rel, rn)
