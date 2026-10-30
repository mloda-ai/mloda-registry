"""DuckDB implementation for offset feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name, quote_ident
from mloda_plugins.compute_framework.base_implementations.sql.sql_window import (
    OrderBy,
    Unbounded,
    WindowFrame,
)

from mloda.community.feature_groups.data_operations.duckdb_helpers import preserve_row_order
from mloda.community.feature_groups.data_operations.row_preserving.offset.base import (
    OffsetFeatureGroup,
)


class DuckdbOffset(Implementation[OffsetFeatureGroup]):
    feature_group = OffsetFeatureGroup
    compute_framework = DuckDBFramework

    @classmethod
    def calculate(
        cls, group: type[OffsetFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.offset(data, features, compute_offset=cls._compute_offset)

    @classmethod
    def _compute_offset(
        cls,
        data: DuckdbRelation,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        offset_type: str,
    ) -> DuckdbRelation:
        quoted_source = quote_ident(source_col)
        order_spec = [OrderBy(order_by, nulls="last")]

        offset_expr: str
        frame: WindowFrame | None = None

        if offset_type.startswith("lag_"):
            offset_n = int(offset_type[len("lag_") :])
            offset_expr = f"LAG({quoted_source}, {offset_n})"
        elif offset_type.startswith("lead_"):
            offset_n = int(offset_type[len("lead_") :])
            offset_expr = f"LEAD({quoted_source}, {offset_n})"
        elif offset_type.startswith("diff_"):
            offset_n = int(offset_type[len("diff_") :])
            # SQL binds OVER to the LAG term only, matching the original behaviour.
            offset_expr = f"{quoted_source} - LAG({quoted_source}, {offset_n})"
        elif offset_type.startswith("pct_change_"):
            offset_n = int(offset_type[len("pct_change_") :])

            # The LAG window is referenced multiple times inside the CASE, so it
            # cannot be a single window func. Precompute LAG into a helper column,
            # then project the CASE referencing that helper (no OVER).
            def compute_pct(rel: DuckdbRelation, rn: str) -> DuckdbRelation:
                prev = pick_helper_column_name(taken=set(rel.columns) | {feature_name})
                qprev = quote_ident(prev)
                rel = rel.window(
                    f"LAG({quoted_source}, {offset_n})",
                    prev,
                    partition_by=partition_by,
                    order_by=order_spec,
                )
                case_expr = (
                    f"CASE WHEN {qprev} IS NOT NULL AND {qprev} != 0 "
                    f"THEN ({quoted_source} - {qprev}) / CAST({qprev} AS DOUBLE) END"
                )
                keep = ", ".join(quote_ident(c) for c in rel.columns if c != prev)
                return rel.project(f"{keep}, {case_expr} AS {quote_ident(feature_name)}")

            return preserve_row_order(data, feature_name, compute_pct)
        elif offset_type == "first_value":
            offset_expr = f"FIRST_VALUE({quoted_source} IGNORE NULLS)"
            # PyArrow parity: the reference scans the entire partition for
            # first/last non-null. DuckDB default ordered-window frame
            # (UNBOUNDED PRECEDING to CURRENT ROW) would make LAST_VALUE
            # return the current row. Explicit UNBOUNDED FOLLOWING gives
            # partition-wide visibility.
            frame = WindowFrame("rows", Unbounded(), Unbounded())
        elif offset_type == "last_value":
            offset_expr = f"LAST_VALUE({quoted_source} IGNORE NULLS)"
            frame = WindowFrame("rows", Unbounded(), Unbounded())
        else:
            raise ValueError(f"Unsupported offset type for DuckDB: {offset_type}")

        # PyArrow parity: results come back in original row order.
        return preserve_row_order(
            data,
            feature_name,
            lambda rel, rn: rel.window(
                offset_expr, feature_name, partition_by=partition_by, order_by=order_spec, frame=frame
            ),
        )
