"""DuckDB implementation of ffill-by-time."""

from __future__ import annotations

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name, quote_ident
from mloda_plugins.compute_framework.base_implementations.sql.sql_window import (
    CurrentRow,
    OrderBy,
    Unbounded,
    WindowFrame,
)

from mloda.community.feature_groups.data_operations.duckdb_helpers import (
    assert_duckdb_source_col_present,
    duckdb_drop_rn_restore,
)
from mloda.community.feature_groups.data_operations.row_preserving.ffill.base import FfillFeatureGroup


class DuckdbFfill(FfillFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {DuckDBFramework}

    @classmethod
    def _assert_source_column_present(cls, data: DuckdbRelation, source_col: str) -> None:
        assert_duckdb_source_col_present(data, source_col)

    @classmethod
    def _compute_ffill(
        cls,
        data: DuckdbRelation,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
    ) -> DuckdbRelation:
        rn = pick_helper_column_name(taken=set(data.columns) | {feature_name})

        quoted_source = quote_ident(source_col)

        # ffill = last non-null value seen so far in the ordered partition.
        # DuckDB supports LAST_VALUE(... IGNORE NULLS) over an unbounded-preceding
        # frame. The frame / order_by / partition_by are expressed structurally via
        # the typed window helper; only the aggregate expression is a raw fragment
        # (mirroring frame_aggregate's ``f"{agg_func}({source_sql})"``).
        func = f"LAST_VALUE({quoted_source} IGNORE NULLS)"
        order_spec: list[OrderBy] = [OrderBy(order_by)]
        frame = WindowFrame("rows", Unbounded(), CurrentRow())

        # PyArrow parity: preserve original input row order. Tag rows, compute the
        # window, then restore the tag order and drop the helper column.
        rel = data.with_row_number(rn)
        rel = rel.window(
            func,
            feature_name,
            partition_by=partition_by,
            order_by=order_spec,
            frame=frame,
        )
        return duckdb_drop_rn_restore(rel, rn)
