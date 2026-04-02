"""DuckDB implementation for offset feature groups."""

from __future__ import annotations

from typing import Set, Type, Union

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.row_preserving.offset.base import (
    OffsetFeatureGroup,
)

_RN_COL = "__mloda_orig_rn__"


class DuckdbOffset(OffsetFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {DuckDBFramework}

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
        quoted_order = quote_ident(order_by)
        quoted_feature = quote_ident(feature_name)
        partition_clause = ", ".join(quote_ident(col) for col in partition_by)

        window_clause = f"PARTITION BY {partition_clause} ORDER BY {quoted_order} ASC NULLS LAST"

        if offset_type.startswith("lag_"):
            offset_n = int(offset_type[len("lag_") :])
            offset_expr = f"LAG({quoted_source}, {offset_n})"
        elif offset_type.startswith("lead_"):
            offset_n = int(offset_type[len("lead_") :])
            offset_expr = f"LEAD({quoted_source}, {offset_n})"
        elif offset_type.startswith("diff_"):
            offset_n = int(offset_type[len("diff_") :])
            offset_expr = f"{quoted_source} - LAG({quoted_source}, {offset_n})"
        elif offset_type.startswith("pct_change_"):
            offset_n = int(offset_type[len("pct_change_") :])
            prev = f"LAG({quoted_source}, {offset_n})"
            offset_expr = f"CASE WHEN {prev} IS NOT NULL AND {prev} != 0 THEN ({quoted_source} - {prev}) / CAST({prev} AS DOUBLE) END"
        elif offset_type == "first_value":
            offset_expr = f"FIRST_VALUE({quoted_source} IGNORE NULLS)"
            window_clause += " ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING"
        elif offset_type == "last_value":
            offset_expr = f"LAST_VALUE({quoted_source} IGNORE NULLS)"
            window_clause += " ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING"
        else:
            raise ValueError(f"Unsupported offset type for DuckDB: {offset_type}")

        # Use query() to preserve original row order
        qrn = quote_ident(_RN_COL)
        sql = (
            f"SELECT *, "  # nosec
            f"{offset_expr} OVER ({window_clause}) AS {quoted_feature}, "
            f"ROW_NUMBER() OVER () AS {qrn} "
            f"FROM __t ORDER BY {qrn}"
        )
        new_rel = data._relation.query("__t", sql)
        result_rel = new_rel.project(
            ", ".join(quote_ident(c) for c in [col for col in new_rel.columns if col != _RN_COL])
        )
        return DuckdbRelation(data.connection, result_rel)
