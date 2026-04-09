"""DuckDB implementation for rank feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.row_preserving.rank.base import (
    RankFeatureGroup,
)

_RN_COL = "__mloda_orig_rn"

_DUCKDB_RANK_FUNCS: dict[str, str] = {
    "row_number": "ROW_NUMBER()",
    "rank": "RANK()",
    "dense_rank": "DENSE_RANK()",
    "percent_rank": "PERCENT_RANK()",
}


class DuckdbRank(RankFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {DuckDBFramework}

    @classmethod
    def _compute_rank(
        cls,
        data: Any,
        feature_name: str,
        partition_by: list[str],
        order_by: str,
        rank_type: str,
    ) -> DuckdbRelation:
        quoted_order = quote_ident(order_by)
        quoted_feature = quote_ident(feature_name)
        partition_clause = ", ".join(quote_ident(col) for col in partition_by)

        if rank_type.startswith("ntile_"):
            ntile_n = int(rank_type[len("ntile_") :])
            rank_expr = f"NTILE({ntile_n})"
        elif rank_type.startswith(("top_", "bottom_")):
            is_top = rank_type.startswith("top_")
            prefix = "top_" if is_top else "bottom_"
            n_val = int(rank_type[len(prefix) :])
            direction = "DESC" if is_top else "ASC"
            rank_expr = (
                f"(ROW_NUMBER() OVER "
                f"(PARTITION BY {partition_clause} ORDER BY {quoted_order} {direction} NULLS LAST) "
                f"<= {n_val})"
            )
        else:
            rank_func = _DUCKDB_RANK_FUNCS.get(rank_type)
            if rank_func is None:
                raise ValueError(f"Unsupported rank type for DuckDB: {rank_type}")
            rank_expr = rank_func

        # PyArrow parity: the reference computes ranks via Python index
        # mapping and returns results in original row order. DuckDB window
        # functions with ORDER BY reorder result rows; tag positions with
        # ROW_NUMBER() and ORDER BY qrn to restore input row order.
        qrn = quote_ident(_RN_COL)
        if rank_type.startswith(("top_", "bottom_")):
            # rank_expr already contains full window expression with boolean comparison
            sql = (
                f"SELECT *, "  # nosec
                f"{rank_expr} AS {quoted_feature}, "
                f"ROW_NUMBER() OVER () AS {qrn} "
                f"FROM __t ORDER BY {qrn}"
            )
        else:
            sql = (
                f"SELECT *, "  # nosec
                f"{rank_expr} OVER "
                f"(PARTITION BY {partition_clause} ORDER BY {quoted_order} ASC NULLS LAST) "
                f"AS {quoted_feature}, "
                f"ROW_NUMBER() OVER () AS {qrn} "
                f"FROM __t ORDER BY {qrn}"
            )
        new_rel = data._relation.query("__t", sql)
        # Drop the helper column
        result_rel = new_rel.project(
            ", ".join(quote_ident(c) for c in [col for col in new_rel.columns if col != _RN_COL])
        )
        return DuckdbRelation(data.connection, result_rel)
