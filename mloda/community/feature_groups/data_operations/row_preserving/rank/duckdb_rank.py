"""DuckDB implementation for rank feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sql.sql_window import OrderBy

from mloda.community.feature_groups.data_operations.reserved_columns import assert_no_reserved_columns
from mloda.community.feature_groups.data_operations.row_preserving.rank.base import (
    RankFeatureGroup,
)

_RN_COL = "__mloda_orig_rn"
_RANK_RN_COL = "__mloda_rank_rn__"

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
        assert_no_reserved_columns(data.columns, framework="DuckDB", operation="rank")

        order_spec = [OrderBy(order_by, nulls="last")]

        if rank_type.startswith(("top_", "bottom_")):
            is_top = rank_type.startswith("top_")
            prefix = "top_" if is_top else "bottom_"
            n_val = int(rank_type[len(prefix) :])
            qhelper = quote_ident(_RANK_RN_COL)
            # PyArrow parity: tag original physical order before windowing so it
            # can be restored after DuckDB reorders rows. The boolean comparison
            # wraps a ROW_NUMBER() window, so compute the row number into a helper
            # column first, then project the comparison (no OVER). DESC for top_,
            # ASC for bottom_.
            rel: DuckdbRelation = data.with_row_number(_RN_COL)
            rel = rel.window(
                "ROW_NUMBER()",
                _RANK_RN_COL,
                partition_by=partition_by,
                order_by=[OrderBy(order_by, descending=is_top, nulls="last")],
            )
            rel = rel.project(f"*, ({qhelper} <= {n_val}) AS {quote_ident(feature_name)}")
            rel = rel.order(quote_ident(_RN_COL))
            keep = ", ".join(quote_ident(c) for c in rel.columns if c not in (_RN_COL, _RANK_RN_COL))
            return rel.project(keep)

        if rank_type.startswith("ntile_"):
            ntile_n = int(rank_type[len("ntile_") :])
            rank_func = f"NTILE({ntile_n})"
        else:
            mapped = _DUCKDB_RANK_FUNCS.get(rank_type)
            if mapped is None:
                raise ValueError(f"Unsupported rank type for DuckDB: {rank_type}")
            rank_func = mapped

        # PyArrow parity: the reference computes ranks via Python index mapping
        # and returns results in original row order. DuckDB window functions with
        # ORDER BY reorder result rows; tag positions with ROW_NUMBER() and
        # restore the original input row order afterwards.
        rel = data.with_row_number(_RN_COL)
        rel = rel.window(
            rank_func,
            feature_name,
            partition_by=partition_by,
            order_by=order_spec,
        )
        rel = rel.order(quote_ident(_RN_COL))
        keep = ", ".join(quote_ident(c) for c in rel.columns if c != _RN_COL)
        return rel.project(keep)
