"""DuckDB implementation for rank feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

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
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
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
        else:
            rank_func = _DUCKDB_RANK_FUNCS.get(rank_type)
            if rank_func is None:
                raise ValueError(f"Unsupported rank type for DuckDB: {rank_type}")
            rank_expr = rank_func

        # Use query() with ROW_NUMBER() to preserve original row order.
        # The _RN_COL column tracks original position, then we
        # sort by it and drop it to return rows in the original order.
        qrn = quote_ident(_RN_COL)
        sql = (
            f"SELECT *, "  # nosec B608
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
