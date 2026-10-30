"""DuckDB implementation for rank feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name, quote_ident
from mloda_plugins.compute_framework.base_implementations.sql.sql_window import OrderBy

from mloda.community.feature_groups.data_operations.duckdb_helpers import preserve_row_order
from mloda.community.feature_groups.data_operations.row_preserving.rank.base import (
    RankFeatureGroup,
)

_DUCKDB_RANK_FUNCS: dict[str, str] = {
    "row_number": "ROW_NUMBER()",
    "rank": "RANK()",
    "dense_rank": "DENSE_RANK()",
    "percent_rank": "PERCENT_RANK()",
}


class DuckdbRank(Implementation[RankFeatureGroup]):
    feature_group = RankFeatureGroup
    compute_framework = DuckDBFramework

    @classmethod
    def calculate(
        cls, group: type[RankFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.rank(data, features, compute_rank=cls._compute_rank)

    @classmethod
    def _compute_rank(
        cls,
        data: Any,
        feature_name: str,
        partition_by: list[str],
        order_by: str,
        rank_type: str,
    ) -> DuckdbRelation:
        order_spec = [OrderBy(order_by, nulls="last")]

        if rank_type.startswith(("top_", "bottom_")):
            is_top = rank_type.startswith("top_")
            prefix = "top_" if is_top else "bottom_"
            n_val = int(rank_type[len(prefix) :])

            # The boolean comparison wraps a ROW_NUMBER() window, so compute the
            # row number into a helper column first, then project the comparison
            # (no OVER). DESC for top_, ASC for bottom_.
            def compute(rel: DuckdbRelation, rn: str) -> DuckdbRelation:
                rank_rn = pick_helper_column_name(taken=set(rel.columns) | {feature_name})
                qhelper = quote_ident(rank_rn)
                rel = rel.window(
                    "ROW_NUMBER()",
                    rank_rn,
                    partition_by=partition_by,
                    order_by=[OrderBy(order_by, descending=is_top, nulls="last")],
                )
                keep = ", ".join(quote_ident(c) for c in rel.columns if c != rank_rn)
                return rel.project(f"{keep}, ({qhelper} <= {n_val}) AS {quote_ident(feature_name)}")

            return preserve_row_order(data, feature_name, compute)

        if rank_type.startswith("ntile_"):
            ntile_n = int(rank_type[len("ntile_") :])
            rank_func = f"NTILE({ntile_n})"
        else:
            mapped = _DUCKDB_RANK_FUNCS.get(rank_type)
            if mapped is None:
                raise ValueError(f"Unsupported rank type for DuckDB: {rank_type}")
            rank_func = mapped

        # PyArrow parity: results come back in original row order.
        return preserve_row_order(
            data,
            feature_name,
            lambda rel, rn: rel.window(rank_func, feature_name, partition_by=partition_by, order_by=order_spec),
        )
