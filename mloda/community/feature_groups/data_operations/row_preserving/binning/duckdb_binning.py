"""DuckDB implementation for binning feature groups."""

from __future__ import annotations

from typing import Any, List, Set, Type, Union

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.row_preserving.binning.base import (
    BinningFeatureGroup,
)


class DuckdbBinning(BinningFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {DuckDBFramework}

    @classmethod
    def _compute_binning(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        op: str,
        n_bins: int,
    ) -> DuckdbRelation:
        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)

        if op == "bin":
            expr = (
                f"CASE WHEN {quoted_source} IS NULL THEN NULL "
                f"WHEN MAX({quoted_source}) OVER () = MIN({quoted_source}) OVER () THEN 0 "
                f"ELSE LEAST(CAST(FLOOR("
                f"({quoted_source} - MIN({quoted_source}) OVER ()) / "
                f"NULLIF((MAX({quoted_source}) OVER () - MIN({quoted_source}) OVER ()) / {n_bins}.0, 0)"
                f") AS INTEGER), {n_bins - 1}) END"
            )
            raw_sql = f"*, {expr} AS {quoted_feature}"
            result: DuckdbRelation = data.select(_raw_sql=raw_sql)
            return result

        if op == "qbin":
            return cls._rank_based_qbin(data, feature_name, source_col, n_bins)

        raise ValueError(f"Unsupported binning operation for DuckDB: {op}")

    @classmethod
    def _rank_based_qbin(
        cls,
        data: DuckdbRelation,
        feature_name: str,
        source_col: str,
        n_bins: int,
    ) -> DuckdbRelation:
        """Rank-based quantile binning matching NTILE semantics.

        Computes qbin in Python to avoid DuckDB window-ORDER-BY row reordering,
        then appends the result column back to the relation.
        """
        arrow_table = data.to_arrow_table()
        values = arrow_table.column(source_col).to_pylist()

        indexed: List[tuple[Any, int]] = [(v, i) for i, v in enumerate(values) if v is not None]
        indexed.sort(key=lambda pair: pair[0])
        n = len(indexed)

        result_values: List[Any] = [None] * len(values)
        if n > 0:
            for rank, (_, orig_idx) in enumerate(indexed):
                result_values[orig_idx] = rank * n_bins // n

        return data.append_column(feature_name, result_values)
