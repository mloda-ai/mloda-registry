"""DuckDB implementation for binning feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.row_preserving.binning.base import (
    BinningFeatureGroup,
)


class DuckdbBinning(BinningFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
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
            not_nan = f"NOT isnan({quoted_source})"
            min_expr = f"MIN({quoted_source}) FILTER (WHERE {not_nan}) OVER ()"
            max_expr = f"MAX({quoted_source}) FILTER (WHERE {not_nan}) OVER ()"
            expr = (
                f"CASE WHEN {quoted_source} IS NULL OR isnan({quoted_source}) THEN NULL "
                f"WHEN {max_expr} = {min_expr} THEN 0 "
                f"ELSE LEAST(CAST(FLOOR("
                f"({quoted_source} - {min_expr}) / "
                f"NULLIF(({max_expr} - {min_expr}) / {n_bins}.0, 0)"
                f") AS INTEGER), {n_bins - 1}) END"
            )
            raw_sql = f"*, {expr} AS {quoted_feature}"
            result: DuckdbRelation = data.select(_raw_sql=raw_sql)
            return result

        if op == "qbin":
            # PyArrow parity: PyArrow _quantile_bin() assigns bins via index
            # mapping and naturally preserves row order. DuckDB NTILE()
            # reorders rows via ORDER BY; tag positions with ROW_NUMBER()
            # and restore via .order() to match PyArrow output.
            qrn = quote_ident("__mloda_rn__")
            expr = (
                f"CASE WHEN {quoted_source} IS NULL OR isnan({quoted_source}) THEN NULL "
                f"ELSE LEAST(NTILE({n_bins}) OVER ("
                f"PARTITION BY CASE WHEN {quoted_source} IS NOT NULL "
                f"AND NOT isnan({quoted_source}) THEN 1 END "
                f"ORDER BY {quoted_source}) - 1, {n_bins - 1}) END"
            )
            with_rn = data.select(_raw_sql=f"*, ROW_NUMBER() OVER () AS {qrn}")
            with_qbin = with_rn.select(_raw_sql=f"*, {expr} AS {quoted_feature}")
            sorted_rel = with_qbin.order(qrn)
            keep = ", ".join(quote_ident(c) for c in sorted_rel.columns if c != "__mloda_rn__")
            qbin_result: DuckdbRelation = sorted_rel.select(_raw_sql=keep)
            return qbin_result

        raise ValueError(f"Unsupported binning operation for DuckDB: {op}")
