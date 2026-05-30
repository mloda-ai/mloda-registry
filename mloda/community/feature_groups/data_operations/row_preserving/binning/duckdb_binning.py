"""DuckDB implementation for binning feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name, quote_ident

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
            # The whole-input MIN/MAX window aggregates are constant across every
            # row, so materializing them into helper columns and referencing those
            # is result-identical to inlining the window expressions in the CASE.
            bin_min = pick_helper_column_name(taken=set(data.columns) | {feature_name})
            bin_max = pick_helper_column_name(taken=set(data.columns) | {feature_name, bin_min})
            not_nan = f"NOT isnan({quoted_source})"
            rel: DuckdbRelation = data.window(f"MIN({quoted_source}) FILTER (WHERE {not_nan})", bin_min)
            rel = rel.window(f"MAX({quoted_source}) FILTER (WHERE {not_nan})", bin_max)
            qmin = quote_ident(bin_min)
            qmax = quote_ident(bin_max)
            expr = (
                f"CASE WHEN {quoted_source} IS NULL OR isnan({quoted_source}) THEN NULL "
                f"WHEN {qmax} = {qmin} THEN 0 "
                f"ELSE LEAST(CAST(FLOOR("
                f"({quoted_source} - {qmin}) / "
                f"NULLIF(({qmax} - {qmin}) / {n_bins}.0, 0)"
                f") AS INTEGER), {n_bins - 1}) END"
            )
            rel = rel.project(f"*, {expr} AS {quoted_feature}")
            keep = ", ".join(quote_ident(c) for c in rel.columns if c not in (bin_min, bin_max))
            return rel.project(keep)

        if op == "qbin":
            # PyArrow parity: PyArrow _quantile_bin() assigns bins via index
            # mapping and naturally preserves row order. DuckDB NTILE()
            # reorders rows via ORDER BY; tag positions with a row-number
            # column and restore via .order() to match PyArrow output.
            # The NTILE partition key is a SQL CASE expression, so it must be
            # projected into a helper column before being used as partition_by.
            rn = pick_helper_column_name(taken=set(data.columns) | {feature_name})
            qbin_part = pick_helper_column_name(taken=set(data.columns) | {feature_name, rn})
            qbin_ntile = pick_helper_column_name(taken=set(data.columns) | {feature_name, rn, qbin_part})
            qbin_rel: DuckdbRelation = data.with_row_number(rn)
            part_case = f"CASE WHEN {quoted_source} IS NOT NULL AND NOT isnan({quoted_source}) THEN 1 END"
            qbin_rel = qbin_rel.project(f"*, {part_case} AS {quote_ident(qbin_part)}")
            qbin_rel = qbin_rel.window(f"NTILE({n_bins})", qbin_ntile, partition_by=[qbin_part], order_by=[source_col])
            qntile = quote_ident(qbin_ntile)
            expr = (
                f"CASE WHEN {quoted_source} IS NULL OR isnan({quoted_source}) THEN NULL "
                f"ELSE LEAST({qntile} - 1, {n_bins - 1}) END"
            )
            qbin_rel = qbin_rel.project(f"*, {expr} AS {quoted_feature}")
            qbin_rel = qbin_rel.order(quote_ident(rn))
            keep = ", ".join(quote_ident(c) for c in qbin_rel.columns if c not in (rn, qbin_part, qbin_ntile))
            return qbin_rel.project(keep)

        raise ValueError(f"Unsupported binning operation for DuckDB: {op}")
