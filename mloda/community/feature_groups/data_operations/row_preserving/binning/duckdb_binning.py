"""DuckDB implementation for binning feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name, quote_ident

from mloda.community.feature_groups.data_operations.duckdb_helpers import preserve_row_order
from mloda.community.feature_groups.data_operations.row_preserving.binning.base import (
    BinningFeatureGroup,
)


class DuckdbBinning(Implementation[BinningFeatureGroup]):
    feature_group = BinningFeatureGroup
    compute_framework = DuckDBFramework

    @classmethod
    def calculate(
        cls, group: type[BinningFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.assign_bins(data, features, compute_binning=cls._compute_binning)

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
            # PyArrow parity: results come back in original row order.
            # The NTILE partition key is a SQL CASE expression, so it must be
            # projected into a helper column before being used as partition_by.
            def compute(rel: DuckdbRelation, rn: str) -> DuckdbRelation:
                qbin_part = pick_helper_column_name(taken=set(rel.columns) | {feature_name})
                qbin_ntile = pick_helper_column_name(taken=set(rel.columns) | {feature_name, qbin_part})
                part_case = f"CASE WHEN {quoted_source} IS NOT NULL AND NOT isnan({quoted_source}) THEN 1 END"
                rel = rel.project(f"*, {part_case} AS {quote_ident(qbin_part)}")
                rel = rel.window(f"NTILE({n_bins})", qbin_ntile, partition_by=[qbin_part], order_by=[source_col])
                qntile = quote_ident(qbin_ntile)
                expr = (
                    f"CASE WHEN {quoted_source} IS NULL OR isnan({quoted_source}) THEN NULL "
                    f"ELSE LEAST({qntile} - 1, {n_bins - 1}) END"
                )
                keep = ", ".join(quote_ident(c) for c in rel.columns if c not in (qbin_part, qbin_ntile))
                return rel.project(f"{keep}, {expr} AS {quoted_feature}")

            return preserve_row_order(data, feature_name, compute)

        raise ValueError(f"Unsupported binning operation for DuckDB: {op}")
