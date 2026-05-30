"""DuckDB implementation for binning feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.reserved_columns import assert_no_reserved_columns
from mloda.community.feature_groups.data_operations.row_preserving.binning.base import (
    BinningFeatureGroup,
)

# Helper column names for materializing window results (covered by the __mloda_ reserved guard).
_BIN_MIN = "__mloda_bin_min__"
_BIN_MAX = "__mloda_bin_max__"
_RN_COL = "__mloda_rn__"
_QBIN_PART = "__mloda_qbin_part__"
_QBIN_NTILE = "__mloda_qbin_ntile__"


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
        assert_no_reserved_columns(data.columns, framework="DuckDB", operation="binning")

        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)

        if op == "bin":
            # The whole-input MIN/MAX window aggregates are constant across every
            # row, so materializing them into helper columns and referencing those
            # is result-identical to inlining the window expressions in the CASE.
            not_nan = f"NOT isnan({quoted_source})"
            rel: DuckdbRelation = data.window(f"MIN({quoted_source}) FILTER (WHERE {not_nan})", _BIN_MIN)
            rel = rel.window(f"MAX({quoted_source}) FILTER (WHERE {not_nan})", _BIN_MAX)
            qmin = quote_ident(_BIN_MIN)
            qmax = quote_ident(_BIN_MAX)
            expr = (
                f"CASE WHEN {quoted_source} IS NULL OR isnan({quoted_source}) THEN NULL "
                f"WHEN {qmax} = {qmin} THEN 0 "
                f"ELSE LEAST(CAST(FLOOR("
                f"({quoted_source} - {qmin}) / "
                f"NULLIF(({qmax} - {qmin}) / {n_bins}.0, 0)"
                f") AS INTEGER), {n_bins - 1}) END"
            )
            rel = rel.project(f"*, {expr} AS {quoted_feature}")
            keep = ", ".join(quote_ident(c) for c in rel.columns if c not in (_BIN_MIN, _BIN_MAX))
            return rel.project(keep)

        if op == "qbin":
            # PyArrow parity: PyArrow _quantile_bin() assigns bins via index
            # mapping and naturally preserves row order. DuckDB NTILE()
            # reorders rows via ORDER BY; tag positions with a row-number
            # column and restore via .order() to match PyArrow output.
            # The NTILE partition key is a SQL CASE expression, so it must be
            # projected into a helper column before being used as partition_by.
            qbin_rel: DuckdbRelation = data.with_row_number(_RN_COL)
            part_case = f"CASE WHEN {quoted_source} IS NOT NULL AND NOT isnan({quoted_source}) THEN 1 END"
            qbin_rel = qbin_rel.project(f"*, {part_case} AS {quote_ident(_QBIN_PART)}")
            qbin_rel = qbin_rel.window(
                f"NTILE({n_bins})", _QBIN_NTILE, partition_by=[_QBIN_PART], order_by=[source_col]
            )
            qntile = quote_ident(_QBIN_NTILE)
            expr = (
                f"CASE WHEN {quoted_source} IS NULL OR isnan({quoted_source}) THEN NULL "
                f"ELSE LEAST({qntile} - 1, {n_bins - 1}) END"
            )
            qbin_rel = qbin_rel.project(f"*, {expr} AS {quoted_feature}")
            qbin_rel = qbin_rel.order(quote_ident(_RN_COL))
            keep = ", ".join(quote_ident(c) for c in qbin_rel.columns if c not in (_RN_COL, _QBIN_PART, _QBIN_NTILE))
            return qbin_rel.project(keep)

        raise ValueError(f"Unsupported binning operation for DuckDB: {op}")
