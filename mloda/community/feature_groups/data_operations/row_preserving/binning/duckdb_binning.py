"""DuckDB implementation for binning feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name, quote_ident

from mloda.community.feature_groups.data_operations.duckdb_helpers import column_types, with_input_position
from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.duckdb_numeric_source import (
    duckdb_non_numeric_descriptor,
)
from mloda.community.feature_groups.data_operations.row_preserving.binning.base import (
    BinningFeatureGroup,
)

# Integer types whose values can exceed 2**53.
_WIDE_INTEGER_TYPES = ("BIGINT", "UBIGINT", "HUGEINT", "UHUGEINT")

_DECIMAL_PREFIXES = ("DECIMAL", "NUMERIC", "BIGNUM")


def _non_numeric_descriptor(data: Any, source_col: str) -> str | None:
    """Binning also rejects DECIMAL/NUMERIC/BIGNUM, which arithmetic accepts."""
    described = duckdb_non_numeric_descriptor(data, source_col)
    if described is not None:
        return described
    dtype_str = column_types(data).get(source_col)
    if dtype_str is not None and dtype_str.upper().startswith(_DECIMAL_PREFIXES):
        return dtype_str
    return None


class DuckdbBinning(Implementation[BinningFeatureGroup]):
    feature_group = BinningFeatureGroup
    compute_framework = DuckDBFramework

    @classmethod
    def calculate(
        cls, group: type[BinningFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.assign_bins(
            data,
            features,
            compute_binning=cls._compute_binning,
            non_numeric=_non_numeric_descriptor,
            integer_bounds=cls._integer_bounds,
        )

    @classmethod
    def _integer_bounds(cls, data: Any, source_col: str) -> tuple[int, int] | None:
        if column_types(data).get(source_col, "").upper() not in _WIDE_INTEGER_TYPES:
            return None
        q = quote_ident(source_col)
        row = data.aggregate(f"MIN({q}), MAX({q})").to_arrow_table().to_pylist()[0]
        lo, hi = list(row.values())
        return None if lo is None else (int(lo), int(hi))

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
            dsource = f"CAST({quoted_source} AS DOUBLE)"
            rel: DuckdbRelation = data.window(f"MIN({dsource}) FILTER (WHERE {not_nan})", bin_min)
            rel = rel.window(f"MAX({dsource}) FILTER (WHERE {not_nan})", bin_max)
            qmin = quote_ident(bin_min)
            qmax = quote_ident(bin_max)
            expr = (
                f"CASE WHEN {quoted_source} IS NULL OR isnan({quoted_source}) THEN NULL "
                f"WHEN {qmax} = {qmin} THEN 0 "
                f"ELSE LEAST(CAST(FLOOR("
                f"({dsource} - {qmin}) / "
                f"NULLIF(({qmax} - {qmin}) / {n_bins}.0, 0)"
                f") AS BIGINT), {n_bins - 1}) END"
            )
            rel = rel.project(f"*, {expr} AS {quoted_feature}")
            keep = ", ".join(quote_ident(c) for c in rel.columns if c not in (bin_min, bin_max))
            return rel.project(keep)

        if op == "qbin":
            # bin = (row_number - 1) * n // count over present rows; the partition key is a CASE
            # expression, so it is projected into a helper column first. rn breaks ties by input position.
            return with_input_position(
                data, feature_name, lambda rel, rn: cls._qbin_window(rel, rn, feature_name, source_col, n_bins)
            )

        raise ValueError(f"Unsupported binning operation for DuckDB: {op}")

    @classmethod
    def _qbin_window(
        cls, rel: DuckdbRelation, rn: str, feature_name: str, source_col: str, n_bins: int
    ) -> DuckdbRelation:
        quoted_source = quote_ident(source_col)
        taken = set(rel.columns) | {feature_name}
        part = pick_helper_column_name(taken=taken)
        rank = pick_helper_column_name(taken=taken | {part})
        cnt = pick_helper_column_name(taken=taken | {part, rank})
        present = f"{quoted_source} IS NOT NULL AND NOT isnan({quoted_source})"
        rel = rel.project(f"*, CASE WHEN {present} THEN 1 END AS {quote_ident(part)}")
        rel = rel.window("ROW_NUMBER()", rank, partition_by=[part], order_by=[source_col, rn])
        rel = rel.window(f"COUNT({quoted_source}) FILTER (WHERE NOT isnan({quoted_source}))", cnt)
        expr = f"CASE WHEN {present} THEN ({quote_ident(rank)} - 1) * {n_bins} // {quote_ident(cnt)} END"
        keep = ", ".join(quote_ident(c) for c in rel.columns if c not in (part, rank, cnt))
        return rel.project(f"{keep}, {expr} AS {quote_ident(feature_name)}")
