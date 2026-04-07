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

        # qbin requires NTILE() with ORDER BY, which reorders rows in DuckDB's
        # relational API. Blocked until DuckdbRelation exposes a public order() method.
        # See: https://github.com/mloda-ai/mloda/issues/251
        raise ValueError(f"Unsupported binning operation for DuckDB: {op}")
