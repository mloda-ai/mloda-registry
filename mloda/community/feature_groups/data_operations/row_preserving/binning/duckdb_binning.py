"""DuckDB implementation for binning feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

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
        elif op == "qbin":
            expr = (
                f"CASE WHEN {quoted_source} IS NULL THEN NULL "
                f"ELSE LEAST(NTILE({n_bins}) OVER (ORDER BY {quoted_source}) - 1, {n_bins - 1}) END"
            )
        else:
            raise ValueError(f"Unsupported binning operation for DuckDB: {op}")

        raw_sql = f"*, {expr} AS {quoted_feature}"
        result: DuckdbRelation = data.select(_raw_sql=raw_sql)
        return result
