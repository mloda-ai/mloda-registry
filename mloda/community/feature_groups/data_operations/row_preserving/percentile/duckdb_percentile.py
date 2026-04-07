"""DuckDB implementation for percentile feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.row_preserving.percentile.base import (
    PercentileFeatureGroup,
)


class DuckdbPercentile(PercentileFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {DuckDBFramework}

    @classmethod
    def _compute_percentile(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        percentile: float,
    ) -> DuckdbRelation:
        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)
        partition_clause = ", ".join(quote_ident(col) for col in partition_by)

        # Safety: identifiers are quote_ident()-quoted. The percentile value is a
        # Python float validated to [0.0, 1.0] by the base class, so it cannot
        # produce SQL injection via float.__format__.
        raw_sql = (
            f"*, QUANTILE_CONT({quoted_source}, {percentile}) "
            f"OVER (PARTITION BY {partition_clause}) AS {quoted_feature}"
        )
        result: DuckdbRelation = data.select(_raw_sql=raw_sql)
        return result
