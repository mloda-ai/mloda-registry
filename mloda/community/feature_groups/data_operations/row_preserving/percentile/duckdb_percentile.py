"""DuckDB implementation for percentile feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.duckdb_helpers import (
    column_types,
    nan_to_null_sql,
    preserve_row_order,
)
from mloda.community.feature_groups.data_operations.mask_utils import build_sql_case_when
from mloda.community.feature_groups.data_operations.row_preserving.percentile.base import (
    PercentileFeatureGroup,
)


class DuckdbPercentile(Implementation[PercentileFeatureGroup]):
    feature_group = PercentileFeatureGroup
    compute_framework = DuckDBFramework

    @classmethod
    def calculate(
        cls, group: type[PercentileFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.percentile(data, features, compute_percentile=cls._compute_percentile)

    @classmethod
    def _compute_percentile(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        percentile: float,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> DuckdbRelation:
        quoted_source = quote_ident(source_col)

        source_sql = quoted_source
        if mask_spec is not None:
            source_sql = build_sql_case_when(DuckDBFramework.mask_engine(), data, mask_spec, quoted_source)

        # Safety: identifiers are quote_ident()-quoted. The percentile value is a
        # Python float validated to [0.0, 1.0] by the base class, so it cannot
        # produce SQL injection via float.__format__.
        agg_source = nan_to_null_sql(source_sql, column_types(data).get(source_col, ""))
        return preserve_row_order(
            data,
            feature_name,
            lambda rel, rn: rel.window(
                f"QUANTILE_CONT({agg_source}, {percentile})", feature_name, partition_by=partition_by
            ),
        )
