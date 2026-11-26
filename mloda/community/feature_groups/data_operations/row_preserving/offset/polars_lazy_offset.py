"""Polars Lazy implementation for offset feature groups."""

from __future__ import annotations

from typing import Any

import polars as pl
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.helper_columns import unique_helper_name
from mloda.community.feature_groups.data_operations.polars_helpers import NARROW_INTEGERS
from mloda.community.feature_groups.data_operations.row_preserving.offset.base import (
    OffsetFeatureGroup,
)


class PolarsLazyOffset(Implementation[OffsetFeatureGroup]):
    feature_group = OffsetFeatureGroup
    compute_framework = PolarsLazyDataFrame

    @classmethod
    def calculate(
        cls, group: type[OffsetFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.offset(data, features, compute_offset=cls._compute_offset)

    @classmethod
    def _compute_offset(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        offset_type: str,
    ) -> pl.LazyFrame:
        """PyArrow parity: offset requires sorting by order_by, which reorders rows.
        Tag rows with an index before sorting and restore input order afterward."""
        schema = data.collect_schema()
        orig_idx_col = unique_helper_name("__mloda_orig_idx", set(schema.names()) | {feature_name})
        numeric_source = pl.col(source_col)
        if schema[source_col] in NARROW_INTEGERS:
            numeric_source = numeric_source.cast(pl.Int64)
        elif schema[source_col] == pl.UInt64:
            numeric_source = numeric_source.cast(pl.Int128)  # lossless, so the difference cannot wrap

        # Track original row order
        data = data.with_row_index(orig_idx_col)

        # Sort by partition_by + order_by (nulls last) for correct offset
        data = data.sort(partition_by + [order_by], nulls_last=True, maintain_order=True)

        if offset_type.startswith("lag_"):
            offset_n = int(offset_type[len("lag_") :])
            expr = pl.col(source_col).shift(offset_n).over(partition_by).alias(feature_name)
        elif offset_type.startswith("lead_"):
            offset_n = int(offset_type[len("lead_") :])
            expr = pl.col(source_col).shift(-offset_n).over(partition_by).alias(feature_name)
        elif offset_type.startswith("diff_"):
            offset_n = int(offset_type[len("diff_") :])
            diff = numeric_source - numeric_source.shift(offset_n)
            if schema[source_col] == pl.UInt64:
                diff = diff.cast(pl.Int64)  # strict: a result outside int64 raises like the reference
            expr = diff.over(partition_by).alias(feature_name)
        elif offset_type.startswith("pct_change_"):
            offset_n = int(offset_type[len("pct_change_") :])
            prev = numeric_source.shift(offset_n)
            change = numeric_source - prev
            if schema[source_col] == pl.UInt64:
                change, prev_f = change.cast(pl.Float64), prev.cast(pl.Float64)
            else:
                prev_f = prev
            expr = (
                pl.when(prev.is_not_null() & (prev != 0))
                .then(change / prev_f)
                .otherwise(pl.lit(None))
                .over(partition_by)
                .alias(feature_name)
            )
        elif offset_type == "first_value":
            expr = pl.col(source_col).drop_nulls().first().over(partition_by).alias(feature_name)
        elif offset_type == "last_value":
            expr = pl.col(source_col).drop_nulls().last().over(partition_by).alias(feature_name)
        else:
            raise ValueError(f"Unsupported offset type: {offset_type}")

        result = data.with_columns(expr)
        # Restore original row order and drop helper column
        return result.sort(orig_idx_col).drop(orig_idx_col)
