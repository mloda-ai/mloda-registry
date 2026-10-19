"""Polars lazy implementation of key-lookup."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl
from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.lookup.base import (
    LookupFeatureGroup,
    assert_key_family,
    build_unmatched_error,
)


def _source_family(dtype: pl.DataType) -> str:
    if dtype == pl.String:
        return "str"
    # UInt64 cannot be compared at Int64 width.
    if dtype.is_integer() and dtype != pl.UInt64:
        return "int"
    return "other"


class PolarsLazyLookup(LookupFeatureGroup):
    """Polars lazy backend: ``replace_strict`` stays lazy; ``raise`` collects only the unmatched rows."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PolarsLazyDataFrame}

    @classmethod
    def _compute_lookup(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
        mapping: Mapping[Any, Any],
        policy: str,
        reasons: Mapping[Any, str],
    ) -> pl.LazyFrame:
        source_dtype = data.collect_schema()[source_col]
        assert_key_family(feature_name, source_col, mapping, _source_family(source_dtype), str(source_dtype))

        values = pl.Series(list(mapping.values()))
        source = pl.col(source_col)
        # Compare ints at Int64 width so keys outside the source width never match.
        probe = source.cast(pl.Int64) if source_dtype.is_integer() else source
        keys = pl.Series(list(mapping.keys()), dtype=pl.Int64 if source_dtype.is_integer() else source_dtype)
        mapped = probe.replace_strict(keys, values, default=None, return_dtype=values.dtype)
        # Mapped values are never null, so a null result on a non-null source is an unmatched key.
        unmatched = source.is_not_null() & mapped.is_null()

        if policy == "raise":
            keys_found = data.filter(unmatched).select(source).collect().to_series()
            if keys_found.len():
                raise build_unmatched_error(feature_name, keys_found.len(), keys_found.unique().to_list(), reasons)
        elif policy == "keep":
            mapped = pl.when(unmatched).then(source.cast(values.dtype)).otherwise(mapped)

        return data.with_columns(mapped.alias(feature_name))
