"""Polars Lazy implementation for string operation feature groups."""

from __future__ import annotations

from typing import Any

import polars as pl
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.string.base import (
    StringFeatureGroup,
)


class PolarsLazyStringOps(Implementation[StringFeatureGroup]):
    feature_group = StringFeatureGroup
    compute_framework = PolarsLazyDataFrame

    @classmethod
    def calculate(
        cls, group: type[StringFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.transform_string(data, features, compute_string=cls._compute_string)

    @classmethod
    def _compute_string(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
        op: str,
    ) -> pl.LazyFrame:
        col = pl.col(source_col)

        if op == "upper":
            expr = col.str.to_uppercase()
        elif op == "lower":
            expr = col.str.to_lowercase()
        elif op == "trim":
            expr = col.str.strip_chars()
        elif op == "length":
            expr = col.str.len_chars()
        elif op == "reverse":
            expr = col.str.reverse()
        else:
            raise ValueError(f"Unsupported string operation: {op}")

        return data.with_columns(expr.alias(feature_name))
