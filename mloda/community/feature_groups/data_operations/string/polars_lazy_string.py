"""Polars Lazy implementation for string operation feature groups."""

from __future__ import annotations

from typing import Set, Type, Union

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.string.base import (
    StringFeatureGroup,
)


class PolarsLazyStringOps(StringFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PolarsLazyDataFrame}

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
