"""Polars Lazy implementation for datetime extraction feature groups."""

from __future__ import annotations

from typing import Any

import polars as pl
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.datetime.base import (
    DateTimeFeatureGroup,
)


class PolarsLazyDateTimeExtraction(Implementation[DateTimeFeatureGroup]):
    feature_group = DateTimeFeatureGroup
    compute_framework = PolarsLazyDataFrame

    @classmethod
    def calculate(
        cls, group: type[DateTimeFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.extract_datetime(data, features, compute_datetime=cls._compute_datetime)

    @classmethod
    def _compute_datetime(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
        op: str,
    ) -> pl.LazyFrame:
        col = pl.col(source_col)

        if op == "year":
            expr = col.dt.year()
        elif op == "month":
            expr = col.dt.month()
        elif op == "day":
            expr = col.dt.day()
        elif op == "hour":
            expr = col.dt.hour()
        elif op == "minute":
            expr = col.dt.minute()
        elif op == "second":
            expr = col.dt.second()
        elif op == "dayofweek":
            expr = col.dt.weekday() - 1
        elif op == "is_weekend":
            # Null propagation: when col is null, weekday() returns null,
            # >= 6 returns null (boolean null), and .cast(Int64) produces
            # a Polars null that propagates to None in Python. This relies
            # on Polars' implicit null propagation through comparison and cast.
            expr = (col.dt.weekday() >= 6).cast(pl.Int64)
        elif op == "quarter":
            expr = col.dt.quarter()
        else:
            raise ValueError(f"Unsupported datetime operation: {op}")

        return data.with_columns(expr.alias(feature_name))
