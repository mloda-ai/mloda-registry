"""Polars Lazy implementation for string operation feature groups."""

from __future__ import annotations

from typing import Any

import polars as pl
from mloda.provider import Cause, ComputeFramework, Divergence, DivergenceKind, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.string.base import (
    TRIM_WHITESPACE,
    StringFeatureGroup,
)

_TRIM_CHARS = "".join(map(chr, TRIM_WHITESPACE))


class PolarsLazyStringOps(Implementation[StringFeatureGroup]):
    feature_group = StringFeatureGroup
    compute_framework = PolarsLazyDataFrame
    divergences = (
        Divergence(
            kind=DivergenceKind.VALUE,
            cause=Cause.BACKEND,
            subtypes=frozenset({"upper", "lower"}),
            reason="full case mapping: ß upper is SS, İ lower keeps a combining dot, a final sigma lowers to ς",
            docs="https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/data-operation-patterns/known-divergences.md#polars-upperlower-use-full-case-mapping",
        ),
    )

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
            expr = col.str.strip_chars(_TRIM_CHARS)
        elif op == "length":
            expr = col.str.len_chars().cast(pl.Int64)
        elif op == "reverse":
            expr = col.str.split("").list.reverse().list.join("")
        else:
            raise ValueError(f"Unsupported string operation: {op}")

        return data.with_columns(expr.alias(feature_name))
