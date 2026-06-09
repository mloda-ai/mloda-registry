"""Shared polars classmethods for the point- and scalar-arithmetic families.

The polars point and scalar arithmetic backends carry byte-for-byte-identical
``compute_framework_rule`` / ``_input_columns_and_framework`` /
``_assert_source_column_is_numeric`` implementations. This mixin holds the one
copy; the concrete polars backends inherit it (first in their MRO) and supply
only ``_compute_arithmetic``. Keeping one mixin per module preserves
optional-dependency isolation: this module imports only polars.
"""

from __future__ import annotations

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.arithmetic.base import ArithmeticFeatureGroupBase
from mloda.community.feature_groups.data_operations.arithmetic.polars_numeric_source import (
    polars_non_numeric_descriptor,
)


class PolarsArithmeticMixin(ArithmeticFeatureGroupBase):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PolarsLazyDataFrame}

    @classmethod
    def _input_columns_and_framework(cls, data: pl.LazyFrame) -> tuple[list[str], str]:
        return list(data.collect_schema().names()), "Polars"

    @classmethod
    def _assert_source_column_is_numeric(cls, data: pl.LazyFrame, source_col: str) -> None:
        descriptor = polars_non_numeric_descriptor(data.collect_schema()[source_col])
        if descriptor is not None:
            cls._raise_non_numeric_source(source_col, descriptor)
