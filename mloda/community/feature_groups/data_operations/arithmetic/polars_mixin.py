"""Shared polars classmethods for the point- and scalar-arithmetic families.

``PolarsArithmeticMixin`` is a plain runtime class typed against
``ArithmeticFeatureGroupBase`` for mypy only, so it stays out of FeatureGroup
plugin discovery. It supplies ``compute_framework_rule``,
``_input_columns_and_framework``, and the ``_non_numeric_descriptor`` hook
consumed by the base's ``_assert_source_column_is_numeric`` template; the
structural guards live in ``tests/test_numeric_source.py``. Keeping one mixin
per module preserves optional-dependency isolation: this module imports only
polars.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.arithmetic.polars_numeric_source import (
    polars_non_numeric_descriptor,
)

if TYPE_CHECKING:
    from mloda.community.feature_groups.data_operations.arithmetic.base import ArithmeticFeatureGroupBase

    _ArithmeticMixinBase = ArithmeticFeatureGroupBase
else:
    _ArithmeticMixinBase = object


class PolarsArithmeticMixin(_ArithmeticMixinBase):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PolarsLazyDataFrame}

    @classmethod
    def _input_columns_and_framework(cls, data: pl.LazyFrame) -> tuple[list[str], str]:
        return list(data.collect_schema().names()), "Polars"

    @classmethod
    def _non_numeric_descriptor(cls, data: pl.LazyFrame, source_col: str) -> object | None:
        return polars_non_numeric_descriptor(data.collect_schema()[source_col])
