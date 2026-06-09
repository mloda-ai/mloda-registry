"""Shared pandas classmethods for the point- and scalar-arithmetic families.

The pandas point and scalar arithmetic backends carry byte-for-byte-identical
``compute_framework_rule`` / ``_input_columns_and_framework`` /
``_assert_source_column_is_numeric`` implementations. This mixin holds the one
copy; the concrete pandas backends inherit it (first in their MRO) and supply
only ``_compute_arithmetic``. Keeping one mixin per module preserves
optional-dependency isolation: this module imports only pandas.
"""

from __future__ import annotations

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.arithmetic.base import ArithmeticFeatureGroupBase
from mloda.community.feature_groups.data_operations.arithmetic.pandas_numeric_source import (
    pandas_non_numeric_descriptor,
)


class PandasArithmeticMixin(ArithmeticFeatureGroupBase):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}

    @classmethod
    def _input_columns_and_framework(cls, data: pd.DataFrame) -> tuple[list[str], str]:
        return list(data.columns), "Pandas"

    @classmethod
    def _assert_source_column_is_numeric(cls, data: pd.DataFrame, source_col: str) -> None:
        descriptor = pandas_non_numeric_descriptor(data[source_col])
        if descriptor is not None:
            cls._raise_non_numeric_source(source_col, descriptor)
