"""Shared polars classmethods for the point- and scalar-arithmetic families.

``PolarsArithmeticMixin`` is a plain class supplying the framework's
``_non_numeric_descriptor``, which the point and scalar Implementations pass to
the group's ``apply_arithmetic``. One mixin per module keeps optional-dependency
isolation: this module imports only polars.
"""

from __future__ import annotations

import polars as pl

from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.polars_numeric_source import (
    polars_non_numeric_descriptor,
)


class PolarsArithmeticMixin:
    @classmethod
    def _non_numeric_descriptor(cls, data: pl.LazyFrame, source_col: str) -> object | None:
        return polars_non_numeric_descriptor(data.collect_schema()[source_col])
