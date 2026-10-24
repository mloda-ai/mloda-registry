"""Shared pandas classmethods for the point- and scalar-arithmetic families.

``PandasArithmeticMixin`` is a plain class supplying the framework's
``_non_numeric_descriptor``, which the point and scalar Implementations pass to
the group's ``apply_arithmetic``. One mixin per module keeps optional-dependency
isolation: this module imports only pandas.
"""

from __future__ import annotations

import pandas as pd

from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.pandas_numeric_source import (
    pandas_non_numeric_descriptor,
)


class PandasArithmeticMixin:
    @classmethod
    def _non_numeric_descriptor(cls, data: pd.DataFrame, source_col: str) -> object | None:
        return pandas_non_numeric_descriptor(data[source_col])
