"""Shared pyarrow classmethods for the point- and scalar-arithmetic families.

``PyArrowArithmeticMixin`` is a plain class supplying the framework's
``_non_numeric_descriptor``, which the point and scalar Implementations pass to
the group's ``apply_arithmetic``. One mixin per module keeps optional-dependency
isolation: this module imports only pyarrow.
"""

from __future__ import annotations

import pyarrow as pa

from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.pyarrow_numeric_source import (
    pyarrow_non_numeric_descriptor,
)


class PyArrowArithmeticMixin:
    @classmethod
    def _non_numeric_descriptor(cls, data: pa.Table, source_col: str) -> object | None:
        return pyarrow_non_numeric_descriptor(data.column(source_col).type)
