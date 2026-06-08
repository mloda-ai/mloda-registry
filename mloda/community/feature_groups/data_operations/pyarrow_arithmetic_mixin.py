"""Shared pyarrow classmethods for the point- and scalar-arithmetic families.

The pyarrow point and scalar arithmetic backends carry byte-for-byte-identical
``compute_framework_rule`` / ``_input_columns_and_framework`` /
``_assert_source_column_is_numeric`` implementations. This mixin holds the one
copy; the concrete pyarrow backends inherit it (first in their MRO) and supply
only ``_compute_arithmetic``. Keeping one mixin per module preserves
optional-dependency isolation: this module imports only pyarrow.
"""

from __future__ import annotations

import pyarrow as pa

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.arithmetic_base import ArithmeticFeatureGroupBase
from mloda.community.feature_groups.data_operations.pyarrow_numeric_source import pyarrow_non_numeric_descriptor


class PyArrowArithmeticMixin(ArithmeticFeatureGroupBase):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PyArrowTable}

    @classmethod
    def _input_columns_and_framework(cls, data: pa.Table) -> tuple[list[str], str]:
        return list(data.column_names), "PyArrow"

    @classmethod
    def _assert_source_column_is_numeric(cls, data: pa.Table, source_col: str) -> None:
        descriptor = pyarrow_non_numeric_descriptor(data.column(source_col).type)
        if descriptor is not None:
            cls._raise_non_numeric_source(source_col, descriptor)
