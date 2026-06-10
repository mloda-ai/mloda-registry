"""Shared pyarrow classmethods for the point- and scalar-arithmetic families.

``PyArrowArithmeticMixin`` is a plain runtime class typed against
``ArithmeticFeatureGroupBase`` for mypy only, so it stays out of FeatureGroup
plugin discovery. It supplies ``compute_framework_rule``,
``_input_columns_and_framework``, and the ``_non_numeric_descriptor`` hook
consumed by the base's ``_assert_source_column_is_numeric`` template; the
structural guards live in ``tests/test_numeric_source.py``. Keeping one mixin
per module preserves optional-dependency isolation: this module imports only
pyarrow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.pyarrow_numeric_source import (
    pyarrow_non_numeric_descriptor,
)

if TYPE_CHECKING:
    from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.base import ArithmeticFeatureGroupBase

    _ArithmeticMixinBase = ArithmeticFeatureGroupBase
else:
    _ArithmeticMixinBase = object


class PyArrowArithmeticMixin(_ArithmeticMixinBase):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PyArrowTable}

    @classmethod
    def _input_columns_and_framework(cls, data: pa.Table) -> tuple[list[str], str]:
        return list(data.column_names), "PyArrow"

    @classmethod
    def _non_numeric_descriptor(cls, data: pa.Table, source_col: str) -> object | None:
        return pyarrow_non_numeric_descriptor(data.column(source_col).type)
