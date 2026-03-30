"""PyArrow implementation for string operation feature groups."""

from __future__ import annotations

from typing import Set, Type, Union

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.string.base import (
    StringFeatureGroup,
)

_PYARROW_STRING_FUNCS: dict[str, str] = {
    "upper": "utf8_upper",
    "lower": "utf8_lower",
    "trim": "utf8_trim_whitespace",
    "length": "utf8_length",
    "reverse": "utf8_reverse",
}


class PyArrowStringOps(StringFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}

    @classmethod
    def _compute_string(
        cls,
        table: pa.Table,
        feature_name: str,
        source_col: str,
        op: str,
    ) -> pa.Table:
        func_name = _PYARROW_STRING_FUNCS.get(op)
        if func_name is None:
            raise ValueError(f"Unsupported string operation: {op}")

        col = table.column(source_col)
        new_col = getattr(pc, func_name)(col)
        return table.append_column(feature_name, new_col)
