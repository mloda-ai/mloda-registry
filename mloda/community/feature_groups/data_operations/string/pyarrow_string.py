"""PyArrow implementation for string operation feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

import pyarrow as pa

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.string.base import (
    StringFeatureGroup,
)


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
        col = table.column(source_col)
        values = col.to_pylist()

        result_values: list[Any] = []
        for val in values:
            if val is None:
                result_values.append(None)
                continue

            if op == "upper":
                result_values.append(val.upper())
            elif op == "lower":
                result_values.append(val.lower())
            elif op == "trim":
                result_values.append(val.strip())
            elif op == "length":
                result_values.append(len(val))
            elif op == "reverse":
                result_values.append(val[::-1])
            else:
                raise ValueError(f"Unsupported string operation: {op}")

        if op == "length":
            new_col = pa.array(result_values, type=pa.int64())
        else:
            new_col = pa.array(result_values, type=pa.string())
        return table.append_column(feature_name, new_col)
