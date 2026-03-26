"""PyArrow implementation for datetime extraction feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

import pyarrow as pa

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.datetime.base import (
    DateTimeFeatureGroup,
)


class PyArrowDateTimeExtraction(DateTimeFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}

    @classmethod
    def _compute_datetime(
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

            if op == "year":
                result_values.append(val.year)
            elif op == "month":
                result_values.append(val.month)
            elif op == "day":
                result_values.append(val.day)
            elif op == "hour":
                result_values.append(val.hour)
            elif op == "minute":
                result_values.append(val.minute)
            elif op == "second":
                result_values.append(val.second)
            elif op == "dayofweek":
                result_values.append(val.weekday())
            elif op == "is_weekend":
                result_values.append(1 if val.weekday() >= 5 else 0)
            elif op == "quarter":
                result_values.append((val.month - 1) // 3 + 1)
            else:
                raise ValueError(f"Unsupported datetime operation: {op}")

        new_col = pa.array(result_values, type=pa.int64())
        return table.append_column(feature_name, new_col)
