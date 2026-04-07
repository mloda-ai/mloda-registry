"""PyArrow implementation for datetime extraction feature groups.

Uses ``pyarrow.compute`` vectorized functions for all operations,
avoiding row-by-row Python loops for performance on large datasets.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.datetime.base import (
    DateTimeFeatureGroup,
)


class PyArrowDateTimeExtraction(DateTimeFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
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

        if op == "year":
            result = pc.year(col)
        elif op == "month":
            result = pc.month(col)
        elif op == "day":
            result = pc.day(col)
        elif op == "hour":
            result = pc.hour(col)
        elif op == "minute":
            result = pc.minute(col)
        elif op == "second":
            result = pc.second(col)
        elif op == "dayofweek":
            result = pc.day_of_week(col)
        elif op == "is_weekend":
            dow = pc.day_of_week(col)
            result = pc.if_else(pc.greater_equal(dow, 5), 1, 0)
        elif op == "quarter":
            result = pc.quarter(col)
        else:
            raise ValueError(f"Unsupported datetime operation: {op}")

        new_col = result.cast(pa.int64())
        return table.append_column(feature_name, new_col)
