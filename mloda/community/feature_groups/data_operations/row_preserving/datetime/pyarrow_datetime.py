"""PyArrow implementation for datetime extraction feature groups.

Uses ``pyarrow.compute`` vectorized functions for all operations,
avoiding row-by-row Python loops for performance on large datasets.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.pyarrow_helpers import assert_temporal_timezone_supported
from mloda.community.feature_groups.data_operations.row_preserving.datetime.base import (
    DateTimeFeatureGroup,
)


class PyArrowDateTimeExtraction(Implementation[DateTimeFeatureGroup]):
    feature_group = DateTimeFeatureGroup
    compute_framework = PyArrowTable

    @classmethod
    def calculate(
        cls, group: type[DateTimeFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.extract_datetime(
            data, features, compute_datetime=cls._compute_datetime, is_date_source=cls._is_date_source
        )

    @staticmethod
    def _is_date_source(data: Any, source_col: str) -> bool:
        return bool(pa.types.is_date(data.column(source_col).type))

    @classmethod
    def _compute_datetime(
        cls,
        table: pa.Table,
        feature_name: str,
        source_col: str,
        op: str,
    ) -> pa.Table:
        col = table.column(source_col)
        assert_temporal_timezone_supported(col.type, source_col)

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
