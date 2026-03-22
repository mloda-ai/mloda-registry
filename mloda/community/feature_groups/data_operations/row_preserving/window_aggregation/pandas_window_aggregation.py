"""Pandas implementation for window aggregation feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

import pandas as pd
import pyarrow as pa

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)

# Mapping from aggregation type to the pandas GroupBy.transform function name.
_PANDAS_AGG_FUNCS: dict[str, str] = {
    "sum": "sum",
    "avg": "mean",
    "count": "count",
    "min": "min",
    "max": "max",
}


class PandasWindowAggregation(WindowAggregationFeatureGroup):
    """Uses pandas groupby().transform() for efficient window computation.
    Accepts and returns PyArrow tables.
    """

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}

    @classmethod
    def _compute_window(
        cls,
        table: pa.Table,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
    ) -> pa.Table:
        """Compute a window aggregation using pandas groupby().transform() and convert back to PyArrow."""
        df = table.to_pandas()

        pandas_func = _PANDAS_AGG_FUNCS.get(agg_type)
        if pandas_func is None:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        # CRITICAL: dropna=False ensures null group keys form their own group,
        # matching PyArrow behavior. Without this, pandas drops null keys entirely.
        result_series = df.groupby(partition_by, dropna=False)[source_col].transform(pandas_func)

        df[feature_name] = result_series

        # Convert count results to int64 (pandas transform may produce float when nulls exist)
        if agg_type == "count":
            df[feature_name] = df[feature_name].astype("int64")

        return pa.Table.from_pandas(df, preserve_index=False)
