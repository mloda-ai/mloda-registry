"""PyArrow implementation for binning feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.binning.base import (
    BinningFeatureGroup,
)


class PyArrowBinning(BinningFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}

    @classmethod
    def _compute_binning(
        cls,
        table: pa.Table,
        feature_name: str,
        source_col: str,
        op: str,
        n_bins: int,
    ) -> pa.Table:
        col = table.column(source_col)

        nan_mask = pc.is_nan(col)
        if pc.any(nan_mask).as_py():
            col = pc.if_else(nan_mask, None, col)

        non_null_count = pc.sum(pc.cast(pc.is_valid(col), pa.int64())).as_py()

        if non_null_count == 0:
            null_col = pa.array([None] * len(col), type=pa.int64())
            return table.append_column(feature_name, null_col)

        if op == "bin":
            result_array = cls._equal_width_bin(col, n_bins)
        elif op == "qbin":
            result_array = cls._quantile_bin(col, n_bins, non_null_count)
        else:
            raise ValueError(f"Unsupported binning operation: {op}")

        return table.append_column(feature_name, result_array)

    @classmethod
    def _equal_width_bin(cls, col: pa.ChunkedArray, n_bins: int) -> pa.Array:
        col_min = pc.min(col)
        col_max = pc.max(col)

        if pc.equal(col_min, col_max).as_py():
            result = pc.if_else(pc.is_null(col), None, 0)
            return pc.cast(result, pa.int64())

        bin_width = pc.divide(
            pc.subtract(col_max, col_min).cast(pa.float64()),
            pa.scalar(n_bins, type=pa.float64()),
        )

        shifted = pc.subtract(col.cast(pa.float64()), col_min.cast(pa.float64()))
        raw_bin = pc.floor(pc.divide(shifted, bin_width))

        max_bin = pa.scalar(n_bins - 1, type=pa.float64())
        clamped = pc.min_element_wise(raw_bin, max_bin)

        result = pc.if_else(pc.is_null(col), None, clamped)
        return pc.cast(result, pa.int64())

    @classmethod
    def _quantile_bin(cls, col: pa.ChunkedArray, n_bins: int, non_null_count: int) -> pa.Array:
        indices = pc.sort_indices(col, null_placement="at_end")
        n = non_null_count

        result_values: list[Any] = [None] * len(col)

        for rank in range(n):
            original_idx = indices[rank].as_py()
            bin_val = rank * n_bins // n
            result_values[original_idx] = bin_val

        return pa.array(result_values, type=pa.int64())
