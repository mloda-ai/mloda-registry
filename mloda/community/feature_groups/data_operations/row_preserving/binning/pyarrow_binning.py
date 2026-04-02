"""PyArrow implementation for binning feature groups."""

from __future__ import annotations

from typing import Set, Type, Union

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

        if op == "bin":
            new_col = cls._equal_width_binning_pyarrow(col, n_bins)
        elif op == "qbin":
            new_col = cls._quantile_binning_pyarrow(table, source_col, n_bins)
        else:
            raise ValueError(f"Unsupported binning operation: {op}")

        return table.append_column(feature_name, new_col)

    @classmethod
    def _equal_width_binning_pyarrow(cls, col: pa.ChunkedArray, n_bins: int) -> pa.Array:
        """Equal-width binning using PyArrow compute operations."""
        col_min_scalar = pc.min(col)
        col_max_scalar = pc.max(col)

        col_min = col_min_scalar.as_py()
        col_max = col_max_scalar.as_py()

        null_mask = pc.is_null(col)
        length = len(col)

        if col_min is None:
            return pa.nulls(length, type=pa.int64())

        if col_min == col_max:
            zeros = pa.array([0] * length, type=pa.int64())
            return pc.if_else(null_mask, pa.scalar(None, type=pa.int64()), zeros)

        bin_width = (col_max - col_min) / n_bins
        col_float = col.cast(pa.float64())

        shifted = pc.subtract(col_float, pa.scalar(float(col_min), type=pa.float64()))
        divided = pc.divide(shifted, pa.scalar(bin_width, type=pa.float64()))
        floored = pc.floor(divided)
        as_int = floored.cast(pa.int64())
        capped = pc.min_element_wise(as_int, pa.scalar(n_bins - 1, type=pa.int64()))

        return pc.if_else(null_mask, pa.scalar(None, type=pa.int64()), capped)

    @classmethod
    def _quantile_binning_pyarrow(cls, table: pa.Table, source_col: str, n_bins: int) -> pa.Array:
        """Quantile (rank-based) binning using PyArrow sort_indices."""
        col = table.column(source_col)
        length = len(col)
        null_mask = pc.is_null(col)

        non_null_count = pc.sum(pc.invert(null_mask).cast(pa.int64())).as_py()

        if non_null_count == 0:
            return pa.nulls(length, type=pa.int64())

        sorted_indices = pc.sort_indices(table, sort_keys=[(source_col, "ascending")], null_placement="at_end")

        result = [None] * length
        for rank in range(non_null_count):
            orig_idx = sorted_indices[rank].as_py()
            result[orig_idx] = min(rank * n_bins // non_null_count, n_bins - 1)

        return pa.array(result, type=pa.int64())
