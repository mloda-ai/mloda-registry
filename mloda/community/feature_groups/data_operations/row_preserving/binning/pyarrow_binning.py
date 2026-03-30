"""PyArrow implementation for binning feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

import pyarrow as pa

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
        values = table.column(source_col).to_pylist()

        non_null = [v for v in values if v is not None]

        if not non_null:
            result_values: list[Any] = [None] * len(values)
            new_col = pa.array(result_values, type=pa.int64())
            return table.append_column(feature_name, new_col)

        if op == "bin":
            result_values = cls._equal_width_binning(values, non_null, n_bins)
        elif op == "qbin":
            result_values = cls._quantile_binning(values, n_bins)
        else:
            raise ValueError(f"Unsupported binning operation: {op}")

        new_col = pa.array(result_values, type=pa.int64())
        return table.append_column(feature_name, new_col)
