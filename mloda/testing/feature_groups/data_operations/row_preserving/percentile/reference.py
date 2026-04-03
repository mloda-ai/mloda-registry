"""Test reference implementation for percentile feature groups.

Accepts PyArrow tables but computes in Python. Used as the cross-framework
comparison baseline in test suites. Uses PyArrow's group_by with list
collection, then computes the percentile from the collected values using
PyArrow's quantile function.
"""

from __future__ import annotations

from typing import Any, Set, Type, Union

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.percentile.base import (
    PercentileFeatureGroup,
)

_IDX_COL = "__mloda_pctl_idx__"


class ReferencePercentile(PercentileFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}

    @classmethod
    def _compute_percentile(
        cls,
        table: pa.Table,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        percentile: float,
    ) -> pa.Table:
        num_rows = table.num_rows
        t_with_idx = table.append_column(_IDX_COL, pa.array(range(num_rows)))

        grouped = t_with_idx.group_by(partition_by).aggregate(
            [
                (source_col, "list"),
                (_IDX_COL, "list"),
            ]
        )
        list_col = f"{source_col}_list"
        idx_list_col = f"{_IDX_COL}_list"

        result_values: list[Any] = [None] * num_rows

        for g in range(grouped.num_rows):
            vals = grouped.column(list_col)[g].as_py()
            indices = grouped.column(idx_list_col)[g].as_py()
            non_null = [v for v in vals if v is not None]

            if not non_null:
                agg_val = None
            else:
                arr = pa.array(non_null, type=pa.float64())
                q_result = pc.quantile(arr, q=percentile)
                agg_val = q_result[0].as_py()

            for idx in indices:
                result_values[idx] = agg_val

        return table.append_column(feature_name, pa.array(result_values))
