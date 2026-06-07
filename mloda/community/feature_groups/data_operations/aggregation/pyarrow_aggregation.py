"""PyArrow implementation for aggregation feature groups.

Uses PyArrow's native ``Table.group_by().aggregate()`` API for vectorized,
C++-backed aggregation.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
)
from mloda.community.feature_groups.data_operations.backend_agg_tables import (
    PYARROW_AGG_FUNCS as _PA_AGG_FUNCS,
    PYARROW_ORDERED_FUNCS as _ORDERED_FUNCS,
    PYARROW_SUPPORTED_AGG_TYPES as _SUPPORTED_AGG_TYPES,
    PYARROW_VARIANCE_FUNCS as _VARIANCE_FUNCS,
)
from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import apply_pyarrow_mask


class PyArrowAggregation(AggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PyArrowTable}

    @classmethod
    def _compute_group(
        cls,
        table: pa.Table,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> pa.Table:
        if mask_spec is not None:
            table = apply_pyarrow_mask(table, source_col, mask_spec)

        if agg_type in _PA_AGG_FUNCS:
            pa_func = _PA_AGG_FUNCS[agg_type]
            grouped = table.group_by(partition_by).aggregate([(source_col, pa_func)])
        elif agg_type in _VARIANCE_FUNCS:
            pa_func, ddof = _VARIANCE_FUNCS[agg_type]
            grouped = table.group_by(partition_by).aggregate([(source_col, pa_func, pc.VarianceOptions(ddof=ddof))])
        elif agg_type in _ORDERED_FUNCS:
            pa_func = _ORDERED_FUNCS[agg_type]
            grouped = table.group_by(partition_by, use_threads=False).aggregate([(source_col, pa_func)])
        else:
            raise unsupported_agg_type_error(agg_type, _SUPPORTED_AGG_TYPES, framework="PyArrow")

        # Rename auto-generated column (e.g. "val_sum") to feature_name.
        auto_col = f"{source_col}_{pa_func}"
        names = [feature_name if c == auto_col else c for c in grouped.column_names]
        return grouped.rename_columns(names)
