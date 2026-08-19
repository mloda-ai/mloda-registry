"""PythonDict implementation for single-column global aggregate broadcast.

Unlike ``aggregation``/``frame_aggregate``, there is no ``partition_by``/
grouping here: exactly one scalar is computed for the whole (post-mask)
column and broadcast to every row, including masked-out rows.
"""

from __future__ import annotations

import statistics
from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_mask_engine import (
    PythonDictMaskEngine,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import row_count

from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import build_mask_from_spec
from mloda.community.feature_groups.data_operations.python_dict_helpers import (
    STD_AGG_TYPES,
    VARIANCE_DDOF,
    is_nan,
    variance,
)
from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.base import (
    ScalarAggregateFeatureGroup,
)


class PythonDictScalarAggregate(ScalarAggregateFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def _compute_aggregation(
        cls,
        data: dict[str, list[Any]],
        feature_name: str,
        source_col: str,
        agg_type: str,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> dict[str, list[Any]]:
        source_values = data[source_col]

        if mask_spec is not None:
            mask = build_mask_from_spec(PythonDictMaskEngine, data, mask_spec)
            source_values = [v if m else None for v, m in zip(source_values, mask)]

        result_value = cls._reduce(agg_type, [v for v in source_values if v is not None])

        num_rows = row_count(data)
        result = dict(data)
        result[feature_name] = [result_value] * num_rows
        return result

    @classmethod
    def _reduce(cls, agg_type: str, non_null: list[Any]) -> Any:
        """Reduce the (already null-filtered) whole-column values to a single scalar."""
        if agg_type == "count":
            return len(non_null)
        if agg_type == "median":
            return statistics.median(non_null) if non_null else None
        if agg_type in VARIANCE_DDOF:
            return variance(non_null, ddof=VARIANCE_DDOF[agg_type], as_std=agg_type in STD_AGG_TYPES)

        if not non_null:
            return None
        if agg_type == "sum":
            return sum(non_null)
        if agg_type in ("avg", "mean"):
            return sum(non_null) / len(non_null)
        if agg_type == "min":
            finite = [v for v in non_null if not is_nan(v)]
            return min(finite) if finite else None
        if agg_type == "max":
            finite = [v for v in non_null if not is_nan(v)]
            return max(finite) if finite else None

        raise unsupported_agg_type_error(agg_type, cls._SUPPORTED_AGG_TYPES, framework="PythonDict")
