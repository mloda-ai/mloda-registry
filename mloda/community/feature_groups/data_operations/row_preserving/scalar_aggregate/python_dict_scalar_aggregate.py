"""PythonDict implementation for single-column global aggregate broadcast.

Unlike ``aggregation``/``frame_aggregate``, there is no ``partition_by``/
grouping here: exactly one scalar is computed for the whole (post-mask)
column and broadcast to every row, including masked-out rows.
"""

from __future__ import annotations

import math
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
from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.base import (
    ScalarAggregateFeatureGroup,
)

# ddof (delta degrees of freedom) per std/var variant. std/var/std_pop/var_pop are
# population (ddof=0); std_samp/var_samp are sample (ddof=1). Mirrors
# python_dict_aggregation.py's mapping.
_VARIANCE_DDOF: dict[str, int] = {
    "std": 0,
    "var": 0,
    "std_pop": 0,
    "var_pop": 0,
    "std_samp": 1,
    "var_samp": 1,
}

# std_* variants take a square root of the variance; var_* variants return the variance.
_STD_AGG_TYPES: frozenset[str] = frozenset({"std", "std_pop", "std_samp"})


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
        if agg_type in _VARIANCE_DDOF:
            return cls._variance(non_null, ddof=_VARIANCE_DDOF[agg_type], as_std=agg_type in _STD_AGG_TYPES)

        if not non_null:
            return None
        if agg_type == "sum":
            return sum(non_null)
        if agg_type in ("avg", "mean"):
            return sum(non_null) / len(non_null)
        if agg_type == "min":
            finite = [v for v in non_null if not cls._is_nan(v)]
            return min(finite) if finite else None
        if agg_type == "max":
            finite = [v for v in non_null if not cls._is_nan(v)]
            return max(finite) if finite else None

        raise unsupported_agg_type_error(agg_type, cls._SUPPORTED_AGG_TYPES, framework="PythonDict")

    @staticmethod
    def _is_nan(value: Any) -> bool:
        """True for a float NaN value; min/max must skip these (PyArrow's pc.min/pc.max do)."""
        return isinstance(value, float) and math.isnan(value)

    @classmethod
    def _variance(cls, non_null: list[float], *, ddof: int, as_std: bool) -> float | None:
        """Population (ddof=0) or sample (ddof=1) variance/std of *non_null*.

        Returns ``None`` when there are fewer than ``ddof + 1`` values (e.g. an
        empty column, or a single-value column for the sample variant).
        """
        n = len(non_null)
        if n - ddof <= 0:
            return None
        mean = sum(non_null) / n
        variance = sum((x - mean) ** 2 for x in non_null) / (n - ddof)
        return variance**0.5 if as_std else variance
