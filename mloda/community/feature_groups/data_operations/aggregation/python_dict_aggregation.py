"""PythonDict implementation for aggregation feature groups.

Groups rows in pure Python (no numpy/pandas/pyarrow): build a
``dict[tuple, list[int]]`` mapping group-key tuples to row indices (in
first-occurrence order), then reduce each group's values per aggregation
type. Supports all 17 aggregation types (no ``supported_op_subtypes()``
override), matching the DuckDB backend's coverage.
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

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
)
from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import build_mask_from_spec

# All aggregation types natively supported by the PythonDict backend (full parity with
# AggregationFeatureGroupBase.AGGREGATION_TYPES; no subtype restriction).
_SUPPORTED_AGG_TYPES: frozenset[str] = frozenset(
    {
        "sum",
        "avg",
        "mean",
        "count",
        "min",
        "max",
        "std",
        "var",
        "std_pop",
        "std_samp",
        "var_pop",
        "var_samp",
        "median",
        "mode",
        "nunique",
        "first",
        "last",
    }
)

# ddof (delta degrees of freedom) per std/var variant. std/var/std_pop/var_pop are
# population (ddof=0); std_samp/var_samp are sample (ddof=1).
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


class PythonDictAggregation(AggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def _compute_group(
        cls,
        data: dict[str, list[Any]],
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> dict[str, list[Any]]:
        if agg_type not in _SUPPORTED_AGG_TYPES:
            raise unsupported_agg_type_error(agg_type, _SUPPORTED_AGG_TYPES, framework="PythonDict")

        partition_by = list(partition_by)
        source_values = data[source_col]

        if mask_spec is not None:
            mask = build_mask_from_spec(PythonDictMaskEngine, data, mask_spec)
            source_values = [v if m else None for v, m in zip(source_values, mask)]

        partition_cols = [data[col] for col in partition_by]

        groups: dict[tuple[Any, ...], list[int]] = {}
        for i in range(len(source_values)):
            key = tuple(col[i] for col in partition_cols)
            groups.setdefault(key, []).append(i)

        result: dict[str, list[Any]] = {col: [] for col in partition_by}
        result[feature_name] = []

        for key, indices in groups.items():
            for col_name, key_value in zip(partition_by, key):
                result[col_name].append(key_value)
            values = [source_values[i] for i in indices]
            result[feature_name].append(cls._reduce(agg_type, values))

        return result

    @classmethod
    def _reduce(cls, agg_type: str, values: list[Any]) -> Any:
        """Reduce one group's raw (possibly null-containing) values to a single result."""
        non_null = [v for v in values if v is not None]

        if agg_type == "count":
            return len(non_null)
        if agg_type == "nunique":
            return len(set(non_null))
        if agg_type == "mode":
            return cls._mode(values)
        if agg_type == "first":
            return non_null[0] if non_null else None
        if agg_type == "last":
            return non_null[-1] if non_null else None
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
            return min(non_null)
        if agg_type == "max":
            return max(non_null)

        raise unsupported_agg_type_error(agg_type, _SUPPORTED_AGG_TYPES, framework="PythonDict")

    @classmethod
    def _mode(cls, values: list[Any]) -> Any:
        """Most frequent non-null value; ties broken by first occurrence in *values*."""
        counts: dict[Any, tuple[int, int]] = {}
        for i, v in enumerate(values):
            if v is None:
                continue
            count, first_idx = counts.get(v, (0, i))
            counts[v] = (count + 1, first_idx)

        best_value: Any = None
        best_count = -1
        best_idx = -1
        for value, (count, first_idx) in counts.items():
            if count > best_count or (count == best_count and first_idx < best_idx):
                best_value = value
                best_count = count
                best_idx = first_idx

        return best_value

    @classmethod
    def _variance(cls, non_null: list[float], *, ddof: int, as_std: bool) -> float | None:
        """Population (ddof=0) or sample (ddof=1) variance/std of *non_null*.

        Returns ``None`` when there are fewer than ``ddof + 1`` values (e.g. an
        empty group, or a single-value group for the sample variant).
        """
        n = len(non_null)
        if n - ddof <= 0:
            return None
        mean = sum(non_null) / n
        variance = sum((x - mean) ** 2 for x in non_null) / (n - ddof)
        return variance**0.5 if as_std else variance
