"""PythonDict implementation for window aggregation feature groups.

The row-preserving twin of ``aggregation/python_dict_aggregation.py``: same
null-handling and tie-breaking rules per aggregation type, but instead of
emitting one row per group it broadcasts the reduced value back to every row
belonging to that group, so the output has the same row count and order as
the input.
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
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)

# All aggregation types supported by the PythonDict backend (no subtype restriction).
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

# Aggregation types whose result depends on row order within the partition; these
# are the only ones sorted by order_by (nulls last).
_ORDER_DEPENDENT_AGG_TYPES = ("first", "last")

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

# Sentinel substituted for any NaN partition-key value so that all NaN-valued rows of a
# partition column hash/compare equal and land in one shared group, matching PyArrow's
# Table.group_by() (which merges all NaN keys into a single group, distinct from None).
_NAN_KEY = object()


class PythonDictWindowAggregation(WindowAggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def _compute_window(
        cls,
        data: dict[str, list[Any]],
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        order_by: str | None = None,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> dict[str, list[Any]]:
        if agg_type not in _SUPPORTED_AGG_TYPES:
            raise unsupported_agg_type_error(agg_type, _SUPPORTED_AGG_TYPES, framework="PythonDict")

        partition_by = list(partition_by)
        num_rows = row_count(data)
        source_values = data[source_col]

        if mask_spec is not None:
            mask = build_mask_from_spec(PythonDictMaskEngine, data, mask_spec)
            source_values = [v if m else None for v, m in zip(source_values, mask)]

        partition_cols = [data[col] for col in partition_by]
        order_vals: list[Any] | None = data[order_by] if order_by is not None else None
        needs_order = order_vals is not None and agg_type in _ORDER_DEPENDENT_AGG_TYPES

        # order_val is carried along but only consulted below for first/last;
        # every other aggregation type is order-independent.
        groups: dict[tuple[Any, ...], list[tuple[int, Any]]] = {}
        for i in range(num_rows):
            key = tuple(cls._group_key_value(col[i]) for col in partition_cols)
            order_val = order_vals[i] if order_vals is not None else None
            groups.setdefault(key, []).append((i, order_val))

        result_values: list[Any] = [None] * num_rows

        for rows in groups.values():
            if needs_order:
                # Stable-sort ascending, nulls last, before reducing first/last.
                rows = sorted(rows, key=lambda r: (r[1] is None, r[1] if r[1] is not None else 0))
            values = [source_values[i] for i, _ in rows]
            reduced = cls._reduce(agg_type, values)
            for i, _ in rows:
                result_values[i] = reduced

        result = dict(data)
        result[feature_name] = result_values
        return result

    @classmethod
    def _group_key_value(cls, value: Any) -> Any:
        """Map a partition-column value to a hashable, NaN-safe group-key component.

        ``float('nan')`` never equals another NaN, so used raw as part of a dict key
        it would split every NaN-valued row into its own singleton group. Substitute
        a shared sentinel so all NaN values collapse into one group key component,
        matching PyArrow's ``Table.group_by()`` (which merges all NaN keys of a
        column into a single group, distinct from a null/None group).
        """
        if isinstance(value, float) and math.isnan(value):
            return _NAN_KEY
        return value

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
            finite = [v for v in non_null if not cls._is_nan(v)]
            return min(finite) if finite else None
        if agg_type == "max":
            finite = [v for v in non_null if not cls._is_nan(v)]
            return max(finite) if finite else None

        raise unsupported_agg_type_error(agg_type, _SUPPORTED_AGG_TYPES, framework="PythonDict")

    @staticmethod
    def _is_nan(value: Any) -> bool:
        """True for a float NaN value; min/max must skip these (PyArrow's pc.min/pc.max do)."""
        return isinstance(value, float) and math.isnan(value)

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
