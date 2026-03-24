"""PyArrow implementation for window aggregation feature groups."""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional, Set, Type, Union

import pyarrow as pa

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)


class PyArrowWindowAggregation(WindowAggregationFeatureGroup):
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
        order_by: Optional[str] = None,
    ) -> pa.Table:
        num_rows = table.num_rows

        # Build group keys per row (using Python objects so None is a valid key)
        keys: list[tuple[Any, ...]] = []
        for i in range(num_rows):
            key = tuple(table.column(col)[i].as_py() for col in partition_by)
            keys.append(key)

        # Collect source values per group, preserving row indices
        groups: dict[tuple[Any, ...], list[tuple[Any, Any]]] = {}
        for i in range(num_rows):
            key = keys[i]
            val = table.column(source_col)[i].as_py()
            order_val = table.column(order_by)[i].as_py() if order_by else i
            if key not in groups:
                groups[key] = []
            groups[key].append((order_val, val))

        # Compute aggregate per group
        agg_results: dict[tuple[Any, ...], Any] = {}
        for key, pairs in groups.items():
            values = [v for _, v in pairs]
            if agg_type in ("first", "last") and order_by:
                sorted_vals = [v for _, v in sorted(pairs, key=lambda p: (p[0] is None, p[0]))]
                agg_results[key] = cls._aggregate(sorted_vals, agg_type)
            else:
                agg_results[key] = cls._aggregate(values, agg_type)

        # Broadcast back to every row
        result_values = [agg_results[keys[i]] for i in range(num_rows)]

        new_col = pa.array(result_values)
        return table.append_column(feature_name, new_col)

    # -- Aggregation helpers --
    # PyArrow has no native window functions, so this implementation uses
    # Python dict-based grouping with these helper methods. Other frameworks
    # (DuckDB, Polars, Pandas, SQLite) use their native aggregation APIs.

    @classmethod
    def _aggregate(cls, values: list[Any], agg_type: str) -> Any:
        """Compute a single aggregate over a list of values (may contain None)."""
        non_null = [v for v in values if v is not None]

        if not non_null:
            return None

        if agg_type == "sum":
            return sum(non_null)
        if agg_type == "avg":
            return sum(non_null) / len(non_null)
        if agg_type == "count":
            return len(non_null)
        if agg_type == "min":
            return min(non_null)
        if agg_type == "max":
            return max(non_null)
        if agg_type == "std":
            return cls._std(non_null)
        if agg_type == "var":
            return cls._var(non_null)
        if agg_type == "median":
            return cls._median(non_null)
        if agg_type == "mode":
            return cls._mode(non_null)
        if agg_type == "nunique":
            return len(set(non_null))
        if agg_type == "first":
            return non_null[0]
        if agg_type == "last":
            return non_null[-1]

        raise ValueError(f"Unsupported aggregation type: {agg_type}")

    @classmethod
    def _std(cls, values: list[Any]) -> Any:
        if len(values) < 2:
            return None
        return cls._var(values) ** 0.5

    @classmethod
    def _var(cls, values: list[Any]) -> Any:
        if len(values) < 2:
            return None
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / (len(values) - 1)

    @classmethod
    def _median(cls, values: list[Any]) -> Any:
        s = sorted(values)
        n = len(s)
        mid = n // 2
        if n % 2 == 0:
            return (s[mid - 1] + s[mid]) / 2.0
        return float(s[mid])

    @classmethod
    def _mode(cls, values: list[Any]) -> Any:
        if not values:
            return None
        counts = Counter(values)
        return counts.most_common(1)[0][0]
