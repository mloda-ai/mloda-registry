"""PythonDict implementation for window aggregation feature groups.

The row-preserving twin of ``aggregation/python_dict_aggregation.py``: same
null-handling and tie-breaking rules per aggregation type, but instead of
emitting one row per group it broadcasts the reduced value back to every row
belonging to that group, so the output has the same row count and order as
the input.
"""

from __future__ import annotations

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
    SUPPORTED_AGG_TYPES,
    group_key_value,
    nulls_last_sort_key,
    reduce_agg,
)
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)

# Aggregation types whose result depends on row order within the partition; these
# are the only ones sorted by order_by (nulls last).
_ORDER_DEPENDENT_AGG_TYPES = ("first", "last")


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
        if agg_type not in SUPPORTED_AGG_TYPES:
            raise unsupported_agg_type_error(agg_type, SUPPORTED_AGG_TYPES, framework="PythonDict")

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
            key = tuple(group_key_value(col[i]) for col in partition_cols)
            order_val = order_vals[i] if order_vals is not None else None
            groups.setdefault(key, []).append((i, order_val))

        result_values: list[Any] = [None] * num_rows

        for rows in groups.values():
            if needs_order:
                # Stable-sort ascending, nulls last, before reducing first/last.
                rows = sorted(rows, key=lambda r: nulls_last_sort_key(r[1]))
            values = [source_values[i] for i, _ in rows]
            reduced = reduce_agg(agg_type, values)
            for i, _ in rows:
                result_values[i] = reduced

        result = dict(data)
        result[feature_name] = result_values
        return result
