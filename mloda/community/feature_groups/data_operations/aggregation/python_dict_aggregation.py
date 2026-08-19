"""PythonDict implementation for aggregation feature groups.

Groups rows in pure Python: builds a ``dict[tuple, list[int]]`` mapping
group-key tuples to row indices, then reduces each group's values per
aggregation type.
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

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
)
from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import build_mask_from_spec
from mloda.community.feature_groups.data_operations.python_dict_helpers import (
    SUPPORTED_AGG_TYPES,
    group_key_value,
    reduce_agg,
)


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
        if agg_type not in SUPPORTED_AGG_TYPES:
            raise unsupported_agg_type_error(agg_type, SUPPORTED_AGG_TYPES, framework="PythonDict")

        partition_by = list(partition_by)
        source_values = data[source_col]

        if mask_spec is not None:
            mask = build_mask_from_spec(PythonDictMaskEngine, data, mask_spec)
            source_values = [v if m else None for v, m in zip(source_values, mask)]

        partition_cols = [data[col] for col in partition_by]

        groups: dict[tuple[Any, ...], list[int]] = {}
        for i in range(len(source_values)):
            key = tuple(group_key_value(col[i]) for col in partition_cols)
            groups.setdefault(key, []).append(i)

        result: dict[str, list[Any]] = {col: [] for col in partition_by}
        result[feature_name] = []

        for indices in groups.values():
            first_idx = indices[0]
            for col_name, col in zip(partition_by, partition_cols):
                result[col_name].append(col[first_idx])
            values = [source_values[i] for i in indices]
            result[feature_name].append(reduce_agg(agg_type, values))

        return result
