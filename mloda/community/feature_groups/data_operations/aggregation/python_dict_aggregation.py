"""PythonDict implementation for aggregation feature groups.

Groups rows in pure Python: builds a ``dict[tuple, list[int]]`` mapping
group-key tuples to row indices, then reduces each group's values per
aggregation type.
"""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_mask_engine import (
    PythonDictMaskEngine,
)

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
    AggregationSpec,
)
from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import build_mask_from_spec
from mloda.community.feature_groups.data_operations.python_dict_helpers import (
    SUPPORTED_AGG_TYPES,
    group_key_value,
    reduce_agg,
)


class PythonDictAggregation(Implementation[AggregationFeatureGroup]):
    feature_group = AggregationFeatureGroup
    compute_framework = PythonDictFramework

    @classmethod
    def calculate(
        cls, group: type[AggregationFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.aggregate(data, features, compute_group=cls._compute_group)

    @classmethod
    def _compute_group(
        cls, data: dict[str, list[Any]], specs: list[AggregationSpec], partition_by: list[str]
    ) -> dict[str, list[Any]]:
        for spec in specs:
            if spec.agg_type not in SUPPORTED_AGG_TYPES:
                raise unsupported_agg_type_error(spec.agg_type, SUPPORTED_AGG_TYPES, framework="PythonDict")

        partition_by = list(partition_by)

        # Each spec's masked values come from the original data.
        source_values: list[list[Any]] = []
        for spec in specs:
            values = data[spec.source_col]
            if spec.mask_spec is not None:
                mask = build_mask_from_spec(PythonDictMaskEngine, data, spec.mask_spec)
                values = [v if m else None for v, m in zip(values, mask)]
            source_values.append(values)

        partition_cols = [data[col] for col in partition_by]

        groups: dict[tuple[Any, ...], list[int]] = {}
        if not partition_by:
            # No partition columns: the whole table is one group, even with zero rows.
            groups[()] = []
        for i in range(len(data[specs[0].source_col])):
            key = tuple(group_key_value(col[i]) for col in partition_cols)
            groups.setdefault(key, []).append(i)

        result: dict[str, list[Any]] = {col: [] for col in partition_by}
        for spec in specs:
            result[spec.feature_name] = []

        for indices in groups.values():
            if partition_by:
                first_idx = indices[0]
                for col_name, col in zip(partition_by, partition_cols):
                    result[col_name].append(col[first_idx])
            for spec, values in zip(specs, source_values):
                result[spec.feature_name].append(reduce_agg(spec.agg_type, [values[i] for i in indices]))

        return result
