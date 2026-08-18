"""PythonDict implementation for percentile feature groups.

Pure-Python, dependency-free implementation with no engine-level percentile
limitations, so it targets FULL support: PERCENTILE_CONT-style linear
interpolation over every partition, including multi-key partitions and null
skipping -- matching pandas, DuckDB, and polars-lazy (neither PyArrow nor
SQLite has a production backend for this operation at all).

Groups rows in pure Python (no numpy/pandas/pyarrow): build a
``dict[tuple, list[int]]`` mapping group-key tuples to row indices, collect
each group's non-null (and, if masked, non-masked) values, sort them, and
interpolate the requested percentile the same way
``ReferencePercentile``/``PandasPercentile`` do, then broadcast the one
scalar result back to every row belonging to that partition.
"""

from __future__ import annotations

import math
from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_mask_engine import (
    PythonDictMaskEngine,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import row_count

from mloda.community.feature_groups.data_operations.mask_utils import build_mask_from_spec
from mloda.community.feature_groups.data_operations.row_preserving.percentile.base import (
    PercentileFeatureGroup,
)


class PythonDictPercentile(PercentileFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def _compute_percentile(
        cls,
        data: dict[str, list[Any]],
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        percentile: float,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> dict[str, list[Any]]:
        partition_by = list(partition_by)
        num_rows = row_count(data)
        source_values = data[source_col]

        if mask_spec is not None:
            mask = build_mask_from_spec(PythonDictMaskEngine, data, mask_spec)
            source_values = [v if m else None for v, m in zip(source_values, mask)]

        partition_cols = [data[col] for col in partition_by]

        groups: dict[tuple[Any, ...], list[int]] = {}
        for i in range(num_rows):
            key = tuple(col[i] for col in partition_cols)
            groups.setdefault(key, []).append(i)

        result_values: list[Any] = [None] * num_rows

        for indices in groups.values():
            values = [source_values[i] for i in indices]
            agg_val = cls._percentile_of(values, percentile)
            for i in indices:
                result_values[i] = agg_val

        result = dict(data)
        result[feature_name] = result_values
        return result

    @classmethod
    def _percentile_of(cls, values: list[Any], percentile: float) -> float | None:
        """PERCENTILE_CONT linear interpolation over the non-null values of one partition."""
        non_null = sorted(v for v in values if v is not None)
        n = len(non_null)
        if n == 0:
            return None

        idx = percentile * (n - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return float(non_null[lo])
        return float(non_null[lo] + (idx - lo) * (non_null[hi] - non_null[lo]))
