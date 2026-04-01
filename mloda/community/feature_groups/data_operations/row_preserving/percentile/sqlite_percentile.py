"""SQLite implementation for percentile feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.percentile.base import (
    PercentileFeatureGroup,
)


def _linear_quantile(sorted_values: list[float], q: float) -> float:
    """Compute quantile with linear interpolation, matching PyArrow/pandas default."""
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    idx = (n - 1) * q
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return float(sorted_values[lo])
    frac = idx - lo
    return float(sorted_values[lo]) + frac * (float(sorted_values[hi]) - float(sorted_values[lo]))


class SqlitePercentile(PercentileFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {SqliteFramework}

    @classmethod
    def _compute_percentile(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        percentile: float,
    ) -> SqliteRelation:
        arrow_table = data.to_arrow_table()
        num_rows = arrow_table.num_rows

        partition_arrays = [arrow_table.column(col).to_pylist() for col in partition_by]
        source_values = arrow_table.column(source_col).to_pylist()

        groups: dict[tuple[Any, ...], list[int]] = {}
        for i in range(num_rows):
            key = tuple(partition_arrays[j][i] for j in range(len(partition_by)))
            groups.setdefault(key, []).append(i)

        result_values: list[Any] = [None] * num_rows

        for key, indices in groups.items():
            non_null = sorted(v for v in (source_values[i] for i in indices) if v is not None)

            if not non_null:
                agg_val = None
            else:
                agg_val = _linear_quantile(non_null, percentile)

            for idx in indices:
                result_values[idx] = agg_val

        return data.append_column(feature_name, result_values)
