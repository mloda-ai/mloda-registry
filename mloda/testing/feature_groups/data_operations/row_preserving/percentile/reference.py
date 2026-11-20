"""Test reference implementation for percentile feature groups.

Accepts PyArrow tables but computes in pure Python. Used as the cross-framework
comparison baseline in test suites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mloda.provider import FeatureSet

from mloda.testing.feature_groups.data_operations.aggregation_helpers import group_rows, quantile
from mloda.testing.feature_groups.data_operations.helpers import masked_column

if TYPE_CHECKING:
    import pyarrow as pa


def _compute_percentile(
    table: pa.Table,
    feature_name: str,
    source_col: str,
    partition_by: list[str],
    percentile: float,
    mask_spec: list[tuple[str, str, Any]] | None = None,
) -> pa.Table:
    import pyarrow as pa

    num_rows = table.num_rows
    keys = [table.column(col).to_pylist() for col in partition_by]
    source = masked_column(table, source_col, mask_spec) if mask_spec is not None else table.column(source_col)
    values = source.to_pylist()

    result_values: list[Any] = [None] * num_rows
    for rows in group_rows(keys, num_rows).values():
        agg_val = quantile([values[i] for i in rows], percentile)
        for i in rows:
            result_values[i] = agg_val

    return table.append_column(feature_name, pa.array(result_values, type=pa.float64()))


def reference_percentile(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """Run the group's percentile loop with the pure Python reference compute."""
    return group.percentile(table, features, compute_percentile=_compute_percentile)
