"""Test reference implementation for resample feature groups, pure Python over PyArrow tables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from mloda.provider import FeatureSet

from mloda.testing.feature_groups.data_operations.aggregation_helpers import compute_groups
from mloda.testing.feature_groups.data_operations.row_preserving.bucket_helpers import (
    assert_timestamp,
    bucket_timestamp,
)

if TYPE_CHECKING:
    import pyarrow as pa


class _Spec(NamedTuple):
    """A resample aggregate in the shape ``compute_groups`` reads."""

    feature_name: str
    source_col: str
    agg_type: str
    mask_spec: None = None


def _compute(
    table: pa.Table, aggs: list[Any], time_column: str, partition_by: list[str], n: int, unit: str
) -> pa.Table:
    import pyarrow as pa

    assert_timestamp(table, time_column, "resample", allow_fixed_offset=False)
    column = table.column(time_column)
    floored = [None if v is None else bucket_timestamp(v, "floor", n, unit) for v in column.to_pylist()]
    bucketed = table.set_column(
        table.schema.get_field_index(time_column), time_column, pa.array(floored, type=column.type)
    )
    specs = [_Spec(a.feature_name, a.source_col, a.agg) for a in aggs]
    return compute_groups(bucketed, specs, [*partition_by, time_column])


def reference_resample(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """Run the group's resample loop with the pure Python reference compute."""
    return group.resample(table, features, compute_resample=_compute)
