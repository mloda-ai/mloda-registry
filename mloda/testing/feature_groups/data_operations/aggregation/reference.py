"""Pure Python reference for aggregation: runs the loop of the group passed in with a kernel-backed compute.

pa.Table is only the I/O container; grouping and aggregation run on ``to_pylist()`` lists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from mloda.provider import FeatureSet

from mloda.testing.feature_groups.data_operations.aggregation_helpers import (
    compute_groups,
)

if TYPE_CHECKING:
    import pyarrow as pa


class AggregationSpec(Protocol):
    """One aggregation as the group's loop hands it to the compute."""

    @property
    def feature_name(self) -> str: ...

    @property
    def source_col(self) -> str: ...

    @property
    def agg_type(self) -> str: ...

    @property
    def mask_spec(self) -> list[tuple[str, str, Any]] | None: ...


def _compute_group(table: pa.Table, specs: list[AggregationSpec], partition_by: list[str]) -> pa.Table:
    """One row per group (first-occurrence order): the partition columns plus one column per spec."""
    return compute_groups(table, specs, partition_by)


def reference_aggregation(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """Run the group's aggregation loop with the pure Python reference compute."""
    return group.aggregate(table, features, compute_group=_compute_group)
