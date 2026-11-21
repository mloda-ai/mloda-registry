"""Pure Python reference for aggregation: runs the loop of the group passed in with a kernel-backed compute.

pa.Table is only the I/O container; grouping and aggregation run on ``to_pylist()`` lists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mloda.provider import FeatureSet

from mloda.testing.feature_groups.data_operations.aggregation_helpers import (
    compute_groups,
)

if TYPE_CHECKING:
    import pyarrow as pa


def reference_aggregation(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """Run the group's aggregation loop with the pure Python reference compute."""
    return group.aggregate(table, features, compute_group=compute_groups)
