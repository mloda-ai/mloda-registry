"""Pure Python reference for scalar aggregate: the shared loop with a kernel-backed compute.

pa.Table is only the I/O container; the aggregate runs on a ``to_pylist()`` list.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from mloda.provider import FeatureSet

from mloda.testing.feature_groups.data_operations.aggregation_helpers import (
    aggregate,
    declared_type,
    unsupported_agg_type_error,
)
from mloda.testing.feature_groups.data_operations.helpers import masked_column

SUPPORTED_SCALAR_AGG_TYPES = frozenset(
    {
        "sum",
        "min",
        "max",
        "avg",
        "mean",
        "count",
        "std",
        "var",
        "std_pop",
        "std_samp",
        "var_pop",
        "var_samp",
        "median",
    }
)


def _compute_aggregation(
    table: pa.Table,
    feature_name: str,
    source_col: str,
    agg_type: str,
    mask_spec: list[tuple[str, str, Any]] | None = None,
) -> pa.Table:
    if agg_type not in SUPPORTED_SCALAR_AGG_TYPES:
        raise unsupported_agg_type_error(agg_type, SUPPORTED_SCALAR_AGG_TYPES, framework="Reference")

    source = masked_column(table, source_col, mask_spec) if mask_spec is not None else table.column(source_col)
    value = aggregate(source.to_pylist(), agg_type)
    out_type = declared_type(agg_type, table.schema.field(source_col).type)
    return table.append_column(feature_name, pa.array([value] * table.num_rows, type=out_type))


def reference_scalar_aggregate(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """Run the group's scalar aggregate loop with the pure Python reference compute."""
    return group.aggregate(table, features, compute_aggregation=_compute_aggregation)
