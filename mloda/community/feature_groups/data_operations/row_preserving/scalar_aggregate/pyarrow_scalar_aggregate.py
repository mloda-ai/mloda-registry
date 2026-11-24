"""PyArrow implementation for single-column global aggregate broadcast."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import apply_pyarrow_mask
from mloda.community.feature_groups.data_operations.pyarrow_helpers import (
    needs_all_nan_repair,
    restore_all_nan_extrema,
)
from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.base import (
    ScalarAggregateFeatureGroup,
)


class PyArrowScalarAggregate(Implementation[ScalarAggregateFeatureGroup]):
    feature_group = ScalarAggregateFeatureGroup
    compute_framework = PyArrowTable

    @classmethod
    def calculate(
        cls,
        group: type[ScalarAggregateFeatureGroup],
        data: Any,
        features: FeatureSet,
        framework: ComputeFramework | None,
    ) -> Any:
        return group.aggregate(data, features, compute_aggregation=cls._compute_aggregation)

    @classmethod
    def _compute_aggregation(
        cls,
        table: pa.Table,
        feature_name: str,
        source_col: str,
        agg_type: str,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> pa.Table:
        if mask_spec is not None:
            column = apply_pyarrow_mask(table, source_col, mask_spec)
        else:
            column = table.column(source_col)

        if agg_type == "sum":
            result = pc.sum(column)
        elif agg_type in ("min", "max"):
            result = pc.min(column) if agg_type == "min" else pc.max(column)
            # Older Arrow returns +/-inf for an all-NaN column.
            if needs_all_nan_repair(agg_type, column.type):
                result = restore_all_nan_extrema(result, pc.all(pc.is_nan(column)))
        elif agg_type in ("avg", "mean"):
            result = pc.mean(column)
        elif agg_type == "count":
            result = pc.count(column)
        elif agg_type in ("std", "std_pop"):
            result = pc.stddev(column)
        elif agg_type in ("var", "var_pop"):
            result = pc.variance(column)
        elif agg_type == "std_samp":
            result = pc.stddev(column, ddof=1)
        elif agg_type == "var_samp":
            result = pc.variance(column, ddof=1)
        elif agg_type == "median":
            q_result = pc.quantile(column, q=0.5)
            if len(q_result) == 0:
                raise ValueError("pc.quantile returned an empty result for median computation")
            result = q_result[0]
        else:
            raise unsupported_agg_type_error(
                agg_type, ScalarAggregateFeatureGroup._SUPPORTED_AGG_TYPES, framework="PyArrow"
            )

        # The scalar's own type keeps the output type on zero rows, where an untyped empty array is null-typed.
        repeated = pa.array([result.as_py()] * table.num_rows, type=result.type)
        return table.append_column(feature_name, repeated)
