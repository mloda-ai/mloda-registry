"""PyArrow implementation for single-column global aggregate broadcast."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.mask_utils import build_mask_from_spec
from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.base import (
    ScalarAggregateFeatureGroup,
)
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_filter_mask_engine import (
    PyArrowFilterMaskEngine,
)


class PyArrowScalarAggregate(ScalarAggregateFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PyArrowTable}

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
            mask = build_mask_from_spec(PyArrowFilterMaskEngine, table, mask_spec)
            null_scalar = pa.scalar(None, type=table.schema.field(source_col).type)
            masked_col = pc.if_else(pc.fill_null(mask, False), table.column(source_col), null_scalar)
            col_idx = table.schema.get_field_index(source_col)
            table = table.set_column(col_idx, source_col, masked_col)

        column = table.column(source_col)

        if agg_type == "sum":
            result = pc.sum(column).as_py()
        elif agg_type == "min":
            result = pc.min(column).as_py()
        elif agg_type == "max":
            result = pc.max(column).as_py()
        elif agg_type in ("avg", "mean"):
            result = pc.mean(column).as_py()
        elif agg_type == "count":
            result = pc.count(column).as_py()
        elif agg_type in ("std", "std_pop"):
            result = pc.stddev(column).as_py()
        elif agg_type in ("var", "var_pop"):
            result = pc.variance(column).as_py()
        elif agg_type == "std_samp":
            result = pc.stddev(column, ddof=1).as_py()
        elif agg_type == "var_samp":
            result = pc.variance(column, ddof=1).as_py()
        elif agg_type == "median":
            q_result = pc.quantile(column, q=0.5)
            if len(q_result) == 0:
                raise ValueError("pc.quantile returned an empty result for median computation")
            result = q_result[0].as_py()
        else:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        repeated = pa.array([result] * table.num_rows)
        return table.append_column(feature_name, repeated)
