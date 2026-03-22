"""PyArrow implementation for offset feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

import pyarrow as pa

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.offset.base import (
    OffsetFeatureGroup,
)


class PyArrowOffset(OffsetFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}

    @classmethod
    def _compute_offset(
        cls,
        table: pa.Table,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        offset_type: str,
    ) -> pa.Table:
        num_rows = table.num_rows

        # Build group keys and collect (index, order_value, source_value) per group
        groups: dict[tuple[Any, ...], list[tuple[int, Any, Any]]] = {}
        for i in range(num_rows):
            key = tuple(table.column(col)[i].as_py() for col in partition_by)
            order_val = table.column(order_by)[i].as_py()
            source_val = table.column(source_col)[i].as_py()
            if key not in groups:
                groups[key] = []
            groups[key].append((i, order_val, source_val))

        # Sort each group by order_by (nulls last)
        for key in groups:
            groups[key].sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0))

        result_values: list[Any] = [None] * num_rows

        for key, sorted_rows in groups.items():
            n = len(sorted_rows)
            source_vals = [row[2] for row in sorted_rows]

            if offset_type.startswith("lag_"):
                offset_n = int(offset_type[len("lag_") :])
                for pos in range(n):
                    idx = sorted_rows[pos][0]
                    if pos >= offset_n:
                        result_values[idx] = source_vals[pos - offset_n]

            elif offset_type.startswith("lead_"):
                offset_n = int(offset_type[len("lead_") :])
                for pos in range(n):
                    idx = sorted_rows[pos][0]
                    if pos + offset_n < n:
                        result_values[idx] = source_vals[pos + offset_n]

            elif offset_type.startswith("diff_"):
                offset_n = int(offset_type[len("diff_") :])
                for pos in range(n):
                    idx = sorted_rows[pos][0]
                    curr = source_vals[pos]
                    if pos >= offset_n and curr is not None and source_vals[pos - offset_n] is not None:
                        result_values[idx] = curr - source_vals[pos - offset_n]

            elif offset_type.startswith("pct_change_"):
                offset_n = int(offset_type[len("pct_change_") :])
                for pos in range(n):
                    idx = sorted_rows[pos][0]
                    curr = source_vals[pos]
                    prev = source_vals[pos - offset_n] if pos >= offset_n else None
                    if curr is not None and prev is not None and prev != 0:
                        result_values[idx] = (curr - prev) / prev

            elif offset_type == "first_value":
                # First non-null value in the partition
                first = next((v for v in source_vals if v is not None), None)
                for pos in range(n):
                    result_values[sorted_rows[pos][0]] = first

            elif offset_type == "last_value":
                # Last non-null value in the partition
                last = next((v for v in reversed(source_vals) if v is not None), None)
                for pos in range(n):
                    result_values[sorted_rows[pos][0]] = last

            else:
                raise ValueError(f"Unsupported offset type: {offset_type}")

        new_col = pa.array(result_values)
        return table.append_column(feature_name, new_col)
