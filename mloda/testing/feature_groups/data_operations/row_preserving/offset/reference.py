"""Test reference implementation for offset feature groups.

Accepts PyArrow tables but computes in Python. Used as the cross-framework
comparison baseline in test suites.
"""

from __future__ import annotations

from typing import Any, Set, Type, Union

import pyarrow as pa

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.offset.base import (
    OffsetFeatureGroup,
)


class ReferenceOffset(OffsetFeatureGroup):
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

        # Bulk-extract all needed columns (one C++ call each, not N .as_py() calls)
        partition_lists = {col: table.column(col).to_pylist() for col in partition_by}
        order_vals = table.column(order_by).to_pylist()
        source_vals = table.column(source_col).to_pylist()

        # Build group keys from extracted Python lists (no PyArrow calls in loop)
        groups: dict[tuple[Any, ...], list[tuple[int, Any, Any]]] = {}
        for i in range(num_rows):
            key = tuple(partition_lists[col][i] for col in partition_by)
            if key not in groups:
                groups[key] = []
            groups[key].append((i, order_vals[i], source_vals[i]))

        # Sort each group by order_by (nulls last)
        for key in groups:
            groups[key].sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0))

        result_values: list[Any] = [None] * num_rows

        for key, sorted_rows in groups.items():
            n = len(sorted_rows)
            vals = [row[2] for row in sorted_rows]

            if offset_type.startswith("lag_"):
                offset_n = int(offset_type[len("lag_") :])
                for pos in range(n):
                    idx = sorted_rows[pos][0]
                    if pos >= offset_n:
                        result_values[idx] = vals[pos - offset_n]

            elif offset_type.startswith("lead_"):
                offset_n = int(offset_type[len("lead_") :])
                for pos in range(n):
                    idx = sorted_rows[pos][0]
                    if pos + offset_n < n:
                        result_values[idx] = vals[pos + offset_n]

            elif offset_type.startswith("diff_"):
                offset_n = int(offset_type[len("diff_") :])
                for pos in range(n):
                    idx = sorted_rows[pos][0]
                    curr = vals[pos]
                    if pos >= offset_n and curr is not None and vals[pos - offset_n] is not None:
                        result_values[idx] = curr - vals[pos - offset_n]

            elif offset_type.startswith("pct_change_"):
                offset_n = int(offset_type[len("pct_change_") :])
                for pos in range(n):
                    idx = sorted_rows[pos][0]
                    curr = vals[pos]
                    prev = vals[pos - offset_n] if pos >= offset_n else None
                    if curr is not None and prev is not None and prev != 0:
                        result_values[idx] = (curr - prev) / prev

            elif offset_type == "first_value":
                first = next((v for v in vals if v is not None), None)
                for pos in range(n):
                    result_values[sorted_rows[pos][0]] = first

            elif offset_type == "last_value":
                last = next((v for v in reversed(vals) if v is not None), None)
                for pos in range(n):
                    result_values[sorted_rows[pos][0]] = last

            else:
                raise ValueError(f"Unsupported offset type: {offset_type}")

        new_col = pa.array(result_values)
        return table.append_column(feature_name, new_col)
