"""Tests for PyArrowFfill compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.ffill.pyarrow_ffill import (
    PyArrowFfill,
)
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.ffill.ffill import (
    FfillTestBase,
)


class TestPyArrowFfill(PyArrowTestMixin, FfillTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowFfill


class TestPyArrowFfillNanPartitionKeyGrouping:
    """A NaN partition-key value must merge with itself, matching ``Table.group_by()``.

    ``pc.not_equal(nan, nan)`` is ``True``, so comparing adjacent sorted rows without
    NaN-awareness would split every NaN-keyed row into its own singleton partition,
    unlike ``Table.group_by()``, which merges all NaN keys of a column into one group.
    """

    def test_nan_partition_rows_share_one_group_ffill_carries_forward(self) -> None:
        import math

        import pyarrow as pa

        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set

        arrow_table = pa.table(
            {
                "grp": pa.array([float("nan"), float("nan"), 1.0, 2.0], type=pa.float64()),
                "ord": pa.array([1, 2, 3, 4], type=pa.int64()),
                "val": pa.array([10.0, None, 100.0, None], type=pa.float64()),
            }
        )

        t_with_idx = arrow_table.append_column("__idx__", pa.array(range(arrow_table.num_rows)))
        grouped = t_with_idx.group_by(["grp"]).aggregate([("__idx__", "list")])
        idx_lists = grouped.column("__idx___list").to_pylist()
        keys = grouped.column("grp").to_pylist()
        nan_group_rows = next(rows for key, rows in zip(keys, idx_lists) if key is not None and math.isnan(key))
        assert nan_group_rows == [0, 1], (
            f"expected PyArrow's live group_by() to place both NaN-keyed rows (0, 1) in one "
            f"group, got {nan_group_rows!r}"
        )

        fs = make_feature_set("val__ffill", ["grp"], "ord")
        result = PyArrowFfill.calculate_feature(arrow_table, fs)
        result_col = extract_column(result, "val__ffill")

        # Row 1 (ord=2, val=None) is second (by ord) in the shared NaN partition PyArrow
        # reports above, so it must carry row 0's value (10.0) forward instead of staying None.
        assert result_col[1] == 10.0, (
            f"expected row 1's ffill to carry the shared NaN partition's value forward "
            f"(10.0, matching PyArrow's own group_by()), got {result_col[1]!r} (PyArrowFfill "
            "split the NaN rows into separate one-row groups)"
        )
