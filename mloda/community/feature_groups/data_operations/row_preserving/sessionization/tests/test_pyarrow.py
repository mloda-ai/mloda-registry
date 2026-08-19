"""Tests for PyArrowSessionization compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.community.feature_groups.data_operations.row_preserving.sessionization.pyarrow_sessionization import (
    PyArrowSessionization,
)
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.sessionization.sessionization import (
    SessionizationTestBase,
)


class TestPyArrowSessionization(PyArrowTestMixin, SessionizationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowSessionization

    def test_non_timestamp_order_column_rejected(self) -> None:
        """A non-timestamp (int64) order column must be rejected with a ValueError.

        The PyArrow backend cannot natively sessionize a non-timestamp column;
        casting int64 values to microseconds silently produces wrong gaps. Per
        the CFW backend-rejection rule it must raise a clear ValueError rather
        than compute a fallback.
        """
        table = pa.table(
            {
                "id": pa.array([1, 2, 3], type=pa.int64()),
                "user": pa.array(["A", "A", "A"], type=pa.string()),
                "ts": pa.array([1, 2, 3], type=pa.int64()),
            }
        )
        fs = make_feature_set("ts__sessionize_30_minute", partition_by=["user"], order_by="ts")
        with pytest.raises(ValueError, match=r"(?i)timestamp"):
            PyArrowSessionization.calculate_feature(table, fs)


class TestPyArrowSessionizationNanPartitionKeyGrouping:
    """A NaN partition-key value must merge with itself, matching ``Table.group_by()``.

    ``pc.not_equal(nan, nan)`` is ``True``, so comparing adjacent sorted rows without
    NaN-awareness would split every NaN-keyed row into its own singleton partition (and
    therefore its own session), unlike ``Table.group_by()``, which merges all NaN keys of
    a column into one group.
    """

    def test_nan_partition_rows_share_one_session_matches_pyarrow_group_by(self) -> None:
        import math
        from datetime import datetime, timezone

        u = timezone.utc
        arrow_table = pa.table(
            {
                "grp": pa.array([float("nan"), float("nan"), 1.0], type=pa.float64()),
                "ts": pa.array(
                    [
                        datetime(2023, 1, 1, 10, 0, 0, tzinfo=u),
                        datetime(2023, 1, 1, 10, 5, 0, tzinfo=u),
                        datetime(2023, 1, 1, 10, 0, 0, tzinfo=u),
                    ],
                    type=pa.timestamp("us", tz="UTC"),
                ),
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

        fs = make_feature_set("ts__sessionize_30_minute", partition_by=["grp"], order_by="ts")
        result = PyArrowSessionization.calculate_feature(arrow_table, fs)
        result_col = result.column("ts__sessionize_30_minute").to_pylist()

        # Rows 0 and 1 share the NaN partition PyArrow reports above, and their gap
        # (5 minutes) is within the 30-minute threshold, so they must share one session id.
        assert result_col[0] == result_col[1], (
            f"expected both NaN-partition rows to share one session id (gap 5 min <= 30 min "
            f"threshold, matching PyArrow's own group_by()), got {result_col!r} (PyArrowSessionization "
            "treated every NaN-keyed row as starting a new partition/session)"
        )
        # Row 2 is a genuinely different partition (grp=1.0) and must not share that session.
        assert result_col[2] != result_col[0], f"expected the grp=1.0 row to start its own session, got {result_col!r}"
