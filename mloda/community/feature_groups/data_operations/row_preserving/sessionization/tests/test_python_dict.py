"""Tests for PythonDictSessionization compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.sessionization.python_dict_sessionization import (
    PythonDictSessionization,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.sessionization.sessionization import (
    SessionizationTestBase,
)


class TestPythonDictSessionization(PythonDictTestMixin, SessionizationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictSessionization


class TestPythonDictNullPartitionKey:
    """A null ``partition_by`` value mixed with non-null values must not crash.

    ``PythonDictSessionization._compute_session`` builds the sort key as
    ``(partition_key(i), order_vals[i])`` where ``partition_key(i)`` embeds the
    raw (possibly ``None``) partition column value directly into the tuple.
    When the same partition column holds both ``None`` and non-null strings,
    Python's ``sorted()`` compares two of those tuples with ``<`` and raises
    ``TypeError: '<' not supported between instances of 'NoneType' and 'str'``.

    Six sibling PythonDict backends in this operation family (``rank``,
    ``offset``, ``ffill``, ``ema``, ``frame_aggregate``, ``window_aggregation``)
    avoid exactly this by sorting on a null-safe key of the shape
    ``(v is None, v if v is not None else 0)`` (see
    ``python_dict_offset.py``'s ``rows.sort(key=...)`` call) instead of
    comparing raw values. Sessionization does not yet follow that pattern.

    PyArrow (the live reference oracle) does not crash: rows whose partition
    key is null get session id ``None`` rather than being folded into another
    partition or raising.
    """

    def test_null_partition_value_matches_pyarrow_oracle(self) -> None:
        from datetime import datetime, timezone

        import pyarrow as pa

        from mloda.community.feature_groups.data_operations.row_preserving.sessionization.pyarrow_sessionization import (
            PyArrowSessionization,
        )
        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set

        u = timezone.utc
        arrow_table = pa.table(
            {
                "user": pa.array(["A", None, "A", None, "B"], type=pa.string()),
                "ts": pa.array(
                    [
                        datetime(2023, 1, 1, 10, 0, 0, tzinfo=u),
                        datetime(2023, 1, 1, 10, 5, 0, tzinfo=u),
                        datetime(2023, 1, 1, 10, 40, 0, tzinfo=u),
                        datetime(2023, 1, 1, 11, 0, 0, tzinfo=u),
                        datetime(2023, 1, 1, 10, 0, 0, tzinfo=u),
                    ],
                    type=pa.timestamp("us", tz="UTC"),
                ),
            }
        )
        data = {name: arrow_table.column(name).to_pylist() for name in arrow_table.column_names}
        fs = make_feature_set("ts__sessionize_30_minute", partition_by=["user"], order_by="ts")

        result = PythonDictSessionization.calculate_feature(data, fs)
        oracle = PyArrowSessionization.calculate_feature(arrow_table, fs)

        result_col = extract_column(result, "ts__sessionize_30_minute")
        oracle_col = extract_column(oracle, "ts__sessionize_30_minute")
        assert result_col == oracle_col, f"{result_col!r} != PyArrow oracle {oracle_col!r}"


class TestPythonDictNanPartitionKeyGrouping:
    """A NaN partition-key value must group with itself, not start a new session every row.

    Comparing raw partition-key tuples with ``key != prev_key`` treats every NaN-keyed row
    after the first as starting a brand new partition (and therefore a brand new session),
    since ``nan != nan``. This test asks PyArrow's own ``Table.group_by()`` directly which
    rows it considers one partition (see
    sessionization/tests/test_pyarrow.py::TestPyArrowSessionizationNanPartitionKeyGrouping
    for the equivalent check against ``PyArrowSessionization`` itself, which merges NaN
    partition keys too).
    """

    def test_nan_partition_rows_share_one_session_matches_pyarrow_group_by(self) -> None:
        import math
        from datetime import datetime, timezone

        import pyarrow as pa

        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set

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

        data = {name: arrow_table.column(name).to_pylist() for name in arrow_table.column_names}
        fs = make_feature_set("ts__sessionize_30_minute", partition_by=["grp"], order_by="ts")
        result = PythonDictSessionization.calculate_feature(data, fs)
        result_col = extract_column(result, "ts__sessionize_30_minute")

        # Rows 0 and 1 share the NaN partition PyArrow reports above, and their gap
        # (5 minutes) is within the 30-minute threshold, so they must share one session id.
        assert result_col[0] == result_col[1], (
            f"expected both NaN-partition rows to share one session id (gap 5 min <= 30 min "
            f"threshold, matching PyArrow's own group_by()), got {result_col!r} (PythonDict "
            "treated every NaN-keyed row as starting a new partition/session)"
        )
        # Row 2 is a genuinely different partition (grp=1.0) and must not share that session.
        assert result_col[2] != result_col[0], (
            f"row 2 (a different partition) must not share row 0's session: {result_col!r}"
        )


class TestPythonDictNanOrderValueSortSurvives:
    """A NaN ``order_by`` value must not corrupt the sort of the surrounding rows.

    Each row's sort key routes through the shared ``nulls_last_sort_key``, which maps both
    ``None`` and NaN to the same last-place sentinel so a NaN ``order_by`` value never gets
    directly compared with ``<`` against a real ``(0, <datetime>)`` entry. That must not
    prevent the two genuinely time-ordered rows in the same partition from landing in the
    same session, regardless of how the NaN row's own resulting session id is disambiguated.
    """

    def test_nan_order_value_does_not_break_sort_of_surrounding_rows(self) -> None:
        from datetime import datetime, timezone

        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set

        u = timezone.utc
        data = {
            "grp": ["A", "A", "A"],
            "ts": [
                datetime(2023, 1, 1, 10, 0, 0, tzinfo=u),
                float("nan"),
                datetime(2023, 1, 1, 10, 5, 0, tzinfo=u),
            ],
        }
        fs = make_feature_set("ts__sessionize_30_minute", partition_by=["grp"], order_by="ts")

        result = PythonDictSessionization.calculate_feature(data, fs)
        result_col = extract_column(result, "ts__sessionize_30_minute")

        # Rows 0 and 2 are 5 minutes apart (within the 30-minute threshold) and must
        # share one session, regardless of what happens to row 1's own NaN-order value.
        assert result_col[0] == result_col[2], (
            f"expected the two real-time rows (0, 2; 5 min apart) to share one session "
            f"despite the NaN order_by value on row 1, got {result_col!r}"
        )


class TestPythonDictSessionizationNanAndNoneAreDistinctContiguousGroups:
    """DEFECT F: NaN and None partition keys must be distinct, each internally contiguous.

    ``sort_key`` routes every partition-key component through ``nulls_last_sort_key``, which
    maps both ``None`` and NaN to the identical tier ``(1, 0)``; when the overall sort ties on
    that tier, ``order_by`` breaks the tie, so a None-keyed row can sort BETWEEN two
    NaN-keyed rows and split what should be one contiguous NaN run into two. PyArrow's own
    ``Table.group_by()`` keeps the null group and the NaN group separate, each internally
    contiguous, so the two NaN rows below must land in one uninterrupted (non-poisoned)
    session, distinct from the None row's own session.
    """

    def test_nan_rows_stay_contiguous_and_the_none_row_is_a_separate_partition(self) -> None:
        from datetime import datetime, timezone

        u = timezone.utc
        data: dict[str, list[Any]] = {
            "grp": [float("nan"), None, float("nan")],
            "ts": [
                datetime(2023, 1, 1, 10, 0, 0, tzinfo=u),
                datetime(2023, 1, 1, 10, 1, 0, tzinfo=u),
                datetime(2023, 1, 1, 10, 5, 0, tzinfo=u),
            ],
        }
        result = PythonDictSessionization._compute_session(data, "sess", "ts", 1800, ["grp"])

        assert result["sess"][0] is not None, (
            f"expected the two NaN-partition rows to form a real (non-poisoned) session, got {result['sess']!r}"
        )
        assert result["sess"][0] == result["sess"][2], (
            f"expected the two NaN-partition rows (0 and 2, gap 5 min <= 30 min threshold) "
            f"to share one session id: {result['sess']!r}"
        )
        assert result["sess"][1] != result["sess"][0], (
            f"expected the None-partition row (1) to belong to a different session than the "
            f"NaN partition: {result['sess']!r}"
        )
