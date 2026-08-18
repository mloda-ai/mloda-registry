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
