"""Tests for PythonDictEma compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.ema.python_dict_ema import (
    PythonDictEma,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.ema.ema import (
    EmaTestBase,
)


class TestPythonDictEma(PythonDictTestMixin, EmaTestBase):
    """All value/semantics/error tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictEma


class TestPythonDictEmaNanPartitionKeyGrouping:
    """A NaN partition-key value must group with itself, not split into singletons.

    Two rows that both carry ``float('nan')`` in a partition column compare unequal
    (``nan != nan``), so building the group key from the raw partition value would split
    them into separate one-row groups instead of one shared group, each re-seeding its own
    EMA recurrence. ``ema`` has no PyArrow production backend, so this derives the expected
    value from the pinned EMA recurrence (``adjust=False``): with ``span=3``
    (``alpha = 2/(span+1) = 0.5``) and the shared NaN partition's values ``[10.0, 20.0]`` in
    ``ord`` order, the second row's EMA is ``0.5 * 20.0 + 0.5 * 10.0 = 15.0``, not a
    freshly-seeded ``20.0``.
    """

    def test_nan_partition_rows_share_one_group_ema_continues_recurrence(self) -> None:
        import math

        import pyarrow as pa

        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set

        arrow_table = pa.table(
            {
                "grp": pa.array([float("nan"), float("nan"), 1.0, 2.0], type=pa.float64()),
                "ord": pa.array([1, 2, 3, 4], type=pa.int64()),
                "val": pa.array([10.0, 20.0, 100.0, 200.0], type=pa.float64()),
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
        fs = make_feature_set("val__ema_3", ["grp"], "ord")
        result = PythonDictEma.calculate_feature(data, fs)
        result_col = extract_column(result, "val__ema_3")

        assert result_col[1] == 15.0, (
            f"expected row 1's EMA to continue the shared NaN partition's recurrence "
            f"(0.5 * 20.0 + 0.5 * 10.0 = 15.0), got {result_col[1]!r} (PythonDict split "
            "the NaN rows into separate one-row groups, re-seeding at 20.0)"
        )
