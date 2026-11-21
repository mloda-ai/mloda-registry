"""Tests for PythonDictOffset compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.user import DataType

from mloda.community.feature_groups.data_operations.row_preserving.offset.python_dict_offset import (
    PythonDictOffset,
)
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.offset.offset import OffsetTestBase
from mloda.testing.typed_data import TypedTestData


class TestPythonDictOffset(PythonDictTestMixin, OffsetTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictOffset


class TestPythonDictOffsetNanPartitionKeyGrouping:
    """A NaN partition-key value must group with itself, not split into singletons.

    Two rows that both carry ``float('nan')`` in a partition column compare unequal
    (``nan != nan``), so building the group key from the raw partition value would split
    them into separate one-row groups instead of one shared group, and ``lag_1`` on a
    one-row group is always ``None``. ``offset`` has no PyArrow production backend;
    ``reference_offset`` (the cross-framework test reference) builds its group key the exact
    same way and reproduces this identical bug, so it is NOT a valid oracle here. Instead,
    this test asks PyArrow's own ``Table.group_by()`` directly which rows it considers one
    partition, and derives the expected ``lag_1`` value from that by hand.
    """

    def test_nan_partition_rows_share_one_group_lag_carries_forward(self) -> None:
        from mloda.testing.feature_groups.data_operations.helpers import (
            assert_pyarrow_groups_rows_together,
            calculate_with,
            extract_column,
            make_feature_set,
        )

        typed = TypedTestData(
            {
                "grp": (DataType.DOUBLE, [float("nan"), float("nan"), 1.0, 2.0]),
                "ord": (DataType.INT64, [1, 2, 3, 4]),
                "val": (DataType.DOUBLE, [10.0, 20.0, 100.0, 200.0]),
            }
        )
        data = typed.to_python_dict()
        fs = make_feature_set("val__lag_1_offset", ["grp"], "ord")
        result = calculate_with(PythonDictOffset, data, fs)
        result_col = extract_column(result, "val__lag_1_offset")

        # Row 1 (ord=2, val=20.0) is second (by ord) in the shared NaN partition PyArrow
        # reports above, so its lag_1 must carry row 0's value (10.0) instead of None.
        assert result_col[1] == 10.0, (
            f"expected row 1's lag_1 to carry the shared NaN partition's first value "
            f"(10.0, matching PyArrow's own group_by()), got {result_col[1]!r} (PythonDict "
            "split the NaN rows into separate one-row groups)"
        )

        assert_pyarrow_groups_rows_together(data["grp"])
