"""Tests for PythonDictFfill compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.ffill.python_dict_ffill import (
    PythonDictFfill,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.ffill.ffill import (
    FfillTestBase,
)


class TestPythonDictFfill(PythonDictTestMixin, FfillTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictFfill


class TestPythonDictFfillNanPartitionKeyGrouping:
    """A NaN partition-key value must group with itself, not split into singletons.

    Two rows that both carry ``float('nan')`` in a partition column compare unequal
    (``nan != nan``), so building the group key from the raw partition value would split
    them into separate one-row groups instead of one shared group, and forward-fill within
    a one-row group can never carry a value into it. This test asks PyArrow's own
    ``Table.group_by()`` directly which rows it considers one partition, and derives the
    expected forward-filled value from that by hand (see
    ffill/tests/test_pyarrow.py::TestPyArrowFfillNanPartitionKeyGrouping for the equivalent
    check against ``PyArrowFfill`` itself, which merges NaN partition keys too).
    """

    def test_nan_partition_rows_share_one_group_ffill_carries_forward(self) -> None:
        from mloda.testing.feature_groups.data_operations.helpers import (
            assert_pyarrow_groups_rows_together,
            extract_column,
            make_feature_set,
        )

        data: dict[str, list[Any]] = {"grp": [float("nan"), float("nan"), 1.0, 2.0], "ord": [1, 2, 3, 4]}
        data["val"] = [10.0, None, 100.0, None]
        fs = make_feature_set("val__ffill", ["grp"], "ord")
        result = calculate_with(PythonDictFfill, data, fs)
        result_col = extract_column(result, "val__ffill")

        # Row 1 (ord=2, val=None) is second (by ord) in the shared NaN partition PyArrow
        # reports above, so it must carry row 0's value (10.0) forward instead of staying None.
        assert result_col[1] == 10.0, (
            f"expected row 1's ffill to carry the shared NaN partition's value forward "
            f"(10.0, matching PyArrow's own group_by()), got {result_col[1]!r} (PythonDict "
            "split the NaN rows into separate one-row groups)"
        )

        assert_pyarrow_groups_rows_together(data["grp"])
