"""Tests for PythonDictPercentile compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.percentile.python_dict_percentile import (
    PythonDictPercentile,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.percentile.percentile import (
    PercentileTestBase,
)


class TestPythonDictPercentile(PythonDictTestMixin, PercentileTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictPercentile


class TestPythonDictNanPartitionKeyGrouping:
    """A NaN partition-key value must group with itself, not split into singletons.

    ``PythonDictPercentile._compute_percentile`` builds the group key as
    ``tuple(col[i] for col in partition_cols)`` (python_dict_percentile.py
    line 57). Two rows that both carry ``float('nan')`` in a partition
    column compare unequal (``nan != nan``), so the dict splits them into
    separate one-row groups instead of one shared group, and each row is
    broadcast only its own percentile (itself) rather than the group's.

    Verified empirically: PyArrow's own ``Table.group_by()`` merges all NaN
    keys of a column into a single group, so "one shared NaN group" is the
    correct target. ``ReferencePercentile`` drives its grouping through
    PyArrow's real ``group_by().aggregate()``, so it is a valid oracle here.
    """

    def test_nan_partition_keys_grouped_together_matches_pyarrow_oracle(self) -> None:
        import pyarrow as pa

        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set
        from mloda.testing.feature_groups.data_operations.row_preserving.percentile.reference import (
            ReferencePercentile,
        )

        arrow_table = pa.table(
            {
                "grp": pa.array([float("nan"), float("nan"), 1.0, 2.0], type=pa.float64()),
                "val": pa.array([10.0, 20.0, 100.0, 200.0], type=pa.float64()),
            }
        )
        data = {name: arrow_table.column(name).to_pylist() for name in arrow_table.column_names}
        fs = make_feature_set("val__p50_percentile", ["grp"])

        result = PythonDictPercentile.calculate_feature(data, fs)
        oracle = ReferencePercentile.calculate_feature(arrow_table, fs)

        result_col = extract_column(result, "val__p50_percentile")
        oracle_col = extract_column(oracle, "val__p50_percentile")

        assert oracle_col == [15.0, 15.0, 100.0, 200.0], f"sanity check on the oracle itself failed: {oracle_col!r}"
        assert result_col == oracle_col, (
            f"expected both NaN-keyed rows to broadcast the shared group's p50 (median of "
            f"10.0/20.0 = 15.0, matching the PyArrow oracle {oracle_col!r}), got PythonDict={result_col!r}"
        )


class TestPythonDictPercentileNanValueSkipped:
    """A NaN value in the source column must be excluded from the percentile, not sort into it.

    Building ``non_null`` as ``sorted(v for v in values if v is not None)`` keeps NaN
    (``v is not None`` is True for NaN); NaN's comparisons are never ``True``, so it can
    land anywhere in the "sorted" list, and PERCENTILE_CONT interpolation over that list
    then produces garbage. ``ReferencePercentile`` interpolates via PyArrow's own
    ``pyarrow.compute.quantile``, which skips NaN like a null (verified empirically), so
    it is a valid live oracle here, unlike the reference implementations for
    rank/offset/frame_aggregate, whose hand-rolled group reducers reproduce the identical
    NaN bug.
    """

    def test_percentile_of_skips_nan_directly(self) -> None:
        """Direct reproduction: ``_percentile_of`` must treat NaN like a missing value."""
        result = PythonDictPercentile._percentile_of([1.0, float("nan"), 3.0], 0.5)
        assert result == 2.0, f"expected the NaN value to be excluded (median of 1.0/3.0 = 2.0), got {result!r}"

    def test_nan_value_skipped_matches_pyarrow_oracle(self) -> None:
        import pyarrow as pa

        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set
        from mloda.testing.feature_groups.data_operations.row_preserving.percentile.reference import (
            ReferencePercentile,
        )

        arrow_table = pa.table(
            {
                "grp": pa.array(["A", "A", "A"], type=pa.string()),
                "val": pa.array([1.0, float("nan"), 3.0], type=pa.float64()),
            }
        )
        data = {name: arrow_table.column(name).to_pylist() for name in arrow_table.column_names}
        fs = make_feature_set("val__p50_percentile", ["grp"])

        result = PythonDictPercentile.calculate_feature(data, fs)
        oracle = ReferencePercentile.calculate_feature(arrow_table, fs)

        result_col = extract_column(result, "val__p50_percentile")
        oracle_col = extract_column(oracle, "val__p50_percentile")

        assert oracle_col == [2.0, 2.0, 2.0], f"sanity check on the oracle itself failed: {oracle_col!r}"
        assert result_col == oracle_col, (
            f"expected the NaN value to be excluded (median of 1.0/3.0 = 2.0, matching the "
            f"PyArrow oracle {oracle_col!r}), got PythonDict={result_col!r}"
        )
