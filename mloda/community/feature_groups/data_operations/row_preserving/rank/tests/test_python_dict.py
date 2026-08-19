"""Tests for PythonDictRank compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.options import Options
from mloda.community.feature_groups.data_operations.row_preserving.rank.python_dict_rank import (
    PythonDictRank,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.rank.rank import (
    RankTestBase,
)


class TestPythonDictRank(CapabilityHookTestMixin, PythonDictTestMixin, RankTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictRank

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (("value__percent_rank_ranked", Options()),)


class TestPythonDictRankNanOrderByHangs:
    """A NaN ``order_by`` value must not hang ``rank`` / ``dense_rank`` / ``percent_rank``.

    ``PythonDictRank._apply_rank`` finds each run of equal ``order_by`` values by comparing
    adjacent sorted rows; plain ``==`` would compare a NaN value against itself on the first
    iteration of a run, and since ``nan == nan`` is ``False``, the run would never advance and
    the enclosing loop would spin forever. ``values_equal`` treats NaN as equal to NaN so the
    run terminates, and the sort key routes through ``nulls_last_sort_key`` so a NaN
    ``order_by`` value sorts deterministically last. With ``order_by`` values ``[nan, 1.0]``,
    each test asserts the non-NaN row (index 1) sorts first and the NaN row (index 0) sorts
    last, exactly as an ordinary null in ``order_by`` already does.
    """

    DATA: dict[str, list[Any]] = {"grp": [1, 1], "val": [float("nan"), 1.0]}

    def test_rank_hangs_on_nan_order_by(self) -> None:
        result = PythonDictRank._compute_rank(self.DATA, "r", [], "val", "rank")
        # Sorted (nulls-last) order: index 1 (1.0) first -> rank 1, index 0 (nan) last -> rank 2.
        assert result["r"] == [2, 1]

    def test_dense_rank_hangs_on_nan_order_by(self) -> None:
        result = PythonDictRank._compute_rank(self.DATA, "r", [], "val", "dense_rank")
        assert result["r"] == [2, 1]

    def test_percent_rank_hangs_on_nan_order_by(self) -> None:
        result = PythonDictRank._compute_rank(self.DATA, "r", [], "val", "percent_rank")
        # percent_rank = (rank - 1) / (n - 1): index 1 (rank 1) -> 0.0, index 0 (rank 2) -> 1.0.
        assert result["r"] == [1.0, 0.0]


class TestPythonDictRankNanPartitionKeyGrouping:
    """A NaN partition-key value must group with itself, not split into singletons.

    Two rows that both carry ``float('nan')`` in a partition column compare unequal
    (``nan != nan``), so building the group key from the raw partition value would split
    them into separate one-row groups instead of one shared group, and each row would be
    ranked alone within its own group instead of together. ``ReferenceRank`` builds its
    group key the exact same way and reproduces this identical bug, so it is NOT a valid
    oracle here. Instead, this test asks PyArrow's own ``Table.group_by()`` directly which
    rows it considers one partition, and derives the expected row numbering from that.
    """

    def test_nan_partition_rows_share_one_group_continues_numbering(self) -> None:
        import math

        import pyarrow as pa

        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set

        arrow_table = pa.table(
            {
                "grp": pa.array([float("nan"), float("nan"), 1.0, 2.0], type=pa.float64()),
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
        fs = make_feature_set("val__row_number_ranked", ["grp"], "val")
        result = PythonDictRank.calculate_feature(data, fs)
        result_col = extract_column(result, "val__row_number_ranked")

        # Row 1 (val=20.0) is second (by val) within the shared NaN partition PyArrow
        # reports above, so its row number must continue from row 0's instead of restarting.
        assert result_col[1] == 2, (
            f"expected row 1's row number to continue the shared NaN partition's numbering "
            f"(row 0 first, row 1 second -> 2), got {result_col[1]!r} (PythonDict split "
            "the NaN rows into separate one-row groups)"
        )


class TestPythonDictRankTopNTreatsNanAsNullNotMaximum:
    """DEFECT B: ``top_N`` must treat NaN as null-like (sorts last), not as the maximum.

    ``_apply_rank``'s ``top_`` branch splits ``sorted_rows`` on ``val is not None`` only, so a
    NaN value (which already sorted last via ``nulls_last_sort_key``) lands in ``non_null``,
    and the ``[::-1]`` reversal used to build the DESC order then puts it FIRST, marking the
    NaN row as the top instead of the actual maximum real value.
    """

    DATA: dict[str, list[Any]] = {"grp": [1, 1], "val": [float("nan"), 1.0]}

    def test_top_1_picks_the_real_value_not_nan(self) -> None:
        result = PythonDictRank._compute_rank(self.DATA, "r", [], "val", "top_1")
        assert result["r"] == [False, True], (
            f"expected the real value (index 1) to be top_1, not the NaN (index 0): {result['r']!r}"
        )


class TestPythonDictRankMixedNoneAndNanTieRunMatchesPyarrowOracle:
    """DEFECT C: a run of mixed None/NaN order_by values must all tie at one rank.

    ``nulls_last_sort_key`` maps both ``None`` and NaN to the same sort tier ``(1, 0)``, so a
    NaN row can sort BETWEEN two None rows within that tier. ``_apply_rank``'s run detection
    then compares adjacent sorted rows with ``values_equal``, which is False for
    ``values_equal(None, nan)``, so the tie run splits at that boundary instead of treating
    all null-like values as one tied group. ``pc.rank`` (``null_placement="at_end"``) is the
    live PyArrow oracle for ``rank``/``dense_rank``; ``percent_rank`` has no PyArrow kernel,
    so it is derived from that same oracle rank via the standard
    ``(rank - 1) / (n - 1)`` formula.
    """

    DATA: dict[str, list[Any]] = {"grp": [1, 1, 1, 1], "val": [None, float("nan"), None, 1.0]}

    @classmethod
    def _pyarrow_rank_oracle(cls, tiebreaker: str) -> list[int]:
        import pyarrow as pa
        import pyarrow.compute as pc

        arr = pa.array(cls.DATA["val"], type=pa.float64())
        ranked = pc.rank(arr, sort_keys="ascending", null_placement="at_end", tiebreaker=tiebreaker)
        return list(ranked.to_pylist())

    def test_rank_matches_pyarrow_oracle(self) -> None:
        oracle = self._pyarrow_rank_oracle("min")
        assert oracle == [2, 2, 2, 1], f"sanity check on the oracle itself failed: {oracle!r}"

        result = PythonDictRank._compute_rank(self.DATA, "r", [], "val", "rank")
        assert result["r"] == oracle, f"{result['r']!r} != PyArrow oracle {oracle!r}"

    def test_dense_rank_matches_pyarrow_oracle(self) -> None:
        oracle = self._pyarrow_rank_oracle("dense")
        assert oracle == [2, 2, 2, 1], f"sanity check on the oracle itself failed: {oracle!r}"

        result = PythonDictRank._compute_rank(self.DATA, "r", [], "val", "dense_rank")
        assert result["r"] == oracle, f"{result['r']!r} != PyArrow oracle {oracle!r}"

    def test_percent_rank_matches_pyarrow_derived_oracle(self) -> None:
        import pytest

        rank_oracle = self._pyarrow_rank_oracle("min")
        n = len(rank_oracle)
        percent_oracle = [(r - 1) / (n - 1) for r in rank_oracle]
        assert percent_oracle == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0.0]), (
            f"sanity check on the oracle itself failed: {percent_oracle!r}"
        )

        result = PythonDictRank._compute_rank(self.DATA, "r", [], "val", "percent_rank")
        assert result["r"] == pytest.approx(percent_oracle), f"{result['r']!r} != oracle {percent_oracle!r}"


class TestPythonDictRankNonFloatNanHangs:
    """DEFECT E: a non-``float`` NaN-like value must not hang or crash rank's run detection.

    ``is_nan`` used to be ``isinstance(value, float) and math.isnan(value)``, so a
    ``Decimal('NaN')`` or ``numpy.float32('nan')`` was invisible to it. With a single-row
    column that made the run-detection loop spin forever (``pos`` never advanced). With two
    or more rows, once the loop-progress fix alone landed, the unrecognized NaN instead
    reached ``nulls_last_sort_key`` as a "real" value and got compared with ``<`` against the
    other row during ``rows.sort()``, raising (e.g. ``decimal.InvalidOperation`` for
    ``Decimal('NaN') < Decimal('1')``). Both rows are required to exercise that comparison;
    a single-element column never calls ``sort``'s comparator at all.
    """

    def test_decimal_nan_sorts_nulls_last_and_ranks_correctly(self) -> None:
        from decimal import Decimal

        data: dict[str, list[Any]] = {"v": [Decimal("NaN"), Decimal("1")]}
        result = PythonDictRank._compute_rank(data, "r", [], "v", "rank")
        # Nulls-last: the real value (index 1) ranks 1, the NaN (index 0) ranks 2.
        assert result["r"] == [2, 1]

    def test_numpy_float32_nan_sorts_nulls_last_and_ranks_correctly(self) -> None:
        import pytest

        np = pytest.importorskip("numpy")

        data: dict[str, list[Any]] = {"v": [np.float32("nan"), np.float32(1.0)]}
        result = PythonDictRank._compute_rank(data, "r", [], "v", "rank")
        assert result["r"] == [2, 1]
