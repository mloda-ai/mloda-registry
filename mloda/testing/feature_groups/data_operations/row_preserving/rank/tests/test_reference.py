"""Tests for ReferenceRank compute implementation."""

from __future__ import annotations

import pyarrow as pa
import pytest

from mloda.testing.feature_groups.data_operations.row_preserving.rank.reference import ReferenceRank


class TestReferenceRankNoneAndNanOrderBy:
    """A None/NaN order_by run must not hang rank / dense_rank / percent_rank, and ranks
    NaN apart from None (matching this reference's own PyArrow >= 25 engine).

    The tie-run loop only advanced ``pos`` inside ``while pos < n and sorted_rows[pos][1] ==
    sorted_rows[run_start][1]``; since ``nan == nan`` is ``False``, a NaN at ``run_start``
    never advanced ``pos`` and the outer loop spun forever. Same defect class as
    PythonDictRank's NaN order_by hang (``TestPythonDictRankNanOrderByHangs``), fixed the
    same way: unconditionally increment ``pos`` after ``run_start = pos`` and use a
    NaN-safe equality helper. Unlike PythonDict, this reference does not tie None and NaN
    into one run: it uses its own local tiering (real values, then NaN, then None)
    independent of PythonDict's helpers, so it stays a valid oracle rather than
    reproducing PythonDict's exact tie choice. See ``TestDuckdbRankNoneAndNanOrderBy`` /
    ``TestPolarsLazyRankNoneAndNanOrderBy`` for the two production backends that rank
    them apart the same way.
    """

    TABLE: pa.Table = pa.table(
        {
            "grp": [1, 1, 1, 1],
            "val": pa.array([None, float("nan"), None, 1.0], type=pa.float64()),
        }
    )

    def test_row_number_ranks_none_and_nan_apart(self) -> None:
        result = ReferenceRank._compute_rank(self.TABLE, "r", ["grp"], "val", "row_number")
        assert result.column("r").to_pylist() == [3, 2, 4, 1]

    def test_rank_ranks_none_and_nan_apart(self) -> None:
        result = ReferenceRank._compute_rank(self.TABLE, "r", ["grp"], "val", "rank")
        assert result.column("r").to_pylist() == [3, 2, 3, 1]

    def test_dense_rank_ranks_none_and_nan_apart(self) -> None:
        result = ReferenceRank._compute_rank(self.TABLE, "r", ["grp"], "val", "dense_rank")
        assert result.column("r").to_pylist() == [3, 2, 3, 1]

    def test_percent_rank_ranks_none_and_nan_apart(self) -> None:
        result = ReferenceRank._compute_rank(self.TABLE, "r", ["grp"], "val", "percent_rank")
        assert result.column("r").to_pylist() == pytest.approx([2 / 3, 1 / 3, 2 / 3, 0.0])
