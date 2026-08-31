"""Tests for ReferenceRank compute implementation."""

from __future__ import annotations

import pyarrow as pa
import pytest

from mloda.testing.feature_groups.data_operations.row_preserving.rank.reference import ReferenceRank


class TestReferenceRankNoneAndNanOrderBy:
    """A None/NaN order_by run must not hang rank / dense_rank / percent_rank.

    ``ReferenceRank._compute_rank``'s tie-run detection advances ``pos`` only inside the
    inner ``while pos < n and sorted_rows[pos][1] == sorted_rows[run_start][1]``, so when
    ``sorted_rows[run_start][1]`` is NaN, ``nan == nan`` is ``False`` even at ``pos ==
    run_start`` and ``pos`` never advances: the outer loop spins forever. This is the same
    defect class as ``PythonDictRank``'s DEFECT E (see ``TestPythonDictRankNonFloatNanHangs``
    in ``test_python_dict.py``), fixed there by unconditionally incrementing ``pos`` right
    after ``run_start = pos`` and using a NaN-safe equality helper. These tests pin the same
    fix here, and that None and NaN tie into one run (matching
    ``TestPythonDictRankMixedNoneAndNanTieRun``).
    """

    TABLE: pa.Table = pa.table(
        {
            "grp": [1, 1, 1, 1],
            "val": pa.array([None, float("nan"), None, 1.0], type=pa.float64()),
        }
    )

    def test_rank_ties_none_and_nan(self) -> None:
        """rank: the None/NaN run ties at rank 2; must complete, not hang."""
        result = ReferenceRank._compute_rank(self.TABLE, "r", [], "val", "rank")
        assert result.column("r").to_pylist() == [2, 2, 2, 1]

    def test_dense_rank_ties_none_and_nan(self) -> None:
        """dense_rank: the None/NaN run ties at dense rank 2; must complete, not hang."""
        result = ReferenceRank._compute_rank(self.TABLE, "r", [], "val", "dense_rank")
        assert result.column("r").to_pylist() == [2, 2, 2, 1]

    def test_percent_rank_ties_none_and_nan(self) -> None:
        """percent_rank: the None/NaN run ties at 1/3; must complete, not hang."""
        result = ReferenceRank._compute_rank(self.TABLE, "r", [], "val", "percent_rank")
        assert result.column("r").to_pylist() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0.0])
