"""Tests for PandasRank compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

import pandas as pd
from mloda.user import Options

from mloda.community.feature_groups.data_operations.row_preserving.rank.pandas_rank import (
    PandasRank,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.pandas import PandasTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.rank.rank import (
    RankTestBase,
)


class TestPandasRank(CapabilityHookTestMixin, PandasTestMixin, RankTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasRank

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (
            ("value__percent_rank_ranked", Options()),
            ("value__dense_rank_ranked", Options()),
        )


class TestPandasRankNoneAndNanOrderBy:
    """Pins pandas' existing tie behavior: a None/NaN order_by run ties at one rank.

    ``pandas`` treats ``NaN`` and ``None`` as the same missing marker in a float64
    column, so both rank last together via ``na_option="bottom"``. Not covered by the
    shared ``RankTestBase`` fixture (backends genuinely disagree on this case; see
    ``ReferenceRank``'s divergent behavior in ``test_reference.py``), so this direct
    ``_compute_rank`` regression test guards against a future tie-run-splitting bug.
    """

    DATA: dict[str, list[Any]] = {"grp": [1, 1, 1, 1], "val": [None, float("nan"), None, 1.0]}

    def test_rank_ties_none_and_nan(self) -> None:
        df = pd.DataFrame(self.DATA)
        result = PandasRank._compute_rank(df, "r", ["grp"], "val", "rank")
        assert list(result["r"]) == [2, 2, 2, 1]

    def test_dense_rank_ties_none_and_nan(self) -> None:
        df = pd.DataFrame(self.DATA)
        result = PandasRank._compute_rank(df, "r", ["grp"], "val", "dense_rank")
        assert list(result["r"]) == [2, 2, 2, 1]

    def test_percent_rank_ties_none_and_nan(self) -> None:
        df = pd.DataFrame(self.DATA)
        result = PandasRank._compute_rank(df, "r", ["grp"], "val", "percent_rank")
        assert list(result["r"]) == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0.0])
