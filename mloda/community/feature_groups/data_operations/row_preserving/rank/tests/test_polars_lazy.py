"""Tests for PolarsLazyRank compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("polars")

import polars as pl
from mloda.user import Options

from mloda.community.feature_groups.data_operations.row_preserving.rank.polars_lazy_rank import (
    PolarsLazyRank,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.polars_lazy import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.rank.rank import (
    RankTestBase,
)


class TestPolarsLazyRank(CapabilityHookTestMixin, ReservedColumnsTestMixin, PolarsLazyTestMixin, RankTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyRank

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (("value__percent_rank_ranked", Options()),)

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "value_int__row_number_ranked"

    @classmethod
    def reserved_columns_order_by(cls) -> str | None:
        return "value_int"


class TestPolarsLazyRankNoneAndNanOrderBy:
    """Pins Polars' existing (divergent) tie behavior: None and NaN rank apart.

    Unlike SQLite/pandas, Polars' ``rank()`` treats a null and a NaN value as
    distinct sort keys, so a None/NaN mix does not tie into one run. Not covered by
    the shared ``RankTestBase`` fixture (backends genuinely disagree on this case;
    see ``ReferenceRank``'s divergent behavior in ``test_reference.py``), so this
    direct ``_compute_rank`` regression test guards against a future change to this
    behavior.
    """

    DATA: dict[str, list[Any]] = {"grp": [1, 1, 1, 1], "val": [None, float("nan"), None, 1.0]}

    def test_rank_ranks_none_and_nan_apart(self) -> None:
        lf = pl.LazyFrame(self.DATA)
        result = PolarsLazyRank._compute_rank(lf, "r", ["grp"], "val", "rank")
        assert result.collect()["r"].to_list() == [3, 2, 3, 1]

    def test_dense_rank_ranks_none_and_nan_apart(self) -> None:
        lf = pl.LazyFrame(self.DATA)
        result = PolarsLazyRank._compute_rank(lf, "r", ["grp"], "val", "dense_rank")
        assert result.collect()["r"].to_list() == [3, 2, 3, 1]

    def test_percent_rank_ranks_none_and_nan_apart(self) -> None:
        lf = pl.LazyFrame(self.DATA)
        result = PolarsLazyRank._compute_rank(lf, "r", ["grp"], "val", "percent_rank")
        assert result.collect()["r"].to_list() == pytest.approx([2 / 3, 1 / 3, 2 / 3, 0.0])
