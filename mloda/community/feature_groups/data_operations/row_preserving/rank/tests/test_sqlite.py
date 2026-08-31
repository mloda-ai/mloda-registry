"""Tests for SqliteRank compute implementation."""

from __future__ import annotations

import sqlite3
from typing import Any

import pyarrow as pa
import pytest
from mloda.user import Options
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.rank.sqlite_rank import (
    SqliteRank,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.rank.rank import (
    RankTestBase,
)


class TestSqliteRank(CapabilityHookTestMixin, ReservedColumnsTestMixin, SqliteTestMixin, RankTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteRank

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (("value__percent_rank_ranked", Options()),)

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "value_int__row_number_ranked"

    @classmethod
    def reserved_columns_order_by(cls) -> str | None:
        return "value_int"


class TestSqliteRankNoneAndNanOrderBy:
    """Pins SQLite's existing tie behavior: a None/NaN order_by run ties at one rank.

    SQLite's ``RANK()``/``DENSE_RANK()``/``PERCENT_RANK()`` treat SQL ``NULL`` and NaN
    the same as one tied group. Not covered by the shared ``RankTestBase`` fixture
    (backends genuinely disagree on this case; see ``ReferenceRank``'s divergent
    behavior in ``mloda/testing/.../rank/tests/test_reference.py``), so this direct
    ``_compute_rank`` regression test guards against a future tie-run-splitting bug.
    """

    def setup_method(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        table = pa.table(
            {
                "grp": [1, 1, 1, 1],
                "val": pa.array([None, float("nan"), None, 1.0], type=pa.float64()),
            }
        )
        self.rel = SqliteRelation.from_arrow(self.conn, table)

    def teardown_method(self) -> None:
        self.conn.close()

    def test_rank_ties_none_and_nan(self) -> None:
        """rank: None and NaN tie at rank 2, the real value ranks 1."""
        result = SqliteRank._compute_rank(self.rel, "r", ["grp"], "val", "rank")
        assert result.to_arrow_table().column("r").to_pylist() == [2, 2, 2, 1]

    def test_dense_rank_ties_none_and_nan(self) -> None:
        """dense_rank: None and NaN tie at dense rank 2."""
        result = SqliteRank._compute_rank(self.rel, "r", ["grp"], "val", "dense_rank")
        assert result.to_arrow_table().column("r").to_pylist() == [2, 2, 2, 1]

    def test_percent_rank_ties_none_and_nan(self) -> None:
        """percent_rank: None and NaN tie at 1/3."""
        result = SqliteRank._compute_rank(self.rel, "r", ["grp"], "val", "percent_rank")
        assert result.to_arrow_table().column("r").to_pylist() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0.0])
