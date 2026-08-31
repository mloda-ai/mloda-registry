"""Tests for DuckdbRank compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.user import Options
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

from mloda.community.feature_groups.data_operations.row_preserving.rank.duckdb_rank import (
    DuckdbRank,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.rank.rank import (
    RankTestBase,
)


class TestDuckdbRank(CapabilityHookTestMixin, ReservedColumnsTestMixin, DuckdbTestMixin, RankTestBase):
    """Standard tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbRank

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (("value__percent_rank_ranked", Options()),)

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "value_int__row_number_ranked"

    @classmethod
    def reserved_columns_order_by(cls) -> str | None:
        return "value_int"


class TestDuckdbRankNoneAndNanOrderBy:
    """Pins DuckDB's existing (divergent) tie behavior: None and NaN rank apart.

    Unlike SQLite/pandas, DuckDB's ``RANK()``/``DENSE_RANK()``/``PERCENT_RANK()``
    treat SQL ``NULL`` and NaN as distinct sort keys, so a None/NaN mix does not tie
    into one run. Not covered by the shared ``RankTestBase`` fixture (backends
    genuinely disagree on this case; see ``ReferenceRank``'s divergent behavior in
    ``mloda/testing/.../rank/tests/test_reference.py``), so this direct
    ``_compute_rank`` regression test guards against a future change to this behavior.
    """

    def setup_method(self) -> None:
        self.conn = duckdb.connect()
        table = pa.table(
            {
                "grp": [1, 1, 1, 1],
                "val": pa.array([None, float("nan"), None, 1.0], type=pa.float64()),
            }
        )
        self.rel = DuckdbRelation.from_arrow(self.conn, table)

    def teardown_method(self) -> None:
        self.conn.close()

    def test_rank_ranks_none_and_nan_apart(self) -> None:
        """rank: None and NaN do NOT tie; NaN ranks 2, None rows rank 3."""
        result = DuckdbRank._compute_rank(self.rel, "r", ["grp"], "val", "rank")
        assert result.to_arrow_table().column("r").to_pylist() == [3, 2, 3, 1]

    def test_dense_rank_ranks_none_and_nan_apart(self) -> None:
        """dense_rank: None and NaN do NOT tie; NaN dense-ranks 2, None rows rank 3."""
        result = DuckdbRank._compute_rank(self.rel, "r", ["grp"], "val", "dense_rank")
        assert result.to_arrow_table().column("r").to_pylist() == [3, 2, 3, 1]

    def test_percent_rank_ranks_none_and_nan_apart(self) -> None:
        """percent_rank: None and NaN do NOT tie."""
        result = DuckdbRank._compute_rank(self.rel, "r", ["grp"], "val", "percent_rank")
        assert result.to_arrow_table().column("r").to_pylist() == pytest.approx([2 / 3, 1 / 3, 2 / 3, 0.0])
