"""Tests for SqliteRank compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.user import Options

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

    @classmethod
    def reserved_columns_helper_names(cls) -> set[str]:
        return {"__mloda_rn0__", "__mloda_rn1__"}

    @classmethod
    def reserved_columns_extra_feature_names(cls) -> tuple[str, ...]:
        return ("value_int__top_3_ranked", "value_int__bottom_2_ranked")
