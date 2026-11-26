"""Tests for SqliteRank compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from mloda.user import Options

from mloda.community.feature_groups.data_operations.row_preserving.rank.sqlite_rank import (
    SqliteRank,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.frameworks import SqliteTestMixin, SqliteViewTestMixin
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
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

    def test_bool_flag_column_is_reported_as_bool(self) -> None:
        """The relation's reported type of the top-n flag matches its arrow table."""
        fs = make_feature_set("value_int__top_3_ranked", ["region"], "value_int")
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        name = "value_int__top_3_ranked"
        reported = result.types[result.columns.index(name)]
        assert reported == pa.bool_()
        assert result.to_arrow_table().schema.field(name).type == reported


class TestSqliteRankViewInput(SqliteViewTestMixin, TestSqliteRank):
    """Runs the Rank suite on a view input."""
