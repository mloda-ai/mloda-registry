"""Tests for SqliteBinning compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.binning.sqlite_binning import (
    SqliteBinning,
)
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.binning.binning import (
    BinningTestBase,
)


class TestSqliteBinning(ReservedColumnsTestMixin, SqliteTestMixin, BinningTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteBinning

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "value_int__bin_3"

    @classmethod
    def reserved_columns_partition_by(cls) -> list[str] | None:
        return None

    @classmethod
    def reserved_columns_helper_names(cls) -> set[str]:
        return {"__mloda_rn0__", "__mloda_rn1__", "__mloda_rn2__"}

    @classmethod
    def reserved_columns_extra_feature_names(cls) -> tuple[str, ...]:
        return ("value_int__qbin_3",)
