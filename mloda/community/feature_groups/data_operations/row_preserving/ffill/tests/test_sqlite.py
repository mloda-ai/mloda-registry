"""Tests for SqliteFfill compute implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.ffill.sqlite_ffill import (
    SqliteFfill,
)
from mloda.testing.feature_groups.data_operations.base import declares_nan_as_null
from mloda.testing.feature_groups.data_operations.mixins.frameworks import SqliteTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.ffill.ffill import (
    FfillTestBase,
)


class TestSqliteFfill(SqliteTestMixin, FfillTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteFfill

    @classmethod
    def pinned_divergences(cls) -> Mapping[str, Mapping[str, list[Any]]]:
        divergences: dict[str, Mapping[str, list[Any]]] = {
            # NaN and None partition keys merge, carrying a value across the reference partition boundary.
            "10-partition_keys-v__ffill": {"v__ffill": [1, 1, 1, 1]},
        }
        if declares_nan_as_null(cls.implementation_class()):
            # NaN and None tie in order_by (sqlite3 stores NaN as NULL), so ffill uses a different predecessor.
            divergences["3-order_tiers-v__ffill"] = {"v__ffill": [None, 20, None]}
            # A NaN source value is stored as NULL and filled instead of carried as a value.
            divergences["7-nan-v__ffill"] = {"v__ffill": [1.0, 1.0, 1.0]}
            divergences["8-nan-v__ffill"] = {"v__ffill": [None, None, None]}
        return divergences
