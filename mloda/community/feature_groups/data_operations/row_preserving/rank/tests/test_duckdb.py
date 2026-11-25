"""Tests for DuckdbRank compute implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.user import Options

from mloda.community.feature_groups.data_operations.row_preserving.rank.duckdb_rank import (
    DuckdbRank,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.frameworks import DuckdbTestMixin
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
    def pinned_divergences(cls) -> Mapping[str, Mapping[str, list[Any]]]:
        return {
            **super().pinned_divergences(),
            # NaN sorts above every real value in descending order, so top_2 selects NaN and 3.0 instead of 3.0 and 2.0.
            "13-top_bottom-o__top_2_ranked": {"o__top_2_ranked": [False, True, True, False, False]},
        }

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
