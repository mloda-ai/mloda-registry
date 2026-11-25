"""Tests for PolarsLazyRank compute implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.user import Options

from mloda.community.feature_groups.data_operations.row_preserving.rank.polars_lazy_rank import (
    PolarsLazyRank,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PolarsLazyTestMixin
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
