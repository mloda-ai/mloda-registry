"""Tests for PandasRank compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.core.abstract_plugins.components.options import Options
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
