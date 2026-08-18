"""Tests for PythonDictRank compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.options import Options
from mloda.community.feature_groups.data_operations.row_preserving.rank.python_dict_rank import (
    PythonDictRank,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.rank.rank import (
    RankTestBase,
)


class TestPythonDictRank(CapabilityHookTestMixin, PythonDictTestMixin, RankTestBase):
    """All tests inherited from the base class.

    No overrides: PythonDict aims for the same full support level as pandas,
    DuckDB, polars-lazy, and SQLite (all rank types, including the parametric
    ntile_N / top_N / bottom_N families).
    """

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictRank

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (("value__percent_rank_ranked", Options()),)
