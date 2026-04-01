"""Tests for PyArrowRank compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.rank.pyarrow_rank import (
    PyArrowRank,
)
from mloda.testing.feature_groups.data_operations.helpers import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.rank.rank import (
    RankTestBase,
)


class TestPyArrowRank(PyArrowTestMixin, RankTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowRank
