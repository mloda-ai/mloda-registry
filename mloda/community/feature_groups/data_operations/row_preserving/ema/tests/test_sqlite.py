"""Tests for SqliteEma: EMA is REJECTED, not computed.

SQLite has no native exponentially weighted (EWM) aggregate, and a recursive /
Python emulation is forbidden by the CFW-backend rule. The SQLite backend must
reject EMA up-front with a clear ValueError. This file inherits only the single
rejection assertion from ``EmaRejectionTestBase`` -- none of the value tests.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.ema.sqlite_ema import (
    SqliteEma,
)
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.ema.ema import (
    EmaRejectionTestBase,
)


class TestSqliteEmaRejected(SqliteTestMixin, EmaRejectionTestBase):
    """EMA on SQLite raises a clear ValueError."""

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteEma
