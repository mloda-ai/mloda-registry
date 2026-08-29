"""Tests for SqliteStringOps compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.community.feature_groups.data_operations.string.sqlite_string import (
    SqliteStringOps,
)
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin
from mloda.testing.feature_groups.data_operations.string.string import (
    StringTestBase,
)


class TestSqliteStringOps(ReservedColumnsTestMixin, SqliteTestMixin, StringTestBase):
    """All tests inherited from the base class.

    SQLite supports only 'trim' and 'length' natively in a way that matches
    the PyArrow reference. 'upper'/'lower' are ASCII-only in SQLite and
    'reverse' has no native SQLite function, so all three are refused at
    match time and resolved by another framework.
    """

    @classmethod
    def supported_ops(cls) -> set[str]:
        return {"trim", "length"}

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteStringOps

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "name__trim"

    @classmethod
    def reserved_columns_partition_by(cls) -> list[str] | None:
        return None


class TestSqliteUnsupportedOps:
    """SQLite refuses upper/lower/reverse at match time; the resolver falls
    back to another framework rather than silently producing ASCII-only output."""

    @pytest.mark.parametrize("operation", ["upper", "lower", "reverse"])
    def test_unsupported_op_does_not_match(self, operation: str) -> None:
        from mloda.user import Options

        options = Options()
        result = SqliteStringOps.match_feature_group_criteria(f"name__{operation}", options, None)
        assert result is False

    def test_supported_ops_still_match(self) -> None:
        from mloda.user import Options

        options = Options()
        for op in ("trim", "length"):
            result = SqliteStringOps.match_feature_group_criteria(f"name__{op}", options, None)
            assert result is True, f"Expected name__{op} to match SqliteStringOps"
