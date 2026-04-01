"""Tests for SqliteStringOps compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.community.feature_groups.data_operations.string.sqlite_string import (
    SqliteStringOps,
)
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin
from mloda.testing.feature_groups.data_operations.string import (
    StringTestBase,
)


class TestSqliteStringOps(SqliteTestMixin, StringTestBase):
    """All tests inherited from the base class.

    SQLite does not support the 'reverse' operation natively,
    so supported_ops excludes it. SQLite's UPPER/LOWER only
    handle ASCII characters, so unicode accented characters
    are not transformed.
    """

    @classmethod
    def supported_ops(cls) -> set[str]:
        return {"upper", "lower", "trim", "length"}

    @classmethod
    def expected_upper(cls) -> list[Any]:
        # SQLite UPPER only handles ASCII; accent e (\u00e9) stays lowercase
        return ["ALICE", "BOB", None, "", " EVE ", "FRANK", "GRACE", "ALICE", "  ", "BOB", "H\u00e9LLO", None]

    # Note: expected_lower is NOT overridden because the test data's only
    # non-ASCII character (accent e in "hello") is already lowercase.
    # SQLite's ASCII-only LOWER happens to produce the correct result
    # by coincidence. If test data included uppercase accented characters,
    # this override would be required.

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteStringOps

    def test_cross_framework_upper(self) -> None:
        """Skip: SQLite UPPER differs from PyArrow for non-ASCII characters."""
        pytest.skip("SQLite UPPER handles only ASCII; unicode results differ from PyArrow")


class TestSqliteReverseUnsupported:
    """SQLite does not support 'reverse', so it should not match at all."""

    def test_reverse_does_not_match(self) -> None:
        from mloda.core.abstract_plugins.components.options import Options

        options = Options()
        result = SqliteStringOps.match_feature_group_criteria("name__reverse", options, None)
        assert result is False

    def test_supported_ops_still_match(self) -> None:
        from mloda.core.abstract_plugins.components.options import Options

        options = Options()
        for op in ("upper", "lower", "trim", "length"):
            result = SqliteStringOps.match_feature_group_criteria(f"name__{op}", options, None)
            assert result is True, f"Expected name__{op} to match SqliteStringOps"
