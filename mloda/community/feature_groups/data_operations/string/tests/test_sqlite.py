"""Tests for SqliteStringOps compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from mloda.community.feature_groups.data_operations.string.sqlite_string import (
    SqliteStringOps,
)
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin
from mloda.testing.feature_groups.data_operations.string.string import (
    StringTestBase,
)


class TestSqliteStringOps(SqliteTestMixin, StringTestBase):
    """All tests inherited from the base class.

    SQLite does not support the 'reverse' operation natively,
    so supported_ops excludes it.
    """

    @classmethod
    def supported_ops(cls) -> set[str]:
        return {"upper", "lower", "trim", "length"}

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteStringOps

    def test_unicode_lower_regression(self) -> None:
        """Regression test: LOWER must lowercase non-ASCII uppercase characters.

        SQLite's native LOWER() is ASCII-only and leaves characters like
        'É', 'Ö', 'Ω' unchanged. PyArrow / Python str.lower() handle the
        full unicode range. This test locks in the correct unicode-aware
        behavior so a regression to ASCII-only LOWER is caught.
        """
        table = pa.table(
            {
                "name": pa.array(["H\u00c9LLO", "W\u00d6RLD", "\u03a9-OMEGA"], type=pa.string()),
            }
        )
        data = self.create_test_data(table)
        fs = make_feature_set("name__lower")
        result = self.implementation_class().calculate_feature(data, fs)

        result_col = self.extract_column(result, "name__lower")
        assert result_col == ["h\u00e9llo", "w\u00f6rld", "\u03c9-omega"]


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
