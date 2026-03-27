"""Tests for SqliteStringOps compute implementation."""

from __future__ import annotations

import sqlite3
from typing import Any

import pyarrow as pa
import pytest

from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.string.sqlite_string import (
    SqliteStringOps,
)
from mloda.testing.feature_groups.data_operations.string import (
    StringTestBase,
)


class TestSqliteStringOps(StringTestBase):
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

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteStringOps

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        self.conn = sqlite3.connect(":memory:")
        return SqliteRelation.from_arrow(self.conn, arrow_table)

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        arrow_table = result.to_arrow_table()
        return list(arrow_table.column(column_name).to_pylist())

    def get_row_count(self, result: Any) -> int:
        return int(len(result))

    def get_expected_type(self) -> Any:
        return SqliteRelation

    def test_cross_framework_upper(self) -> None:
        """Skip: SQLite UPPER differs from PyArrow for non-ASCII characters."""
        pytest.skip("SQLite UPPER handles only ASCII; unicode results differ from PyArrow")


class TestSqliteReverseUnsupported:
    """SQLite does not support the reverse operation and should raise ValueError."""

    def test_reverse_raises_error(self) -> None:
        conn = sqlite3.connect(":memory:")
        arrow_table = pa.table({"name": ["Alice"], "value": [1]})
        relation = SqliteRelation.from_arrow(conn, arrow_table)

        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.core.abstract_plugins.components.options import Options
        from mloda.user import Feature

        feature = Feature("name__reverse", options=Options())
        fs = FeatureSet()
        fs.add(feature)

        with pytest.raises(ValueError, match="reverse"):
            SqliteStringOps.calculate_feature(relation, fs)

        conn.close()
