"""SQLite framework test mixin."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pyarrow as pa


def _maybe_parse_iso_datetime(value: Any) -> Any:
    """Convert an ISO-8601 timestamp string into a ``datetime`` if possible.

    SQLite stores timestamps as TEXT and surfaces them as ``str`` via
    ``to_arrow_table``. Cross-framework tests compare against ``datetime``
    objects, so the mixin normalizes string values that parse as ISO
    timestamps. Non-string values and non-timestamp strings are passed
    through unchanged.
    """
    if not isinstance(value, str):
        return value
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return value


class SqliteTestMixin:
    """Mixin implementing adapter methods for SQLite.

    Creates ``self.conn`` in ``setup_method`` before delegating to the parent.
    """

    def setup_method(self) -> None:
        import sqlite3

        self.conn = sqlite3.connect(":memory:")
        super().setup_method()  # type: ignore[misc]

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

        return SqliteRelation.from_arrow(self.conn, arrow_table)

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        # SQLite stores timestamps as TEXT. Parse ISO-8601 strings back into
        # ``datetime`` so cross-framework tests can compare against ``datetime``
        # references uniformly. The new SQLite result-type contract tests go
        # through ``to_arrow_table`` directly and still see ``pa.string()``.
        raw = result.to_arrow_table().column(column_name).to_pylist()
        return [_maybe_parse_iso_datetime(v) for v in raw]

    def get_row_count(self, result: Any) -> int:
        return len(result)

    def get_expected_type(self) -> Any:
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

        return SqliteRelation

    def compute_framework_class(self) -> Any:
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

        return SqliteFramework
