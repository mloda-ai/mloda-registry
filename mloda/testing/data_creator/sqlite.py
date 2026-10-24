"""SQLite test data creator for data operations plugins."""

from __future__ import annotations

import sqlite3
from typing import Any

from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

from mloda.testing.data_creator.base import DataOperationsTestDataCreator
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator


class SqliteDataOpsTestDataCreator(DataOperationsTestDataCreator):
    compute_framework = SqliteFramework

    @classmethod
    def create(cls, connection: sqlite3.Connection | None = None) -> Any:
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

        table = PyArrowDataOpsTestDataCreator.create()
        return SqliteRelation.from_arrow(connection or sqlite3.connect(":memory:"), table)
