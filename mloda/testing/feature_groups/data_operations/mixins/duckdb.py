"""DuckDB framework test mixin."""

from __future__ import annotations

from typing import Any

import pyarrow as pa


class DuckdbTestMixin:
    """Mixin implementing adapter methods for DuckDB.

    Requires ``duckdb`` to be importable. Concrete test modules should guard
    with ``duckdb = pytest.importorskip("duckdb")``.

    Creates ``self.conn`` in ``setup_method`` before delegating to the parent.
    """

    def setup_method(self) -> None:
        import duckdb

        self.conn = duckdb.connect()
        super().setup_method()  # type: ignore[misc]

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        return DuckdbRelation.from_arrow(self.conn, arrow_table)

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return list(result.to_arrow_table().column(column_name).to_pylist())

    def get_row_count(self, result: Any) -> int:
        return int(result.to_arrow_table().num_rows)

    def get_expected_type(self) -> Any:
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        return DuckdbRelation
