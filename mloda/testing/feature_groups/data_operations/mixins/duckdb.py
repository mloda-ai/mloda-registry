"""DuckDB framework test mixin."""

from __future__ import annotations

from typing import Any

import pyarrow as pa


def pin_connection_utc_via_core(con: Any) -> None:
    """Pin a DuckDB connection's session timezone to UTC via the real core chokepoint.

    Routes ``con`` through ``DuckDBFramework.set_framework_connection_object``, which
    applies the same UTC-session guarantee that mloda's DuckDBFramework applies in
    production (mloda 0.9.0). Using this helper means tests rely on the core guarantee
    instead of the host default timezone (or a per-feature-group pin).

    Imports are done inside the function body to keep module import light and to match
    the lazy-import style used elsewhere in this file (where ``duckdb`` is imported
    inside methods).
    """
    import uuid

    from mloda.user import ParallelizationMode
    from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework

    framework = DuckDBFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset({uuid.uuid4()}))
    framework.set_framework_connection_object(con)


class DuckdbTestMixin:
    """Mixin implementing adapter methods for DuckDB.

    Requires ``duckdb`` to be importable. Concrete test modules should guard
    with ``duckdb = pytest.importorskip("duckdb")``.

    Creates ``self.conn`` in ``setup_method`` before delegating to the parent.
    """

    def setup_method(self) -> None:
        import duckdb

        self.conn = duckdb.connect()
        pin_connection_utc_via_core(self.conn)
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
