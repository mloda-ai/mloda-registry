"""Per-framework test mixins for data-operations suites, built on core's framework adapters."""

from __future__ import annotations

import atexit
import functools
import sqlite3
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

from mloda.testing.feature_groups.data_operations.helpers import arrow_table_of
from mloda.testing.framework_adapter import (
    DuckDBAdapter,
    FrameworkAdapter,
    PandasAdapter,
    PolarsLazyAdapter,
    PyArrowAdapter,
    PythonDictAdapter,
    SqliteAdapter,
)
from mloda.testing.typed_data import TypedTestData

if TYPE_CHECKING:
    import pyarrow as pa


def pin_connection_utc_via_core(con: Any) -> None:
    """Pin ``con`` to UTC via the core chokepoint, as production does (mloda >= 0.9.0)."""
    import uuid

    from mloda.user import ParallelizationMode
    from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework

    framework = DuckDBFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset({uuid.uuid4()}))
    framework.set_framework_connection_object(con)


@functools.cache
def _shared_database() -> Any:
    import duckdb

    conn = duckdb.connect()
    atexit.register(conn.close)
    return conn


def duckdb_test_connection() -> Any:
    """Return a UTC-pinned cursor on the per-process shared in-memory database."""
    cursor = _shared_database().cursor()
    pin_connection_utc_via_core(cursor)
    return cursor


def _nullable_int_types_mapper(arrow_type: pa.DataType) -> Any:
    """Map Arrow integer types to pandas nullable integer dtypes, others to None."""
    # Not core's PandasAdapter mapping: core applies nullable dtypes only to columns holding nulls.
    import pandas as pd
    import pyarrow as pa

    mapping = {
        pa.int8(): pd.Int8Dtype(),
        pa.int16(): pd.Int16Dtype(),
        pa.int32(): pd.Int32Dtype(),
        pa.int64(): pd.Int64Dtype(),
        pa.uint8(): pd.UInt8Dtype(),
        pa.uint16(): pd.UInt16Dtype(),
        pa.uint32(): pd.UInt32Dtype(),
        pa.uint64(): pd.UInt64Dtype(),
    }
    return mapping.get(arrow_type)


def _maybe_parse_iso_datetime(value: Any) -> Any:
    """Parse an ISO-8601 string (SQLite stores timestamps as TEXT) into a ``datetime``, else pass through."""
    if not isinstance(value, str):
        return value
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return value


class _ProductionPandasAdapter(PandasAdapter):
    """Frames as production builds them: ``to_pandas()`` widens nullable ints to float64 and bools to object."""

    def from_arrow(self, table: pa.Table) -> Any:
        return table.to_pandas()


class _NullableIntPandasAdapter(PandasAdapter):
    """Integer columns held as nullable Int dtypes, keeping nulls exact."""

    def from_arrow(self, table: pa.Table) -> Any:
        return table.to_pandas(types_mapper=_nullable_int_types_mapper)


class _PlainSqliteAdapter(SqliteAdapter):
    """A bare in-memory connection, without the REGEXP and DQS setup the core adapter applies."""

    def open(self) -> None:
        self._connection = sqlite3.connect(":memory:")


class _SqliteViewAdapter(_PlainSqliteAdapter):
    """The view stands in for an earlier feature's output (a window that reorders and restores input order)."""

    def from_arrow(self, table: pa.Table) -> Any:
        from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name
        from mloda_plugins.compute_framework.base_implementations.sql.sql_window import OrderBy

        rel = super().from_arrow(table)
        cols = list(table.column_names)
        if not cols:
            return rel
        helper = pick_helper_column_name(taken=set(cols))
        return rel.with_row_number(helper, order_by=[OrderBy(cols[0], descending=True)]).select(*cols)


class _SqliteComputedColumnAdapter(_PlainSqliteAdapter):
    """Every input column is a view expression, so its declared type is empty."""

    def from_arrow(self, table: pa.Table) -> Any:
        from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

        rel = super().from_arrow(table)
        cols = list(table.column_names)
        if not cols:
            return rel
        return rel.select(_raw_sql=", ".join(f"+{quote_ident(c)} AS {quote_ident(c)}" for c in cols))


class _SharedDuckDBAdapter(DuckDBAdapter):
    """A UTC-pinned cursor on the per-process shared database."""

    def open(self) -> None:
        self._connection = duckdb_test_connection()


class AdapterTestMixin:
    """Implements the ``DataOpsTestBase`` adapter methods through ``adapter_class``; ``arrow_types`` overrides make a fixture Arrow-only."""

    adapter_class: ClassVar[type[FrameworkAdapter]]

    def setup_method(self) -> None:
        self.adapter = self.adapter_class()
        self.adapter.open()
        self.conn = self.adapter.connection
        try:
            super().setup_method()  # type: ignore[misc]
        except BaseException:
            self.adapter.close()
            raise

    def teardown_method(self) -> None:
        self.adapter.close()
        super().teardown_method()  # type: ignore[misc]

    def create_test_data(self, data: TypedTestData, arrow_types: Mapping[str, pa.DataType] | None = None) -> Any:
        if arrow_types:
            return self.adapter.from_arrow(arrow_table_of(data, arrow_types))
        return self.adapter.from_typed(data)

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return self.adapter.column_values(result, column_name)

    def get_row_count(self, result: Any) -> int:
        return self.adapter.row_count(result)

    def compute_framework_class(self) -> Any:
        return self.adapter_class().compute_framework()


class PythonDictTestMixin(AdapterTestMixin):
    """Adapter methods for PythonDict; the framework ships with mloda core, so no importorskip guard."""

    adapter_class = PythonDictAdapter

    def get_expected_type(self) -> Any:
        return dict


class PyArrowTestMixin(AdapterTestMixin):
    """Adapter methods for PyArrow."""

    adapter_class = PyArrowAdapter

    def get_expected_type(self) -> Any:
        import pyarrow as pa

        return pa.Table


class PolarsLazyTestMixin(AdapterTestMixin):
    """Adapter methods for Polars LazyFrame; guard concrete modules with ``pytest.importorskip("polars")``."""

    adapter_class = PolarsLazyAdapter

    def get_expected_type(self) -> Any:
        import polars as pl

        return pl.LazyFrame


class PandasTestMixin(AdapterTestMixin):
    """Adapter methods for Pandas; guard concrete modules with ``pytest.importorskip("pandas")``."""

    adapter_class: ClassVar[type[PandasAdapter]] = _ProductionPandasAdapter

    def get_expected_type(self) -> Any:
        import pandas as pd

        return pd.DataFrame


class PandasNullableIntTestMixin(PandasTestMixin):
    """Pandas mixin whose fixtures hold integer columns as nullable Int dtypes, keeping nulls exact."""

    adapter_class = _NullableIntPandasAdapter


class DuckdbTestMixin(AdapterTestMixin):
    """Adapter methods for DuckDB; guard concrete modules with ``pytest.importorskip("duckdb")``.

    ``self.conn`` is a cursor on a shared database: non-temp tables, macros and global settings are database-wide, so
    tests needing them should open a private ``duckdb.connect()``.
    """

    adapter_class = _SharedDuckDBAdapter

    def get_expected_type(self) -> Any:
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        return DuckdbRelation


class SqliteTestMixin(AdapterTestMixin):
    """Adapter methods for SQLite; ``self.conn`` is created in ``setup_method``."""

    adapter_class = _PlainSqliteAdapter

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return [_maybe_parse_iso_datetime(v) for v in self.adapter.column_values(result, column_name)]

    def get_expected_type(self) -> Any:
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

        return SqliteRelation


class SqliteViewTestMixin(SqliteTestMixin):
    """Fixtures are a view that reorders and restores input order."""

    adapter_class = _SqliteViewAdapter


class SqliteComputedColumnTestMixin(SqliteTestMixin):
    """Fixtures whose every column is a view expression."""

    adapter_class = _SqliteComputedColumnAdapter
