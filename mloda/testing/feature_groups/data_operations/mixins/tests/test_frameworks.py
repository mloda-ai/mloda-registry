"""Tests for ``mixins/frameworks.py``: the per-framework test mixins and the DuckDB test connection."""

from __future__ import annotations

import importlib
import importlib.util
import sqlite3
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import pyarrow as pa
import pytest
from mloda.user import DataType

from mloda.testing.data_creator.base import DataOperationsTestDataCreator
from mloda.testing.feature_groups.data_operations import helpers
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.mixins.frameworks import DuckdbTestMixin, duckdb_test_connection
from mloda.testing.typed_data import TypedTestData


class _StubTeardownBase(DataOpsTestBase):
    """Minimal concrete base used to check the base teardown."""

    @classmethod
    def implementation_class(cls) -> Any:
        return object

    def create_test_data(self, data: TypedTestData, arrow_types: Mapping[str, pa.DataType] | None = None) -> Any:
        return data

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return []

    def get_row_count(self, result: Any) -> int:
        return 0

    def get_expected_type(self) -> Any:
        return float


# -- Per-framework mixin module -------------------------------------------------

_OLD_MIXIN_MODULES = ["pandas", "polars_lazy", "pyarrow", "python_dict", "duckdb", "sqlite"]


@pytest.mark.parametrize("name", _OLD_MIXIN_MODULES)
def test_per_framework_mixin_module_is_gone(name: str) -> None:
    module = f"mloda.testing.feature_groups.data_operations.mixins.{name}"
    assert importlib.util.find_spec(module) is None


# (mixin class, core adapter class, backend module to importorskip or None)
_FRAMEWORKS = [
    ("PythonDictTestMixin", "PythonDictAdapter", None),
    ("PyArrowTestMixin", "PyArrowAdapter", None),
    ("PolarsLazyTestMixin", "PolarsLazyAdapter", "polars"),
    ("PandasTestMixin", "PandasAdapter", "pandas"),
    ("PandasNullableIntTestMixin", "PandasAdapter", "pandas"),
    ("DuckdbTestMixin", "DuckDBAdapter", "duckdb"),
    ("SqliteTestMixin", "SqliteAdapter", None),
    ("SqliteViewTestMixin", "SqliteAdapter", None),
    ("SqliteComputedColumnTestMixin", "SqliteAdapter", None),
]
_FRAMEWORK_IDS = [name for name, _, _ in _FRAMEWORKS]
_ROUND_TRIP = [entry for entry in _FRAMEWORKS if entry[0] != "SqliteComputedColumnTestMixin"]


def _frameworks_module() -> Any:
    return importlib.import_module("mloda.testing.feature_groups.data_operations.mixins.frameworks")


def _mixin(name: str, backend: str | None) -> Any:
    if backend is not None:
        pytest.importorskip(backend)
    return getattr(_frameworks_module(), name)


def _probe(mixin: Any) -> Any:
    """An instance mixing a framework mixin with ``DataOpsTestBase``."""
    probe_class = type(
        "_FrameworkProbe", (mixin, DataOpsTestBase), {"implementation_class": classmethod(lambda cls: object)}
    )
    return probe_class()


@pytest.mark.parametrize(("mixin_name", "adapter_name", "backend"), _FRAMEWORKS, ids=_FRAMEWORK_IDS)
def test_framework_mixin_declares_the_matching_core_adapter(mixin_name: str, adapter_name: str, backend: Any) -> None:
    from mloda.testing import framework_adapter

    mixin = getattr(_frameworks_module(), mixin_name)
    assert issubclass(mixin.adapter_class, getattr(framework_adapter, adapter_name))


@pytest.mark.parametrize(("mixin_name", "adapter_name", "backend"), _ROUND_TRIP, ids=[m for m, _, _ in _ROUND_TRIP])
def test_framework_mixin_round_trips_the_canonical_typed_data(mixin_name: str, adapter_name: str, backend: Any) -> None:
    probe = _probe(_mixin(mixin_name, backend))
    typed = DataOperationsTestDataCreator.typed_data()
    probe.setup_method()
    try:
        frame = probe.create_test_data(typed)
        assert probe.get_row_count(frame) == 12
        for column, values in typed.to_python_dict().items():
            assert probe.extract_column(frame, column) == values, column
    finally:
        probe.teardown_method()


@pytest.mark.parametrize(
    ("mixin_name", "expected_dtype"), [("PandasTestMixin", "float64"), ("PandasNullableIntTestMixin", "Int64")]
)
def test_pandas_mixin_builds_value_int_with_the_expected_dtype(mixin_name: str, expected_dtype: str) -> None:
    probe = _probe(_mixin(mixin_name, "pandas"))
    probe.setup_method()
    try:
        assert str(probe.test_data["value_int"].dtype) == expected_dtype
    finally:
        probe.teardown_method()


def test_sqlite_extract_column_returns_datetimes_for_timestamps() -> None:
    probe = _probe(_mixin("SqliteTestMixin", None))
    probe.setup_method()
    try:
        values = probe.extract_column(probe.test_data, "timestamp")
        assert values and all(isinstance(v, datetime) for v in values if v is not None)
    finally:
        probe.teardown_method()


def test_arrow_types_override_routes_python_dict_through_arrow() -> None:
    probe = _probe(_mixin("PythonDictTestMixin", None))
    data = TypedTestData({"v": (DataType.DOUBLE, [0.1, None])})
    probe.setup_method()
    try:
        frame = probe.create_test_data(data, arrow_types={"v": pa.float32()})
        assert probe.extract_column(frame, "v") == [0.10000000149011612, None]
    finally:
        probe.teardown_method()


def test_python_dict_setup_never_builds_arrow(monkeypatch: pytest.MonkeyPatch) -> None:
    import mloda._testing.framework_adapter as core_adapter

    from mloda.testing import framework_adapter

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Arrow was built")

    monkeypatch.setattr(helpers, "arrow_table_of", boom, raising=False)
    monkeypatch.setattr(framework_adapter, "to_arrow_table", boom)
    monkeypatch.setattr(core_adapter, "to_arrow_table", boom)
    probe = _probe(_mixin("PythonDictTestMixin", None))
    probe.setup_method()
    try:
        assert isinstance(probe.test_data, dict)
        assert probe.test_data == DataOperationsTestDataCreator.typed_data().to_python_dict()
    finally:
        probe.teardown_method()


# -- Connection ownership ---------------------------------------------------------


def test_sqlite_teardown_closes_the_connection_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _probe(_mixin("SqliteTestMixin", None))
    probe.setup_method()
    conn = probe.conn
    calls: list[int] = []
    original = probe.adapter.close

    def counting_close() -> None:
        calls.append(1)
        original()

    monkeypatch.setattr(probe.adapter, "close", counting_close)
    probe.teardown_method()
    assert len(calls) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_base_teardown_leaves_the_connection_to_the_adapter() -> None:
    closed: list[str] = []

    class _Conn:
        def close(self) -> None:
            closed.append("conn")

    stub = _StubTeardownBase()
    stub.conn = _Conn()  # type: ignore[attr-defined]
    stub.adapter = object()  # type: ignore[attr-defined]
    stub.teardown_method()
    assert closed == []


def _timezone(cursor: Any) -> str:
    return str(cursor.execute("SELECT current_setting('TimeZone')").fetchall()[0][0])


class TestDuckdbTestConnection:
    @pytest.fixture(autouse=True)
    def _require_duckdb(self) -> None:
        pytest.importorskip("duckdb")

    def test_cursors_are_distinct_on_the_same_database(self) -> None:
        first = duckdb_test_connection()
        second = duckdb_test_connection()
        try:
            assert first is not second
            first.execute("CREATE TABLE shared_db_visibility_probe (x INTEGER)")
            first.execute("INSERT INTO shared_db_visibility_probe VALUES (1)")
            assert second.execute("SELECT x FROM shared_db_visibility_probe").fetchall() == [(1,)]
        finally:
            first.execute("DROP TABLE IF EXISTS shared_db_visibility_probe")
            first.close()
            second.close()

    def test_session_settings_do_not_leak_between_cursors(self) -> None:
        first = duckdb_test_connection()
        second = duckdb_test_connection()
        try:
            first.execute("SET TimeZone='Asia/Kolkata'")
            assert _timezone(first) == "Asia/Kolkata"
            assert _timezone(second) == "UTC"
        finally:
            first.close()
            second.close()

    def test_utc_pin_applies_when_database_default_is_not_utc(self) -> None:
        admin = duckdb_test_connection()
        try:
            admin.execute("SET GLOBAL TimeZone='Asia/Tokyo'")
            fresh = duckdb_test_connection()
            try:
                assert _timezone(fresh) == "UTC"
            finally:
                fresh.close()
        finally:
            admin.execute("RESET GLOBAL TimeZone")
            admin.close()


class _MixinProbe(DuckdbTestMixin, DataOpsTestBase):
    @classmethod
    def implementation_class(cls) -> object:
        return object


class TestDuckdbTestMixinLifecycle:
    @pytest.fixture(autouse=True)
    def _require_duckdb(self) -> None:
        pytest.importorskip("duckdb")

    def test_teardown_closes_cursor_and_database_stays_usable(self) -> None:
        probe = _MixinProbe()
        probe.setup_method()
        try:
            assert _timezone(probe.conn) == "UTC"
        finally:
            probe.teardown_method()
        duckdb = pytest.importorskip("duckdb")
        with pytest.raises(duckdb.Error):
            probe.conn.execute("SELECT 1")

        follow_up = duckdb_test_connection()
        try:
            assert follow_up.execute("SELECT 1").fetchall() == [(1,)]
        finally:
            follow_up.close()
