"""Behavioral tests for the shared-database DuckDB test connection."""

from __future__ import annotations

from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin, duckdb_test_connection


def _timezone(cursor: Any) -> str:
    return str(cursor.execute("SELECT current_setting('TimeZone')").fetchall()[0][0])


class TestDuckdbTestConnection:
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
    def test_teardown_closes_cursor_and_database_stays_usable(self) -> None:
        probe = _MixinProbe()
        probe.setup_method()
        try:
            assert _timezone(probe.conn) == "UTC"
        finally:
            probe.teardown_method()
        with pytest.raises(duckdb.Error):
            probe.conn.execute("SELECT 1")

        follow_up = duckdb_test_connection()
        try:
            assert follow_up.execute("SELECT 1").fetchall() == [(1,)]
        finally:
            follow_up.close()
