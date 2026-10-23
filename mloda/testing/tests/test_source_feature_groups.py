"""Hosts for the source-group contract mixins in mloda.testing.source_feature_groups."""

from __future__ import annotations

import csv
import sqlite3
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pyarrow as pa
from mloda.provider import FeatureSet, ReadFileFG
from mloda_plugins.feature_group.input_data.db_sources.sqlite_fg import SqliteFG

from mloda.testing.source_feature_groups import ReadDBFGTestMixin, ReadFileFGTestMixin

SCHEMA_MARKER = "#sensor-schema"


class SensorCsvFG(ReadFileFG):
    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return (".sensorcsv",)

    @classmethod
    def column_names(cls, path: str) -> Collection[str]:
        with open(path, encoding="utf-8") as handle:
            header = handle.readline()
        if SCHEMA_MARKER not in header:
            raise ValueError("its header lacks the #sensor-schema marker")
        return header.replace(SCHEMA_MARKER, "").strip().split(",")

    @classmethod
    def load_neutral(cls, match: Any, features: FeatureSet) -> Any:
        with open(match.access, encoding="utf-8", newline="") as handle:
            names = handle.readline().replace(SCHEMA_MARKER, "").strip().split(",")
            rows = list(csv.reader(handle))
        wanted = sorted(features.get_all_names())
        return pa.table({name: [int(row[list(names).index(name)]) for row in rows] for name in wanted})


def write_sqlite(path: Path, tables: dict[str, dict[str, list[Any]]]) -> None:
    """Create a SQLite database holding one table per entry, each with exactly the given columns."""
    declared = {int: "INTEGER", float: "REAL", str: "TEXT", bytes: "BLOB"}
    connection = sqlite3.connect(path)
    try:
        for table, columns in tables.items():
            names = list(columns)
            quoted_table = '"' + table.replace('"', '""') + '"'
            quoted = ['"' + name.replace('"', '""') + '"' for name in names]
            types = [
                declared[type(next(value for value in columns[name] if value is not None))]
                if any(value is not None for value in columns[name])
                else ""
                for name in names
            ]
            connection.execute(f"CREATE TABLE {quoted_table} ({', '.join(f'{q} {t}' for q, t in zip(quoted, types))})")
            placeholders = ", ".join("?" for _ in names)
            insert = f"INSERT INTO {quoted_table} VALUES ({placeholders})"  # nosec B608 - quoted names, bound values
            connection.executemany(insert, list(zip(*(columns[name] for name in names))))
        connection.commit()
    finally:
        connection.close()


class TestSensorCsvFGContract(ReadFileFGTestMixin):
    """The example ReadFileFG from guide 27 running the ReadFileFG contract."""

    feature_group_class = SensorCsvFG
    present_column = "sensor_present"
    missing_column = "sensor_missing"

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        names = list(columns)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(SCHEMA_MARKER + ",".join(names) + "\n")
            csv.writer(handle).writerows(zip(*(columns[name] for name in names)))

    def write_corrupt_file(self, path: Path) -> None:
        path.write_text("a,b\n1,2\n", encoding="utf-8")


class TestSqliteFGContract(ReadDBFGTestMixin):
    """Runs the ported ReadDBFG contract against core's stock SqliteFG to prove the shipped mixin works."""

    feature_group_class = SqliteFG
    credential_key = "sqlite"
    present_column = "impl_sqlite_present"
    missing_column = "impl_sqlite_missing"

    def write_database(self, path: Path, tables: dict[str, dict[str, list[Any]]]) -> None:
        write_sqlite(path, tables)
