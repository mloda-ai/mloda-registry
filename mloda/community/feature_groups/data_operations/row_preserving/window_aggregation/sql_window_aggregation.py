"""SQL (sqlite3) implementation for window aggregation feature groups.

SQL parameterization follows PEP 249 (DB-API 2.0):
- All VALUES use qmark paramstyle (?) placeholders via cursor.execute(sql, params).
- SQL identifiers (column names, table names) cannot be parameterized per the
  SQL standard. They are validated and quoted using _quote_identifier() which
  escapes embedded double-quotes per SQLite rules.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Set, Type, Union

import pyarrow as pa

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)

# Aggregation types that SQLite supports natively in window functions.
_SQL_NATIVE_AGGS: dict[str, str] = {
    "sum": "SUM",
    "avg": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
}

# Aggregation types that require a Python fallback (no native SQLite window support).
_PYTHON_FALLBACK_AGGS = {"std", "var", "median", "mode", "nunique", "first", "last"}

# Pattern for safe SQL identifiers (alphanumeric, underscore, space, dot).
_SAFE_IDENTIFIER_RE = re.compile(r"^[\w. ]+$")


def _validate_sql_identifier(name: str) -> None:
    """Validate a name is safe for use as a SQL identifier.

    Rejects names containing characters that could enable SQL injection
    when used inside double-quoted identifiers.
    """
    if not _SAFE_IDENTIFIER_RE.match(name):
        raise ValueError(f"Column name contains unsafe characters for SQL: {name!r}")


def _quote_identifier(name: str) -> str:
    """Quote a SQL identifier per SQLite double-quote rules.

    Embedded double-quotes are escaped by doubling them.
    Validates the identifier first.
    """
    _validate_sql_identifier(name)
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


class SqlWindowAggregation(WindowAggregationFeatureGroup):
    """SQL-based implementation of window aggregation (group-by with broadcast).

    Uses an in-memory sqlite3 database to execute SQL window functions.
    Accepts and returns PyArrow tables (workaround until a dedicated SQLite
    compute framework exists).

    All value insertion uses PEP 249 parameterized queries (qmark style).
    SQL identifiers are validated and quoted, not parameterized (SQL standard
    does not support parameterized identifiers).
    """

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}

    @classmethod
    def _compute_window(
        cls,
        table: pa.Table,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
    ) -> pa.Table:
        """Delegate to SQL path for native aggs, Python fallback otherwise."""
        if agg_type in _SQL_NATIVE_AGGS:
            return cls._compute_window_sql(table, feature_name, source_col, partition_by, agg_type)
        return cls._compute_window_python(table, feature_name, source_col, partition_by, agg_type)

    @classmethod
    def _compute_window_sql(
        cls,
        table: pa.Table,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
    ) -> pa.Table:
        """Execute the aggregation as a SQL window function via sqlite3.

        PEP 249 compliance:
        - CREATE TABLE: identifiers validated and quoted, no values to parameterize.
        - INSERT: uses qmark (?) placeholders for all row values.
        - SELECT: identifiers validated and quoted, aggregate function from whitelist.
        """
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()

        col_names = table.column_names
        num_rows = table.num_rows

        # DDL: identifiers are quoted, no user values involved.
        col_defs = ["_row_order INTEGER"]
        for col in col_names:
            col_defs.append(f"{_quote_identifier(col)} TEXT")
        create_sql = f"CREATE TABLE t ({', '.join(col_defs)})"
        cursor.execute(create_sql)

        # DML: PEP 249 qmark paramstyle for all values.
        placeholders = ", ".join(["?"] * (len(col_names) + 1))
        insert_sql = f"INSERT INTO t VALUES ({placeholders})"  # nosec B608
        rows_to_insert: list[list[Any]] = []
        for i in range(num_rows):
            row_values: list[Any] = [i]
            for col in col_names:
                val = table.column(col)[i].as_py()
                row_values.append(None if val is None else str(val))
            rows_to_insert.append(row_values)
        cursor.executemany(insert_sql, rows_to_insert)

        # Query: aggregate function from whitelist, identifiers validated and quoted.
        sql_func = _SQL_NATIVE_AGGS[agg_type]
        quoted_source = _quote_identifier(source_col)
        partition_clause = ", ".join(_quote_identifier(col) for col in partition_by)
        quoted_feature = _quote_identifier(feature_name)
        window_sql = (
            f"SELECT *, {sql_func}({quoted_source}) OVER (PARTITION BY {partition_clause}) "  # nosec B608
            f"AS {quoted_feature} FROM t ORDER BY _row_order"
        )
        cursor.execute(window_sql)
        rows = cursor.fetchall()
        conn.close()

        # The result column is the last column in each row.
        result_col_index = len(col_names) + 1  # +1 for _row_order
        raw_results = [row[result_col_index] for row in rows]

        # Convert results back to appropriate Python types.
        result_values = cls._convert_sql_results(raw_results, agg_type)

        new_col = pa.array(result_values)
        return table.append_column(feature_name, new_col)

    @classmethod
    def _convert_sql_results(cls, raw_results: list[Any], agg_type: str) -> list[Any]:
        """Convert raw SQL string results to proper Python numeric types."""
        converted: list[Any] = []
        for val in raw_results:
            if val is None:
                converted.append(None)
            elif agg_type in ("avg",):
                converted.append(float(val))
            elif agg_type in ("count",):
                converted.append(int(val))
            elif agg_type in ("sum", "min", "max"):
                # Detect whether the value should be int or float.
                float_val = float(val)
                if float_val == int(float_val) and "." not in str(val):
                    converted.append(int(float_val))
                else:
                    converted.append(float_val)
            else:
                converted.append(val)
        return converted

    # -- Python fallback for unsupported SQL aggregations --

    @classmethod
    def _compute_window_python(
        cls,
        table: pa.Table,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
    ) -> pa.Table:
        """Compute aggregation using dict-based Python approach (fallback)."""
        num_rows = table.num_rows

        # Build group keys per row.
        keys: list[tuple[Any, ...]] = []
        for i in range(num_rows):
            key = tuple(table.column(col)[i].as_py() for col in partition_by)
            keys.append(key)

        # Collect source values per group.
        groups: dict[tuple[Any, ...], list[Any]] = {}
        for i in range(num_rows):
            key = keys[i]
            val = table.column(source_col)[i].as_py()
            if key not in groups:
                groups[key] = []
            groups[key].append(val)

        # Compute aggregate per group.
        agg_results: dict[tuple[Any, ...], Any] = {}
        for key, values in groups.items():
            agg_results[key] = cls._aggregate(values, agg_type)

        # Broadcast back to every row.
        result_values = [agg_results[keys[i]] for i in range(num_rows)]

        new_col = pa.array(result_values)
        return table.append_column(feature_name, new_col)
