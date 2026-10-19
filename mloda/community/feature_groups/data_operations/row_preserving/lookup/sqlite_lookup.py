"""SQLite implementation of key-lookup."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

import pyarrow as pa
from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name, quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.lookup.base import (
    LookupFeatureGroup,
    assert_key_family,
    build_unmatched_error,
)
from mloda.community.feature_groups.data_operations.row_preserving.lookup.pyarrow_lookup import (
    _source_family,
)


def _affinity(arrow_type: pa.DataType) -> str:
    if pa.types.is_integer(arrow_type) or pa.types.is_boolean(arrow_type):
        return "INTEGER"
    if pa.types.is_floating(arrow_type):
        return "REAL"
    return "TEXT"


class SqliteLookup(LookupFeatureGroup):
    """SQLite backend: the mapping goes into a temp table via ``executemany``, LEFT JOINed on the key."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SqliteFramework}

    @classmethod
    def _compute_lookup(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        mapping: Mapping[Any, Any],
        policy: str,
        reasons: Mapping[Any, str],
    ) -> SqliteRelation:
        original_cols = list(data.columns)
        original_types = list(data.types)
        source_type = original_types[original_cols.index(source_col)]
        assert_key_family(feature_name, source_col, mapping, _source_family(source_type))

        conn = data.connection
        key_type = pa.array(list(mapping.keys())).type
        value_type = pa.array(list(mapping.values())).type
        map_table = f"_lookup_map_{uuid.uuid4().hex}"
        out_table = f"_lookup_out_{uuid.uuid4().hex}"
        rn = pick_helper_column_name(taken=set(original_cols) | {feature_name})
        q_map = quote_ident(map_table)
        q_source = quote_ident(source_col)

        conn.execute(f"CREATE TEMP TABLE {q_map} (k {_affinity(key_type)} PRIMARY KEY, v {_affinity(value_type)})")
        tagged = None
        try:
            tagged = data.with_row_number(rn, order_by=["rowid"])
            join_sql = f"FROM {quote_ident(tagged.table_name)} AS s LEFT JOIN {q_map} AS m ON s.{q_source} = m.k"  # nosec
            conn.executemany(f"INSERT INTO {q_map} VALUES (?, ?)", list(mapping.items()))  # nosec

            if policy == "raise":
                cursor = conn.execute(
                    f"SELECT s.{q_source}, COUNT(*) {join_sql} "  # nosec
                    f"WHERE s.{q_source} IS NOT NULL AND m.k IS NULL GROUP BY s.{q_source}"
                )
                found = cursor.fetchall()
                if found:
                    raise build_unmatched_error(feature_name, sum(n for _, n in found), [k for k, _ in found], reasons)

            result_expr = f"COALESCE(m.v, s.{q_source})" if policy == "keep" else "m.v"
            select_cols = ", ".join(f"s.{quote_ident(c)}" for c in original_cols)
            conn.execute(
                f"CREATE TEMP TABLE {quote_ident(out_table)} AS "  # nosec
                f"SELECT {select_cols}, {result_expr} AS {quote_ident(feature_name)} {join_sql} "
                f"ORDER BY s.{quote_ident(rn)}"
            )
        finally:
            conn.execute(f"DROP TABLE IF EXISTS {q_map}")
            if tagged is not None:
                tagged.drop()
        return SqliteRelation(conn, out_table, _types=[*original_types, value_type])
