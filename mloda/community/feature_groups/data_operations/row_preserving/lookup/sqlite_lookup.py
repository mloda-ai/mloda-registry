"""SQLite implementation of key-lookup."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

import pyarrow as pa
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name, quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.lookup.base import (
    LookupFeatureGroup,
    assert_key_family,
    build_unmatched_error,
    key_family,
    value_family,
)
from mloda.community.feature_groups.data_operations.row_preserving.lookup.pyarrow_lookup import (
    ARROW_VALUE_TYPES,
    arrow_source_family,
)

_AFFINITIES: dict[str, str] = {"str": "TEXT", "int": "INTEGER", "float": "REAL", "bool": "INTEGER"}


class SqliteLookup(Implementation[LookupFeatureGroup]):
    """SQLite backend: the mapping goes into a temp table via ``executemany``, LEFT JOINed on the key."""

    feature_group = LookupFeatureGroup
    compute_framework = SqliteFramework

    @classmethod
    def calculate(
        cls, group: type[LookupFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.lookup(data, features, compute_lookup=cls._compute_lookup)

    @classmethod
    def _null_result(
        cls,
        data: SqliteRelation,
        original_cols: list[str],
        original_types: list[pa.DataType],
        feature_name: str,
        mapping: Mapping[Any, Any],
    ) -> SqliteRelation:
        out_table = f"_lookup_out_{uuid.uuid4().hex}"
        select_cols = ", ".join(quote_ident(c) for c in original_cols)
        data.connection.execute(
            f"CREATE TEMP TABLE {quote_ident(out_table)} AS "  # nosec
            f"SELECT {select_cols}, NULL AS {quote_ident(feature_name)} FROM {quote_ident(data.table_name)} "
            f"ORDER BY rowid"
        )
        value_type = ARROW_VALUE_TYPES[value_family(mapping)]
        return SqliteRelation(data.connection, out_table, _types=[*original_types, value_type])

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
        if pa.types.is_null(source_type):
            return cls._null_result(data, original_cols, original_types, feature_name, mapping)
        assert_key_family(feature_name, source_col, mapping, arrow_source_family(source_type), str(source_type))

        conn = data.connection
        key_fam = key_family(mapping)
        value_fam = value_family(mapping)
        value_type = ARROW_VALUE_TYPES[value_fam]
        map_table = f"_lookup_map_{uuid.uuid4().hex}"
        out_table = f"_lookup_out_{uuid.uuid4().hex}"
        rn = pick_helper_column_name(taken=set(original_cols) | {feature_name})
        q_map = quote_ident(map_table)
        q_source = quote_ident(source_col)

        conn.execute(f"CREATE TEMP TABLE {q_map} (k {_AFFINITIES[key_fam]} PRIMARY KEY, v {_AFFINITIES[value_fam]})")
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
